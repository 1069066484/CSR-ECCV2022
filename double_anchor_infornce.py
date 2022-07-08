import torch
import torch.nn as nn
import torch.nn.functional as F


class TripletLoss(nn.Module):
    """
    Triplet loss that supports efficient triplet sampling
    """
    def __init__(self, d=2, n=10000):
        super(TripletLoss, self).__init__()
        self.n = n
        self.d = d

    def calc_with_d(self, pos_distance, negative_distance, m=0.3):
        # print(pos_distance.shape, negative_distance.shape); exit() # torch.Size([16]) torch.Size([16, 16])
        bs = pos_distance.size(0)
        triplet_loss = pos_distance - negative_distance   # bs x bs x num_vec
        # print('TripletLoss.forward.triplet_loss', triplet_loss)
        triplet_loss = triplet_loss + m
        eye = torch.eye(bs).to(pos_distance.device)
        triplet_loss = triplet_loss * (1 - eye)
        triplet_loss = F.relu(triplet_loss)[:,:self.n]
        triplet_loss = torch.mean(triplet_loss, dim=1)   #** 0.5
        return torch.mean(triplet_loss)

    def forward(self, sketch, photo, m=0.3):
        if self.d == 2:
            pos_distance = sketch - photo
            pos_distance = torch.pow(pos_distance, 2)
            pos_distance = torch.sqrt(torch.sum(pos_distance, dim=1))
            sketch_self = sketch.unsqueeze(0)
            photo_T = photo.unsqueeze(1)
            negative_distance = sketch_self - photo_T
            negative_distance = torch.pow(negative_distance, 2)
            negative_distance = torch.sqrt(torch.sum(negative_distance, dim=2))
        elif self.d == 1:
            sketch = sketch / torch.norm(sketch.float(), 2, -1)[:,None]
            photo = photo / torch.norm(photo.float(), 2, -1)[:,None]
            pos_distance = 1 - (sketch * photo).sum(-1)
            sketch_self = sketch.unsqueeze(0)
            photo_T = photo.unsqueeze(1)
            negative_distance = 1 - (sketch_self * photo_T).sum(-1)
        elif self.d == 0:
            pos_distance =  - (sketch * photo).sum(-1)
            sketch_self = sketch.unsqueeze(0)
            photo_T = photo.unsqueeze(1)
            negative_distance =  - (sketch_self * photo_T).sum(-1)
        else:
            raise Exception("Bad d in TripletLoss")
        return self.calc_with_d(pos_distance, negative_distance, m)


class SingleAnchorInfoNCE(nn.Module):
    def __init__(self, temperature=1, dist_type=0):
        super(SingleAnchorInfoNCE, self).__init__()
        self.criterion = torch.nn.CrossEntropyLoss()
        self.temperature = temperature
        self.dist_type = dist_type

    def _make_distance(self, features, features1=None, dist_type=None):
        if dist_type is None:
            dist_type = self.dist_type
        if features1 is None:
            features1 = features
        if dist_type == 0:
            features0 = features1.unsqueeze(0)
            features1 = features1.unsqueeze(1)
            d = features0 - features1
            d = torch.pow(d, 2)
            d = torch.sqrt(torch.sum(d, dim=2))
            return d
        elif dist_type == 1:
            return -features @ features1.T
        else:
            features = F.normalize(features)
            features1 = F.normalize(features1)
            return -features @ features1.T

    def forward(self, sk, im):
        """
        :param sk/im:  [batch_size, dims].
        """
        features = torch.stack([sk, im], 1)
        batch_size, n_views, dims = features.shape
        features_ = features.reshape([-1, dims])
        distances = -self._make_distance(features_) / self.temperature
        masks = torch.eye(batch_size, requires_grad=False).unsqueeze(0).repeat(n_views,1,1).\
            reshape(n_views, batch_size**2).T.reshape(batch_size, batch_size*n_views).to(features.device).bool()
        masks_exp = masks.repeat_interleave(n_views, dim=0)
        distances_pos = distances[masks_exp]

        # print(distances.shape, masks_exp.shape, distances_pos.shape)
        distances_neg = distances[~masks_exp].reshape([n_views * batch_size, -1]).repeat_interleave(n_views, dim=0)

        good_indices = ~torch.eye(n_views, requires_grad=False).to(masks_exp.device).reshape([-1]).bool().repeat(batch_size)
        distances_neg = distances_neg[:, 1::2]
        logits_cat = torch.cat([distances_pos.unsqueeze(-1), distances_neg], -1)[good_indices]
        logits_cat = logits_cat[::2]
        labels = torch.zeros([logits_cat.shape[0]], dtype=torch.long, requires_grad=False).to(features.device)
        return self.criterion(logits_cat, labels)



import math
class DoubleAnchorInfoNCE(nn.Module):
    def __init__(self, temperature=1, dist_type=0):
        super(DoubleAnchorInfoNCE, self).__init__()
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss()
        self.temperature = temperature
        self.dist_type = dist_type

    def criterion2(self, logits, logits2, labels, alpha=0):
        logits = torch.exp(logits.double()) + alpha * torch.exp(logits2.double())
        logits = logits / (torch.sum(logits, -1)[...,None] + 1e-10)
        loss = - (torch.log(logits.gather(1, labels[..., None]) + 1e-10) ).mean().float()
        return loss

    def _make_distance(self, features, features1=None, dist_type=None):
        if dist_type is None:
            dist_type = self.dist_type
        if features1 is None:
            features1 = features
        if dist_type == 0:
            features0 = features1.unsqueeze(0)
            features1 = features1.unsqueeze(1)
            d = features0 - features1
            d = torch.pow(d, 2)
            d = torch.sqrt(torch.sum(d, dim=2))
            return d
        elif dist_type == 1:
            return -features @ features1.T
        else:
            features = F.normalize(features)
            features1 = F.normalize(features1)
            return -features @ features1.T

    def make_logits(self, sk, im):
        """
        :param sk/im:  [batch_size, dims].
        :param batch_size: deprecated
        """
        features = torch.stack([sk, im], 1)
        batch_size, n_views, dims = features.shape
        features_ = features.reshape([-1, dims])
        distances = -self._make_distance(features_) / self.temperature
        masks = torch.eye(batch_size, requires_grad=False).unsqueeze(0).repeat(n_views,1,1).\
            reshape(n_views, batch_size**2).T.reshape(batch_size, batch_size*n_views).to(features.device).bool()
        masks_exp = masks.repeat_interleave(n_views, dim=0)
        distances_pos = distances[masks_exp]

        distances_neg = distances[~masks_exp].reshape([n_views * batch_size, -1]).repeat_interleave(n_views, dim=0)
        good_indices = ~torch.eye(n_views, requires_grad=False).to(masks_exp.device).reshape([-1]).bool().repeat(batch_size)
        distances_neg = distances_neg[:, 1::2]
        logits_cat = torch.cat([distances_pos.unsqueeze(-1), distances_neg], -1)[good_indices]
        logits_cat = logits_cat[::2]
        return logits_cat

    def forward(self, sk, sk2, im, alpha=0):
        logits_cat = self.make_logits(sk, im)
        labels = torch.zeros(
            [logits_cat.shape[0]], dtype=torch.long, requires_grad=False).to(sk.device)
        logits_cat2 = self.make_logits(sk2, im)
        return self.criterion2(logits_cat, logits_cat2, labels, alpha)


def _test():
    torch.random.seed()
    for i in range(10):
        sk = torch.rand(4,2)
        sk2 = torch.rand(4, 2)
        im = torch.rand(4, 2)
        print(SingleAnchorInfoNCE(dist_type=2, temperature=0.01)(sk, im),
              SingleAnchorInfoNCE(dist_type=2, temperature=0.01)(sk2, im),
            DoubleAnchorInfoNCE(dist_type=2, temperature=0.01)(sk, sk2, im, 0.5))


if __name__ == "__main__":
    _test()
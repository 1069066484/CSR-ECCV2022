from utils import *


class RecoveryNet(nn.Module):
    def __init__(self, in_dim, cat_dim28, cat_dim56, cp=False, bb='gn'):
        super(RecoveryNet, self).__init__()
        self.cp, self.cp_seq = init_cp(cp) # use checkpoint
        self.in_dim = in_dim
        self.leak = 0.
        self.bb = bb
        self.cat_dim28 = cat_dim28
        self.cat_dim56 = cat_dim56
        self.build_composer()
        self.loss_rec = nn.BCELoss()

    def build_composer(self):
        # ic3: 8, 17, 35, 71      gn: 7, 14, 28, 56
        if self.bb == 'gn' or self.bb == 'dn'or self.bb == 'r18':
            self.rec_net1 = nn.Sequential(
                Reshape(shape=[self.in_dim // 4, 2, 2]),  # 896 ch
                nn.Conv2d(in_channels=self.in_dim // 4, out_channels=256, kernel_size=1, stride=1, padding=0, bias=True),
                nn.BatchNorm2d(256), nn.LeakyReLU(negative_slope=self.leak, inplace=True),
                nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=3, stride=2, padding=0,
                                   output_padding=1, bias=True), nn.BatchNorm2d(256),
                nn.LeakyReLU(negative_slope=self.leak, inplace=True),  # 6 * 6
                nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=3, stride=2, padding=0,
                                   output_padding=1, bias=True), nn.BatchNorm2d(128),
                nn.LeakyReLU(negative_slope=self.leak, inplace=True),
                nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=3, stride=2, padding=1,
                                   output_padding=1, bias=True), nn.BatchNorm2d(128),
                nn.LeakyReLU(negative_slope=self.leak, inplace=True),  # 28 * 28
            )
            self.rec_net2 = nn.Sequential(
                nn.ConvTranspose2d(in_channels=128 + self.cat_dim28, out_channels=128, kernel_size=3, stride=2, padding=1,
                                   output_padding=1, bias=True), nn.BatchNorm2d(128),
                nn.LeakyReLU(negative_slope=self.leak, inplace=True),  # 56 * 56
                nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=True),
                nn.BatchNorm2d(128), nn.LeakyReLU(negative_slope=self.leak, inplace=True),
                nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True),
                nn.BatchNorm2d(64), FakeFn(self.sig4)  # 56 * 56
            )
        else:
            self.rec_net1 = nn.Sequential(
                Reshape(shape=[self.in_dim // 16, 4, 4]),  # 352 ch
                nn.Conv2d(in_channels=self.in_dim // 16, out_channels=256, kernel_size=1, stride=1, padding=0, bias=True),
                nn.BatchNorm2d(256), nn.LeakyReLU(negative_slope=self.leak, inplace=True),
                nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=3, stride=2, padding=0,
                                   output_padding=0, bias=True), nn.BatchNorm2d(256), # 9 * 9
                nn.LeakyReLU(negative_slope=self.leak, inplace=True),
                nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=3, stride=2, padding=1,
                                   output_padding=0, bias=True), nn.BatchNorm2d(128), # 17 * 17
                nn.LeakyReLU(negative_slope=self.leak, inplace=True),
                nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=3, stride=2, padding=0,
                                   output_padding=0, bias=True), nn.BatchNorm2d(128),
                nn.LeakyReLU(negative_slope=self.leak, inplace=True),  # 35 * 35
            )
            self.rec_net2 = nn.Sequential(
                nn.ConvTranspose2d(in_channels=128 + self.cat_dim28, out_channels=128, kernel_size=3, stride=2, padding=0,
                                   output_padding=0, bias=True), nn.BatchNorm2d(128), # 71 * 71
                nn.LeakyReLU(negative_slope=self.leak, inplace=True),
                nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=2, bias=True), # 73 * 73
                nn.BatchNorm2d(128), nn.LeakyReLU(negative_slope=self.leak, inplace=True),
                nn.LeakyReLU(negative_slope=self.leak, inplace=True),
                nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True),
                nn.BatchNorm2d(64), FakeFn(self.sig4)  # 73 * 73
            )
        self.rec_net3 = nn.Sequential(
            nn.Conv2d(in_channels=64 + self.cat_dim56, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(64), nn.LeakyReLU(negative_slope=self.leak, inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(64), nn.LeakyReLU(negative_slope=self.leak, inplace=True),
            nn.Conv2d(in_channels=64, out_channels=4, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(4), nn.Sigmoid()
        )

    def sig4(self, x):
            x1 = x[:, :4]
            x2 = x[:, 4:]
            x1 = torch.sigmoid(x1)
            x2 = F.leaky_relu(x2, negative_slope=self.leak)
            return torch.cat([x1, x2], dim=1)

    def _loss_map(self, x, target):
        x = torch.clamp(x, 0.0, 1.0)
        loss_rec = self.loss_rec(x,target)
        return loss_rec, 0, 0

    def loss(self, in_x, add_x28, add_x56, target):
        inter_out = self.cp(self.rec_net1, in_x)
        out1 = self.rec_net2(torch.cat([inter_out, add_x28], dim=1))
        out2 = self.rec_net3(torch.cat([out1, add_x56], dim=1))
        rec_loss2, overlap2, constraint2 = self._loss_map(out2, target)
        return rec_loss2, overlap2, constraint2

    def forward(self, in_x, add_x28, add_x56):
        inter_out = self.cp(self.rec_net1, in_x)
        out1 = self.rec_net2(torch.cat([inter_out, add_x28], dim=1))
        out2 = self.rec_net3(torch.cat([out1, add_x56], dim=1))
        return out2


def init_cp(cp):
    cp_ = checkpoint if cp else lambda f, x: f(x)
    cp_seq_ = checkpoint_sequential if cp else lambda f, s, x: f(x)
    return cp_, cp_seq_


from double_anchor_infornce import *
from torch.optim import Adam


class CSRNet(nn.Module):
    ITEM_NAMES = {
        'loss_da': 1.0,
        'loss_rec': 1.0,
        'rec_mask0': 1.0,
        'rec_mask1': 1.0,
        'rec_mask2': 1.0,
        'rec_mask3': 1.0,
    }
    def bb2sizes(bb):
        if bb == 'gn':
            return {"in": 224, "rec_c": 28, "rec_f": 56}
        elif bb == 'ic3':
            return {"in": 299, "rec_c": 35, "rec_f": 73}
        else:
            # resnet18
            return {"in": 224, "rec_c": 28, "rec_f": 56}
    def __init__(self,  logger=None,  args=None):
        super(CSRNet,  self).__init__()
        self.logger = logger
        self.args = args
        if args.add_ch == 0:
            args.fusion = 0
        self.loss_num = 10

        if args.bb == 'gn':
            self.print("Init network googlenet")
            self.feat_extractor = models.googlenet(args.imagenet)
            self.feat_extractor.feat_dim_ori = 1024
            cat_dim28, cat_dim56 = 192, 64
        elif args.bb == 'ic3':
            self.print("Init network inception_v3")
            self.feat_extractor = models.inception_v3(args.imagenet)
            self.feat_extractor.feat_dim_ori = 2048
            cat_dim28, cat_dim56 = 192, 64
        elif args.bb == 'r18':
            self.print("Init network resnet_18")
            self.feat_extractor = models.resnet18(args.imagenet)
            self.feat_extractor.feat_dim_ori = 512
            cat_dim28, cat_dim56 = 128, 64
        else:
            self.print("Init network densenet169")
            self.feat_extractor = models.densenet169(args.imagenet, memory_efficient=True)
            self.feat_extractor.feat_dim_ori = 1664
            cat_dim28, cat_dim56 = 512, 256

        self.adj_step = 1
        self.cp, self.cp_seq = init_cp(self.args.cp)
        self.feat_extractor.feat_dim = self.feat_extractor.feat_dim_ori + self.args.add_ch * 3
        self.recovery_net = RecoveryNet(self.feat_extractor.feat_dim * 2,
                                 cat_dim28=cat_dim28,
                                 cat_dim56=cat_dim56,
                                 cp=args.cp, bb=args.bb)

        self.leak = 0.
        self.dummy = DummyLayer()
        self._build_feature_extractor()
        self.double_anchor_infonce = DoubleAnchorInfoNCE(temperature=self.args.tau, dist_type=2)
        self.weights = args.weights if isinstance(args.weights, dict) else eval(args.weights)

        params = [self.feat_extractor, self.recovery_net,
                  self.multi_level_extractor1, self.multi_level_extractor2,
                  self.multi_level_extractor3, self.dummy]
        self.params = params
        self.opt = Adam(sum([list(m.parameters()) for m in params], []), lr=args.lr)
        for s in CSRNet.ITEM_NAMES:
            if s not in self.weights:
                self.weights[s] = CSRNet.ITEM_NAMES[s]
        self.weights["rec_mask"] = [self.weights["rec_mask{}".format(i)] for i in range(4)]
        self.print("\n\nnum_params: {}\topt_params: {} \ninput: {} \nweights: {}\n\n".format(
            num_params(self) ,len(list(self.parameters())), self.args.weights, self.weights))

    def _build_feature_extractor(self):
        if self.args.bb == 'gn':
            in_ch1 = 192
            in_ch2 = 480
            in_ch3 = 832
        elif self.args.bb == 'ic3':
            in_ch1 = 192
            in_ch2 = 768
            in_ch3 = 1280
        elif self.args.bb == 'r18':
            in_ch1 = 64
            in_ch2 = 128
            in_ch3 = 256
        else:
            in_ch1 = 128
            in_ch2 = 256
            in_ch3 = 640
        self.multi_level_extractor1 = nn.Sequential(
            nn.Conv2d(in_ch1, 192, 3),
            nn.BatchNorm2d(192),
            nn.LeakyReLU(negative_slope=self.leak, inplace=True),
            nn.Conv2d(192, self.args.add_ch, 3),
            nn.BatchNorm2d(self.args.add_ch),
            nn.LeakyReLU(negative_slope=self.leak, inplace=True),
            FakeFn(lambda x: x.mean(-1).mean(-1))
        )

        self.multi_level_extractor2 = nn.Sequential(
            nn.Conv2d(in_ch2, 256, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(negative_slope=self.leak, inplace=True),
            nn.Conv2d(256, self.args.add_ch, 3),
            nn.BatchNorm2d(self.args.add_ch),
            nn.LeakyReLU(negative_slope=self.leak, inplace=True),
            FakeFn(lambda x: x.mean(-1).mean(-1))
        )

        self.multi_level_extractor3 = nn.Sequential(
            nn.Conv2d(in_ch3, self.args.add_ch, 1),
            nn.BatchNorm2d(self.args.add_ch),
            nn.LeakyReLU(negative_slope=self.leak, inplace=True),
            FakeFn(lambda x: x.mean(-1).mean(-1))
        )

    def _da_loss(self, sk, im, sk2):
        return self.double_anchor_infonce(sk=sk, sk2=sk2, im=im, alpha=self.args.alpha)

    def get_feats(self, x):
        cp = self.cp
        feats = []

        if self.args.bb == 'gn':
            x = cp(self.feat_extractor.conv1,x)
            x = cp(self.feat_extractor.maxpool1,x)
            x56 = x

            x = cp(self.feat_extractor.conv2,x)
            x = cp(self.feat_extractor.conv3,x)
            x = cp(self.feat_extractor.maxpool2,x)
            feats.append(cp(self.multi_level_extractor1,x))
            x28 = x

            x = cp(self.feat_extractor.inception3a,x)
            x = cp(self.feat_extractor.inception3b,x)
            x = cp(self.feat_extractor.maxpool3,x)
            feats.append(cp(self.multi_level_extractor2,x))

            x = cp(self.feat_extractor.inception4a,x)
            x = cp(self.feat_extractor.inception4b,x)
            x = cp(self.feat_extractor.inception4c,x)
            x = cp(self.feat_extractor.inception4d,x)
            x = cp(self.feat_extractor.inception4e,x)
            x = cp(self.feat_extractor.maxpool4,x)
            feats.append(cp(self.multi_level_extractor3,x))

            x = cp(self.feat_extractor.inception5a, x)
            x = cp(self.feat_extractor.inception5b, x)
            x = cp(self.feat_extractor.avgpool, x)
            x = torch.flatten(x, 1)
            feats.append(x)
        elif self.args.bb == 'ic3':
            # N x 3 x 299 x 299
            x = cp(self.feat_extractor.Conv2d_1a_3x3, x)
            x = cp(self.feat_extractor.Conv2d_2a_3x3, x)
            x = cp(self.feat_extractor.Conv2d_2b_3x3, x)
            x = torch.max_pool2d(x, kernel_size=3, stride=2)
            # N x 64 x 73 x 73

            x56 = x

            x = cp(self.feat_extractor.Conv2d_3b_1x1, x)
            x = cp(self.feat_extractor.Conv2d_4a_3x3, x)
            # x = cp(self.feat_extractor.maxpool2, x)
            x = torch.max_pool2d(x, kernel_size=3, stride=2)
            # N x 192 x 35 x 35
            feats.append(cp(self.multi_level_extractor1, x))
            x28 = x

            x = cp(self.feat_extractor.Mixed_5b, x)
            x = cp(self.feat_extractor.Mixed_5c, x)
            x = cp(self.feat_extractor.Mixed_5d, x)
            x = cp(self.feat_extractor.Mixed_6a, x)
            # N x 768 x 17 x 17
            feats.append(cp(self.multi_level_extractor2, x))

            x = cp(self.feat_extractor.Mixed_6b, x)
            x = cp(self.feat_extractor.Mixed_6c, x)
            x = cp(self.feat_extractor.Mixed_6d, x)
            x = cp(self.feat_extractor.Mixed_6e, x)
            x = cp(self.feat_extractor.Mixed_7a, x)
            # N x 1280 x 8 x 8
            feats.append(cp(self.multi_level_extractor3, x))

            x = cp(self.feat_extractor.Mixed_7b, x)
            x = cp(self.feat_extractor.Mixed_7c, x)
            x = F.adaptive_avg_pool2d(x, (1,1))
            x = torch.flatten(x, 1)
            # N x 2048
            feats.append(x)
        elif self.args.bb == 'r18':
            x = cp(self.feat_extractor.conv1, x)
            x = cp(self.feat_extractor.bn1, x)
            x = self.feat_extractor.maxpool(self.feat_extractor.relu(x))
            x56 = x

            x = cp(self.feat_extractor.layer1, x)
            feats.append(cp(self.multi_level_extractor1, x))

            x = cp(self.feat_extractor.layer2, x)
            x28 = x
            feats.append(cp(self.multi_level_extractor2, x))

            x = cp(self.feat_extractor.layer3, x)
            feats.append(cp(self.multi_level_extractor3, x))

            x = cp(self.feat_extractor.layer4, x)
            x = self.feat_extractor.avgpool(x)
            x = torch.flatten(x, 1)
            # N x 2048
            feats.append(x)
        else:
            x = self.feat_extractor.features[:5](x)
            x56 = x
            x = self.feat_extractor.features[5:6](x)
            feats.append(cp(self.multi_level_extractor1, x))
            x = self.feat_extractor.features[6:7](x)
            x28 = x
            x = self.feat_extractor.features[7:8](x)
            feats.append(cp(self.multi_level_extractor2, x))
            x = self.feat_extractor.features[8:10](x)
            feats.append(cp(self.multi_level_extractor3, x))
            x = self.feat_extractor.features[10:](x)
            x = F.adaptive_avg_pool2d(F.relu(x, inplace=True), (1, 1))
            x = torch.flatten(x, 1)
            feats.append(x)
        return torch.cat(feats, -1), x28, x56

    def chceck_params(self, depth=3):
        D = depth + 1
        NUM = num_params(self)
        def chceck_params(module, depth):
            if depth == 0: return None
            num = num_params(module)
            if num == 0: return None
            print('----' * (D - depth), " t:", type(module), " n:", num, " r:", round(num / NUM, 5))
            for child in module.children():
                chceck_params(child, depth - 1)
        chceck_params(self, depth)

    def print(self,  s):
        if self.logger is None:
            print(s)
        else:
            self.logger.info('{}'.format(s))

    def forward(self,  x):
        return self.get_feats(x)[0]

    def adjust_learning_rate(self, reset=False):  # .996
        self.adj_step += 1
        optimizer = self.opt
        if self.adj_step % 100 == 0:
            for param_group in optimizer.param_groups:
                lr = self.args.lr * math.pow(self.args.decay, float(self.adj_step) / self.args.steps)
                param_group['lr'] = lr
                self.print("learning_rate: lr:{}".format(lr))

    def _recovery_loss(self, condition_image, disordered_sketch, sk_rec, compensate_sk28, compensate_sk56):
        return self.recovery_net.loss(torch.cat([condition_image, disordered_sketch], dim=-1),
                                      compensate_sk28, compensate_sk56, sk_rec)

    def _optimize_params(self, sk, im, disordered, sk_recovery):
        rets = [0] * self.loss_num
        for i in range(len(self.weights["rec_mask"])):
            sk_recovery[:, i] = sk_recovery[:, i] * self.weights["rec_mask"][i]

        bs = sk.shape[0]
        feats_all, feats_all28, feats_all56 = self.get_feats(self.dummy(torch.cat([sk, im, disordered])))
        sk = feats_all[:bs]
        im = feats_all[bs:bs*2]
        compensate_sk = feats_all[bs*2:bs * 3]
        compensate_sk28 = feats_all28[bs * 2:bs * 3]
        compensate_sk56 = feats_all56[bs * 2:bs * 3]
        loss_da = self._da_loss(sk, im, compensate_sk)
        loss_rec = self._recovery_loss(im, compensate_sk, sk_recovery, compensate_sk28, compensate_sk56)

        rets_ = [loss_da, *loss_rec]
        self.adjust_learning_rate()
        self.opt.zero_grad()
        (
            loss_da * abs(self.weights['loss_da']) +
            sum(loss_rec) * abs(self.weights['loss_rec'])
        ).backward()
        self.opt.step()

        rets[:len(rets_)] = rets_
        for i in range(len(rets)):
            rets[i] = float(rets[i].item() if isinstance(rets[i], torch.Tensor) else rets[i])
        return rets

    def optimize_params(self,  sk, im, disordered, sk_recovery):
        return self._optimize_params(sk, im, disordered, sk_recovery)


from easydict import EasyDict as edict
def _test():
    args = edict()
    args.lr = 0.0002
    args.opt = 'Adam'
    args.tau = 0.005
    args.weights = "{'loss_da': 10.0, 'loss_rec': 1.0}"
    args.decay = 0.1
    args.steps = 100000
    args.cp = 1
    args.alpha = 0.3
    args.bb = 'gn'
    args.recovery_net = 1
    args.imagenet = True
    args.trp_d = 2
    args.add_ch = 128 
    afg = CSRNet(None, args).cuda()
    afg.chceck_params(2)

    bs = 2
    sk_ori = torch.rand(bs,3,CSRNet.bb2sizes(args.bb)['in'],CSRNet.bb2sizes(args.bb)['in']).cuda()
    im_ori = torch.rand(bs,3,CSRNet.bb2sizes(args.bb)['in'],CSRNet.bb2sizes(args.bb)['in']).cuda()
    sk = torch.rand(bs,3,CSRNet.bb2sizes(args.bb)['in'],CSRNet.bb2sizes(args.bb)['in']).cuda()
    label = torch.rand([bs, 4, CSRNet.bb2sizes(args.bb)['rec_f'], CSRNet.bb2sizes(args.bb)['rec_f']]).cuda()
    for i in range(100):
        print(i, afg.optimize_params(sk_ori, im_ori, sk, label))


if __name__=="__main__":
    _test()


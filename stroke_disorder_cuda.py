import glob
import math
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA
np.random.seed(0)

def pool2d(A, kernel_size, stride, padding, pool_mode='max'):
    """
    2d Pooling
    :param A: input 2D array
    :param kernel_size: int, the size of the window
    :param stride: int, the stride of the window
    :param padding: int, implicit zero paddings on both sides of the input
    :param pool_mode: string, 'max' or 'avg'
    :return: pooled matrix
    """
    if pool_mode == 'max':
        return F.max_pool2d(A[None,...].float(),kernel_size,stride,padding)[0]
    elif pool_mode == 'avg':
        return F.avg_pool2d(A[None,...].float(),kernel_size,stride,padding)[0]


import time
def group_pixel_into_strokes(mat, inf_val=255, edge=1):
    """
    Ln1 of the algorithm, group pixels into strokes
    :param mat: input sketch
    :param inf_val: background color
    :param edge: if distance between two pixels <= edge, then they belongs to the same stroke
    :return: 1) matrix[i][j] = s indicates that pixel(i, j) belongs to stroke s,
            2) the number of strokes
    """
    cls = torch.zeros([mat.shape[0], mat.shape[1]], dtype=torch.int16).cuda() - 1
    mat = torch.from_numpy(mat).cuda()
    curr = 0
    xs_stroke, ys_stroke = torch.where(mat != inf_val)
    for x, y in zip(xs_stroke, ys_stroke):
        cls[x][y] = torch.max(cls[x-edge: x+1, y-edge: y+edge+1])
        if cls[x][y] == -1:
            cls[x][y] = curr
            curr += 1

    pooled = torch.round(pool2d(cls, edge*2+1, 1, edge, 'max')).int()
    xs, ys = torch.where((cls != -1) * (pooled != cls))
    cls_np = cls.cpu().numpy()
    xs = xs.cpu().numpy()
    ys = ys.cpu().numpy()
    pooled = pooled.cpu().numpy()

    # there is an edge between cls[x][y] and pooled[x][y]
    # build parent via depth first searching
    edge_list = [[] for _ in range(curr)]
    for x, y in zip(xs, ys):
        edge_list[cls_np[x][y]].append(pooled[x][y])
        edge_list[pooled[x][y]].append(cls_np[x][y])

    parent = {}
    curr_parent = -1
    for c in range(curr):
        if c in parent:
            continue
        curr_parent += 1
        stack = [c]
        while len(stack):
            pop = stack.pop()
            if pop in parent:
                continue
            parent[pop] = curr_parent
            for to in edge_list[pop]:
                stack.append(to)

    curr_parent += 1
    for i in range(curr):
        cls[cls == i] = parent[i]

    return cls.cpu().numpy(), curr_parent


def extract_strokes(mat, inf_val=255, edge=1, t_target=10):
    """
    Ln2-9 of the algorithm
    :param mat: input sketch
    :param inf_val: background color
    :param edge: if distance between two pixels <= edge, then they belongs to the same stroke
    :param t_target: n_s
    :return: 1) matrix[i][j] = s indicates that pixel(i, j) belongs to stroke s,
            2) the number of strokes,
            3) visualization mat
    """
    cls, n = group_pixel_into_strokes(mat, inf_val, edge)
    mat_viss = [visual_cls(cls)]
    cls = torch.from_numpy(cls).cuda()
    mat = torch.from_numpy(mat).cuda()
    line_len = (mat.shape[0] + mat.shape[1]) // 20

    while n < t_target:
        group_max_pix = -1
        group_max_index = 0
        for i in range(0, n):
            sum_of_group_i = torch.sum(cls == i)
            if sum_of_group_i > group_max_pix:
                group_max_pix = sum_of_group_i
                group_max_index = i

        mat_max = (inf_val - inf_val * (cls == group_max_index)).int()
        edge_tmp = edge * 16

        pooled = pool2d(mat_max.float(), edge_tmp * 2 + 1, 1, 0, 'avg')
        max_xy = torch.argmin(pooled + mat_max[edge_tmp:-edge_tmp, edge_tmp:-edge_tmp])
        max_x = max_xy // pooled.shape[1] + edge_tmp
        max_y = max_xy % pooled.shape[1] + edge_tmp
        xs, ys = torch.where(mat_max[max_x-edge_tmp:max_x+edge_tmp+1, max_y-edge_tmp:max_y+edge_tmp+1] != inf_val)
        x = max_x.cpu().item()
        y = max_y.cpu().item()

        # the directions of maximum variance in the data
        direction = PCA(n_components=2).fit(torch.stack([xs, ys], -1).cpu().numpy()).components_[0]
        direction[1] = -direction[1]
        main_direction = direction
        direction = (main_direction * line_len).astype(np.int)

        thickness = 2

        pt1 = (y - direction[0], x - direction[1])
        pt2 = (y + direction[0], x + direction[1])

        mat_max = mat_max.cpu().numpy().astype(np.uint8)
        cv2.line(mat_max, pt1, pt2, color=(255,255,255), thickness=thickness)
        cls_curr, n_curr = group_pixel_into_strokes(mat_max, inf_val, edge)

        # visualization
        if 1:
            direction = (main_direction * line_len).astype(np.int)
            pt1 = (y - direction[0], x - direction[1])
            pt2 = (y + direction[0], x + direction[1])
            vis_cls = visual_cls(cls.cpu().numpy())
            cv2.line(vis_cls, pt1, pt2, color=(0,0,0), thickness=2)
            mat_viss.append(vis_cls)

        cls[cls == group_max_index] = -1
        cls[cls_curr == 0] = n
        for i in range(1, n_curr):
            cls[cls_curr == i] = n
            n += 1
    return cls.cpu().numpy(), n, mat_viss


def visual_single_ch(mat, name="1"):
    mi = np.min(mat)
    ma = np.max(mat)
    if mi == ma:
        print("BAD MAT {}".format(mat[0][0]))
        return
    mat = (mat - mi) / (ma - mi)
    mat = np.stack([mat] * 3, -1)
    cv2.imshow(name, mat)
    cv2.waitKey()


def bounding_box(arr, dst_pix=0):
    '''
    return (x_min, y_min, x_max, y_max)
    '''
    if len(arr.shape) == 3:
        arr = arr[:,:,0]
    arr = arr.float()
    x = torch.mean(arr, 1)
    y = torch.mean(arr, 0)
    x_idx = torch.where(x != dst_pix)[0]
    y_idx = torch.where(y != dst_pix)[0]
    if len(x_idx) == 0: x_idx = y_idx = [-1]
    return torch.min(x_idx).item(), torch.min(y_idx).item(), torch.max(x_idx).item() + 1, torch.max(y_idx).item() + 1


def stroke_disorder(mat, inf_val=255, edge=1, t_target=10, thresh=20, pd=0.1):
    """
    Stroke Disorder Algorithm
    :param mat: input sketch
    :param inf_val: background color
    :param edge: if distance between two pixels <= edge, then they belongs to the same stroke
    :param t_target: n_s
    :param thresh: we leave the strokes whose pixels <= thresh
    :param pd: p_d
    :return:1) matrix[i][j] = s indicates that pixel(i, j) belongs to stroke s, some strokes are disordered,
            2) the number of strokes,
            3) visualization mat
    """
    cls, n, mat_viss = extract_strokes(mat, inf_val=inf_val, edge=edge, t_target=t_target)
    # during training, output of extract_strokes are saved and reused

    cls = torch.from_numpy(cls).cuda()
    mat = torch.from_numpy(mat).cuda()

    counts = [torch.sum(cls == i) for i in range(n)]
    good_indices = [i for i in range(n) if counts[i] > thresh]

    selected = np.random.choice(good_indices, min(max(int(pd * n), 1), len(good_indices)), replace=False)
    arrs = []
    for i in selected:
        arr_curr = (cls == i)
        cls[arr_curr] = -1
        arr_curr = arr_curr.int()
        box = bounding_box(arr_curr, dst_pix=0)

        shape = np.array(mat.shape)

        # tx + x_max < mat.shape[0]
        # tx + x_min >= 0
        tx = np.clip(int(np.random.rand() * shape[0] * pd), a_min=-box[0], a_max=shape[0] - box[1] - 1)
        ty = np.clip(int(np.random.rand() * shape[1] * pd), a_min=-box[2], a_max=shape[1] - box[3] - 1)

        mat_trans = np.float32([[1, 0, tx], [0, 1, ty]])
        arr_curr = cv2.warpAffine(src=arr_curr.cpu().numpy().astype(np.uint8), M=mat_trans, dsize=[shape[1], shape[0]], borderValue=0)

        center_x = (box[0] + box[2]) * 0.5
        center_y = (box[1] + box[3]) * 0.5
        half_diag_len = 0.5 * math.sqrt((box[0] - box[2]) ** 2 + (box[1] - box[3]) ** 2) ** 0.5
        if 0 <= center_x + half_diag_len + tx < shape[0] and \
                0 <= center_y + half_diag_len + ty < shape[1]:
            tr = np.random.rand() * pd * pd * math.pi
            mat_rotation = cv2.getRotationMatrix2D([center_x + tx, center_y + ty], tr * 180 / math.pi, 1)
            arr_curr = cv2.warpAffine(arr_curr, mat_rotation, shape, borderValue=0)
        arrs.append(arr_curr)
        cls[arr_curr == 1] = i
    return cls, n, mat_viss


import colorsys
import random


def get_n_hls_colors(num):
    hls_colors = []
    i = 0
    step = 360.0 / num
    while i < 360:
        h = i
        s = 90 + random.random() * 10
        l = 50 + random.random() * 10
        _hlsc = [h / 360.0, l / 100.0, s / 100.0]
        hls_colors.append(_hlsc)
        i += step
    return hls_colors


def ncolors(num):
    rgb_colors = []
    if num < 1:
        return rgb_colors
    hls_colors = get_n_hls_colors(num)
    for hlsc in hls_colors:
        _r, _g, _b = colorsys.hls_to_rgb(hlsc[0], hlsc[1], hlsc[2])
        r, g, b = [int(x * 255.0) for x in (_r, _g, _b)]
        rgb_colors.append([r, g, b])

    return rgb_colors


colors = np.array(ncolors(20))


def visual_cls(cls):
    cls = np.stack([cls] * 3, -1)
    cls = cls.astype(np.int16)

    for x in range(cls.shape[0]):
        for y in range(cls.shape[1]):
            for i in range(3):
                if cls[x, y, i] >= 0:
                    cls[x, y, i] = colors[cls[x, y, i] % len(colors), i]
    cls[cls < 0] = 0
    return 255 - cls.astype(np.uint8)


def visual():
    for png in glob.glob('*.png'):
        sketch = cv2.imread(png)
        sketch = cv2.resize(sketch, (300,300))
        edge = 1

        cls5, n5, mat_viss5 = stroke_disorder(sketch[:,:,0], edge=edge, t_target=5)
        cls5 = visual_cls(cls5.cpu().numpy())
        cv2.imshow("mat_vis5", np.concatenate([cls5] + mat_viss5, 1))

        cls10, n10, mat_viss10 = stroke_disorder(sketch[:,:,0], edge=edge, t_target=10)
        cls10 = visual_cls(cls10.cpu().numpy())
        cv2.imshow("mat_vis10", np.concatenate([cls10] + mat_viss10, 1))
        cv2.waitKey()


if __name__=="__main__":
    visual()



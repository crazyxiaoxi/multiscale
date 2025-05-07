# dataset_multi_scale.py
import torch
from torch.utils.data import Dataset
from pathlib import Path
import cv2

def load_and_preprocess(img, size=512):
    # img: numpy array BGR
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (size, size))
    t = torch.from_numpy(img).permute(2,0,1).float() / 255.0
    return t

class MultiScaleDataset(Dataset):
    def __init__(self, img_dir, label_dir, scales=[2,3,5], patch_size=512):
        self.images = sorted(Path(img_dir).glob("*.jpg"))
        self.labels = sorted(Path(label_dir).glob("*.txt"))
        self.scales = scales
        self.patch_size = patch_size

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # 1. 读原图与标签
        img_path = self.images[idx]
        label_path = self.labels[idx]
        img_full = cv2.imread(str(img_path))
        H, W = img_full.shape[:2]

        labels = []
        with open(label_path) as f:
            for line in f:
                cls, xc, yc, w, h = map(float, line.split())
                labels.append([cls, xc, yc, w, h])
        labels = torch.tensor(labels)

        patches_s = []
        patches_m = []
        patches_l = []
        for scale in self.scales:
            h_step, w_step = H/scale, W/scale
            patches = []
            for i in range(scale):
                for j in range(scale):
                    y0, x0 = int(i*h_step), int(j*w_step)
                    y1, x1 = int((i+1)*h_step), int((j+1)*w_step)
                    crop = img_full[y0:y1, x0:x1]
                    patches.append(load_and_preprocess(crop, self.patch_size))
            if scale == self.scales[0]:
                patches_s = patches
            elif scale == self.scales[1]:
                patches_m = patches
            else:
                patches_l = patches
        targets = {
            "batch_idx": torch.zeros((labels.shape[0], 1), dtype=torch.float32),  # 因为 batch_size=1，全是0
            "cls": labels[:, 0:1],      # 类别
            "bboxes": labels[:, 1:],    # bboxes 是 xywh 格式，归一化过
        }
        return patches_s, patches_m, patches_l, targets

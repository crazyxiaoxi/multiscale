import torch
from torch.utils.data import Dataset
from pathlib import Path
import cv2


def load_and_preprocess(img, size=512):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (size, size))
    t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
    return t


class MultiScaleDataset(Dataset):
    def __init__(self, img_dir, label_dir, scales=[2, 3, 5], patch_size=512):
        self.images = sorted(Path(img_dir).glob("*.jpg"))
        self.labels = sorted(Path(label_dir).glob("*.txt"))
        self.scales = scales
        self.patch_size = patch_size

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        label_path = self.labels[idx]
        img_full = cv2.imread(str(img_path))
        H, W = img_full.shape[:2]

        # 读取标签
        labels = []
        with open(label_path) as f:
            for line in f:
                cls, xc, yc, w, h = map(float, line.strip().split())
                labels.append([cls, xc, yc, w, h])
        labels = torch.tensor(labels)

        patches_all = []
        targets_all = []

        for scale in self.scales:
            h_step, w_step = H / scale, W / scale
            patches = []
            targets = []

            # 生成所有patch并筛选
            for i in range(scale):
                for j in range(scale):
                    y0, x0 = int(i * h_step), int(j * w_step)
                    y1, x1 = int((i + 1) * h_step), int((j + 1) * w_step)

                    crop = img_full[y0:y1, x0:x1]
                    patch = load_and_preprocess(crop, self.patch_size)

                    patch_labels = []
                    for label in labels:
                        cls, xc, yc, w, h = label.tolist()
                        x_center = xc * W
                        y_center = yc * H
                        box_w = w * W
                        box_h = h * H

                        # 检查目标完整性
                        if not (x0 <= x_center <= x1 and y0 <= y_center <= y1):
                            continue
                        if (x_center - box_w / 2 < x0 or x_center + box_w / 2 > x1 \
                            or y_center - box_h / 2 < y0 or y_center + box_h / 2 > y1):
                            continue

                        new_xc = (x_center - x0) / (x1 - x0)
                        new_yc = (y_center - y0) / (y1 - y0)
                        new_w = box_w / (x1 - x0)
                        new_h = box_h / (y1 - y0)

                        patch_labels.append([cls, new_xc, new_yc, new_w, new_h])

                    # 仅保留包含目标的patch
                    if patch_labels:
                        patch_labels_tensor = torch.tensor(patch_labels)
                        patch_targets = {
                            "batch_idx": torch.zeros((patch_labels_tensor.shape[0], 1), dtype=torch.float32),
                            "cls": patch_labels_tensor[:, 0:1],
                            "bboxes": patch_labels_tensor[:, 1:]
                        }
                        patches.append(patch)
                        targets.append(patch_targets)

            # 如果该scale没有任何包含目标的patch，则降级为添加一个空目标patch
            if not patches:
                # 取左上角patch
                y1, x1 = int(h_step), int(w_step)
                crop = img_full[0:y1, 0:x1]
                patch = load_and_preprocess(crop, self.patch_size)
                patches.append(patch)
                # 对应空targets
                targets.append({
                    "batch_idx": torch.zeros((0, 1), dtype=torch.float32),
                    "cls": torch.zeros((0, 1)),
                    "bboxes": torch.zeros((0, 4))
                })

            patches_all.append(patches)
            targets_all.append(targets)

        return (
            patches_all[0], patches_all[1], patches_all[2],
            targets_all[0], targets_all[1], targets_all[2]
        )

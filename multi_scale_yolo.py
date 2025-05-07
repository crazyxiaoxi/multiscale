
import torch
import torch.nn as nn
from ultralytics.nn.tasks import DetectionModel

class MultiScaleYOLO(nn.Module):
    def __init__(self, cfg_s, cfg_m, cfg_l):
        super().__init__()
        # 按 yaml 构建三套 DetectionModel
        self.model_s = DetectionModel(cfg_s)
        self.model_m = DetectionModel(cfg_m)
        self.model_l = DetectionModel(cfg_l)
        self._share_weights()

    def _share_weights(self):
        layers = list(self.model_s.model.children())
        print(layers[:2])
        exit()
        # 保留前两层独立，后面完全共享
        shared = layers[3:]  # 从第三层开始

        # 重建 model_m / model_l
        self.model_m.model = nn.Sequential(*layers[:2], *shared)
        self.model_l.model = nn.Sequential(*layers[:2], *shared)

    def forward(self, patches_s, patches_m, patches_l):
        """
        patches_*: List[Tensor]，每个 Tensor 形状 [3,512,512]
        返回三组 preds，形式为 List[Dict]
        """
        preds_s = [self.model_s(p.unsqueeze(0)) for p in patches_s]
        preds_m = [self.model_m(p.unsqueeze(0)) for p in patches_m]
        preds_l = [self.model_l(p.unsqueeze(0)) for p in patches_l]
        return preds_s, preds_m, preds_l

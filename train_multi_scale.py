import torch
from torch.utils.data import DataLoader
from types import SimpleNamespace

from ultralytics.utils.loss import v8DetectionLoss
from multi_scale_yolo import MultiScaleYOLO
from dataset_multi_scale import MultiScaleDataset

def train():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    cfg_s = 'yaml/model_s.yaml'
    cfg_m = 'yaml/model_m.yaml'
    cfg_l = 'yaml/model_l.yaml'

    # 多尺度模型导入
    model = MultiScaleYOLO(cfg_s, cfg_m, cfg_l).to(device)

    # loss 超参数（box、cls、dfl、fl_gamma）
    hyp = {
        'box': 7.5,
        'cls': 0.5,
        'dfl': 1.5,
        'fl_gamma': 0.0
    }


    model.model_s.args = hyp

    # 损失函数
    criterion = v8DetectionLoss(model.model_s)

    # 数据集与加载器
    dataset = MultiScaleDataset(
        img_dir='/home/shared_directory/datasets/yolo_sdss/SDSS/images/train2017',
        label_dir='/home/shared_directory/datasets/yolo_sdss/SDSS/labels/train2017',
        scales=[2, 3, 5],
        patch_size=512
    )
    loader = DataLoader(
        dataset,
        batch_size=1,  
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    # 优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    epochs = 50
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for patches_s, patches_m, patches_l, labels in loader:
            patches_s = patches_s[0]
            patches_m = patches_m[0]
            patches_l = patches_l[0]

            patches_s = [p.to(device) for p in patches_s]
            patches_m = [p.to(device) for p in patches_m]
            patches_l = [p.to(device) for p in patches_l]

            targets = {
                'batch_idx': labels['batch_idx'].to(device).float()[0],  # (N,)
                'cls': labels['cls'].to(device).float()[0],              # (N,)
                'bboxes': labels['bboxes'].to(device).float()[0]         # (N, 4)
            }

            preds_s, preds_m, preds_l = model(patches_s, patches_m, patches_l)

            loss_s = sum(criterion(p, targets)[0] for p in preds_s)
            loss_m = sum(criterion(p, targets)[0] for p in preds_m)
            loss_l = sum(criterion(p, targets)[0] for p in preds_l)


            loss = loss_s + loss_m + loss_l
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(loader)
        print(f"Epoch {epoch + 1}/{epochs} — Loss: {avg_loss:.4f}")

    torch.save(model.state_dict(), 'runs/multi_scale_yolo.pt')

if __name__ == '__main__':
    train()

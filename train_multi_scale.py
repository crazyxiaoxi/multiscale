import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from ultralytics.utils.loss import v8DetectionLoss
from multi_scale_yolo import MultiScaleYOLO
from dataset_multi_scale import MultiScaleDataset

def train():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    cfg_s = 'yaml/model_s.yaml'
    cfg_m = 'yaml/model_m.yaml'
    cfg_l = 'yaml/model_l.yaml'

    model = MultiScaleYOLO(cfg_s, cfg_m, cfg_l).to(device)

    hyp = {
        'box': 7.5,
        'cls': 0.5,
        'dfl': 1.5,
        'fl_gamma': 0.0
    }
    model.model_s.args = hyp

    criterion = v8DetectionLoss(model.model_s)

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

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    epochs = 50
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        pbar = tqdm(loader, desc=f"Epoch {epoch + 1}/{epochs}")
        for patches_s, patches_m, patches_l, targets_s, targets_m, targets_l in pbar:
            patches_s = [p.to(device) for p in patches_s[0]]
            patches_m = [p.to(device) for p in patches_m[0]]
            patches_l = [p.to(device) for p in patches_l[0]]

            def process_targets(targets):
                t = targets[0]
                return {
                    'batch_idx': t['batch_idx'].view(-1, 1).to(device),
                    'cls': t['cls'].view(-1, 1).to(device),
                    'bboxes': t['bboxes'].view(-1, 4).to(device),
                }

            tgt_s = process_targets(targets_s)
            tgt_m = process_targets(targets_m)
            tgt_l = process_targets(targets_l)

            preds_s, preds_m, preds_l = model(patches_s, patches_m, patches_l)

            loss_s = sum(criterion(pred, tgt_s)[0] for pred in preds_s)
            loss_m = sum(criterion(pred, tgt_m)[0] for pred in preds_m)
            loss_l = sum(criterion(pred, tgt_l)[0] for pred in preds_l)

            total_loss = loss_s + loss_m + loss_l

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            running_loss += total_loss.item()
            pbar.set_postfix({'loss': total_loss.item()})

        avg_loss = running_loss / len(loader)
        print(f"Epoch {epoch + 1}/{epochs} — Loss: {avg_loss:.4f}")

    torch.save(model.state_dict(), 'runs/multi_scale_yolo.pt')

if __name__ == '__main__':
    train()

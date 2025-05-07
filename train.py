import argparse
from ultralytics import YOLO

parser = argparse.ArgumentParser(description="Train YOLO model with custom grid size and model name.")
parser.add_argument("--data", type=str, required=True, help="Model suffix (e.g., 'dsc' for yolov8dsc)")
parser.add_argument("--model", type=str, required=True, help="Model suffix (e.g., 'dsc' for yolov8dsc)")
parser.add_argument("--device", type=str, default="0", help="CUDA device(s) to use (e.g., '0' or '0,1')")
parser.add_argument("--ep", type=int, default="200", help="CUDA device(s) to use (e.g., '0' or '0,1')")
args = parser.parse_args()
dataset = args.data
model_name = f"yolov8{args.model}"


model = YOLO(f"./yaml/{model_name}.yaml")
#model = YOLO("/home/xijiawen/code/ultralytics/runs/train/4x4_ep200_pconv3/weights/best.pt")

ep = args.ep
if dataset == "4x4":
    img_size = 500 // 4  
elif dataset == "4x4fuse":
    img_size = 500 // 4
elif dataset == "2x2":
    img_size = 500 // 2
elif dataset == "origin":
    img_size = 500

results = model.train(
    data=f"./yaml/multi{dataset}.yaml",
    epochs=ep,
    imgsz=img_size,
    device=args.device,
    project="runs/newdataset",
    name=f"{dataset}_ep{ep}_{args.model}",
    save=True
)

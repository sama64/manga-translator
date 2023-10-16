import os
from ultralytics import YOLO

root_folder_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

# detection
model = YOLO(f"{root_folder_path}/models/detection.pt")  # YOLO("yolov8m-seg.pt")

model.train(
    data=f"{root_folder_path}/train_configs/train_detect.yaml", epochs=1000, batch=-1, device=0, workers=0, patience=100
)

# #segmentation
# model = YOLO(f"{root_folder_path}/models/segmentation.pt")  # YOLO("yolov8m-seg.pt")

# model.train(
#     data="train_seg.yaml", epochs=1000, batch=10, patience=100, device=0, workers=0
# )

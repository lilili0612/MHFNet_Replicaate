import argparse, sys, os, warnings

# 【修复报错的关键】强制 matplotlib 使用无头后端，必须写在前面
import matplotlib
matplotlib.use('Agg')

warnings.filterwarnings('ignore')

from pathlib import Path
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from ultralytics import YOLO

if __name__ == '__main__':
    # # model = YOLO("D:/YOLO-MIF-master/runs/train2/YOLOv8n-vsff2/weights/best.pt")
    # model = YOLO("runs/train2/YOLOv8n-vsff23/weights/best.pt")
    # metrics = model.val(data="qxs_data.yaml", #"D:/YOLO-MIF-master/data-vsff2.yaml",
    #           split='test', #'val',
    #           imgsz=256,
    #           batch=16,
    #           channels=4,
    #           use_simotm='RGBT',
    #           # conf=0.5,  #conf=0.5是用于“可视化画图(过滤掉低分数的框，让图片看起来更干净)”，而不是用于“计算指标”
    #           iou=0.5,
    #           # rect=False,
    #           save_json=True,  # if you need to cal coco metrice
    #           project='runs/val2',
    #           name='YOLOv8n-vsff2',
    #           )

    # #qxs_rgb
    # model = YOLO("runs/train2/YOLOv8n-RGB_single/weights/best.pt")
    # metrics = model.val(data="qxs_rgb.yaml", #"D:/YOLO-MIF-master/data-vsff2.yaml",
    #           split='test', #'val',
    #           imgsz=256,
    #           batch=16,
    #           channels=3,
    #           use_simotm='BGR',
    #           # conf=0.5,  #conf=0.5是用于“可视化画图(过滤掉低分数的框，让图片看起来更干净)”，而不是用于“计算指标”
    #           iou=0.5,
    #           # rect=False,
    #           save_json=True,  # if you need to cal coco metrice
    #           project='runs/val2',
    #           name='YOLOv8n-RGB_single',
    #           )

    #qxs_sar
    model = YOLO("runs/train2/YOLOv8n-SAR_single/weights/best.pt")
    metrics = model.val(data="qxs_sar.yaml", #"D:/YOLO-MIF-master/data-vsff2.yaml",
              split='test', #'val',
              imgsz=256,
              batch=16,
              channels=3,
              use_simotm='BGR',
              # conf=0.5,  #conf=0.5是用于“可视化画图(过滤掉低分数的框，让图片看起来更干净)”，而不是用于“计算指标”
              iou=0.5,
              # rect=False,
              save_json=True,  # if you need to cal coco metrice
              project='runs/val2',
              name='YOLOv8n-SAR_single',
              )

    print(metrics.results_dict)
    print(f"Mean Average Precision @.5:.95 : {metrics.box.map}")
    print(f"Mean Average Precision @ .50 : {metrics.box.map50}")
    print(f"Mean Average Precision @ .70 : {metrics.box.map75}")

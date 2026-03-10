import os
import glob

# 请将这里的路径替换为你真实的 labels 文件夹路径
# 需要包含 train, val, test 下的 rgb 和 sar 标签目录
label_dirs = [
    '/mnt/sdb/lijing/ImageFusion/ShipDetection/SAR_OPT_datasets/SAR-OPT/MHFNet_dataset/QXS-datasets/dataset_split/labels/train/rgb',
    '/mnt/sdb/lijing/ImageFusion/ShipDetection/SAR_OPT_datasets/SAR-OPT/MHFNet_dataset/QXS-datasets/dataset_split/labels/train/sar',
    '/mnt/sdb/lijing/ImageFusion/ShipDetection/SAR_OPT_datasets/SAR-OPT/MHFNet_dataset/QXS-datasets/dataset_split/labels/val/rgb',
    '/mnt/sdb/lijing/ImageFusion/ShipDetection/SAR_OPT_datasets/SAR-OPT/MHFNet_dataset/QXS-datasets/dataset_split/labels/val/sar',
    '/mnt/sdb/lijing/ImageFusion/ShipDetection/SAR_OPT_datasets/SAR-OPT/MHFNet_dataset/QXS-datasets/dataset_split/labels/test/rgb',
    '/mnt/sdb/lijing/ImageFusion/ShipDetection/SAR_OPT_datasets/SAR-OPT/MHFNet_dataset/QXS-datasets/dataset_split/labels/test/sar',
]

for d in label_dirs:
    if not os.path.exists(d):
        continue
    txt_files = glob.glob(os.path.join(d, '*.txt'))
    for txt_file in txt_files:
        with open(txt_file, 'r') as f:
            lines = f.readlines()

        with open(txt_file, 'w') as f:
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 5:
                    # 强制将第一个数字（类别ID）改为 0
                    parts[0] = '0'
                    f.write(' '.join(parts) + '\n')
print("✅ 所有标签的类别 ID 已成功修复为 0！")
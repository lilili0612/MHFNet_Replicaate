import os
import shutil
import random
import numpy as np

# 设定随机种子，保证每次划分结果一致
random.seed(0)

val_size = 0.1
test_size = 0.2
postfix = 'png'  # 根据你的数据集，这里修改为 png

# 原始数据集路径
# base_dir = '/mnt/sdb/lijing/ImageFusion/ShipDetection/SAR_OPT_datasets/SAR-OPT/MHFNet_dataset/QXS-datasets'
base_dir = '/mnt/sdb/lijing/ImageFusion/ShipDetection/SAR_OPT_datasets/SAR-OPT/MHFNet_dataset/Suez-datasets'
imgpath = os.path.join(base_dir, 'images')
txtpath = os.path.join(base_dir, 'labels')

# 定义需要处理的模态
modalities = ['rgb', 'sar']

# 1. 创建目标文件夹结构 (保留 rgb 和 sar 子文件夹)
# 为了不污染原文件夹，我们将划分后的数据放在一个叫 'dataset_split' 的新文件夹中
out_dir = 'dataset_split'
splits = ['train', 'val', 'test']

for split in splits:
    for mod in modalities:
        os.makedirs(f'{base_dir}/{out_dir}/images/{split}/{mod}', exist_ok=True)
        os.makedirs(f'{base_dir}/{out_dir}/labels/{split}/{mod}', exist_ok=True)

# 2. 获取基准文件列表 (以 labels/rgb 下的 txt 文件为准)
ref_label_path = os.path.join(txtpath, 'rgb')
# 过滤出所有的 txt 文件
listdir = np.array([i for i in os.listdir(ref_label_path) if i.endswith('.txt')])
random.shuffle(listdir)

# 3. 计算划分的索引
num_total = len(listdir)
train_end = int(num_total * (1 - val_size - test_size))
val_end = int(num_total * (1 - test_size))

train_files = listdir[:train_end]
val_files = listdir[train_end:val_end]
test_files = listdir[val_end:]

print(f'Train set size: {len(train_files)} | Val set size: {len(val_files)} | Test set size: {len(test_files)}')


# 4. 定义拷贝函数，同时处理 rgb 和 sar
def copy_dataset(file_list, split_name):
    for f in file_list:
        base_name = f[:-4]  # 去掉 '.txt' 获取纯文件名

        for mod in modalities:
            # 原始文件路径
            src_img = os.path.join(imgpath, mod, f'{base_name}.{postfix}')
            src_label = os.path.join(txtpath, mod, f)

            # 目标文件路径
            dst_img = os.path.join(base_dir, out_dir, 'images', split_name, mod, f'{base_name}.{postfix}')
            dst_label = os.path.join(base_dir, out_dir, 'labels', split_name, mod, f)

            # 拷贝文件 (增加 exists 判断，防止个别配对文件丢失导致报错)
            if os.path.exists(src_img):
                shutil.copy(src_img, dst_img)
            else:
                print(f"Warning: Image not found {src_img}")

            if os.path.exists(src_label):
                shutil.copy(src_label, dst_label)
            else:
                print(f"Warning: Label not found {src_label}")


# 5. 执行拷贝
print("Starting to copy train set...")
copy_dataset(train_files, 'train')
print("Starting to copy val set...")
copy_dataset(val_files, 'val')
print("Starting to copy test set...")
copy_dataset(test_files, 'test')

print("✅ Dataset splitting completed successfully!")
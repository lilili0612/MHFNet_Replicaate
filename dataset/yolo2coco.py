import os
import cv2
import json
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import argparse

# classes = ['ore carrier', 'fishing boat', 'passenger ship', 'general cargo ship', 'bulk cargo carrier', 'container ship']
# 【修复1】将类别强行统一为数据集实际的1个类别
classes = ['ship']

parser = argparse.ArgumentParser()
parser.add_argument('--image_path', default='',type=str, help="path of images")
parser.add_argument('--label_path', default='',type=str, help="path of labels .txt")
parser.add_argument('--save_path', type=str,default='data.json', help="if not split the dataset, give a path to a json file")
arg = parser.parse_args()

def yolo2coco(arg):
    print("Loading data from ", arg.image_path, arg.label_path)

    assert os.path.exists(arg.image_path)
    assert os.path.exists(arg.label_path)
    
    originImagesDir = arg.image_path                                   
    originLabelsDir = arg.label_path
    # images dir name
    indexes = os.listdir(originImagesDir)

    dataset = {'categories': [], 'annotations': [], 'images': []}
    for i, cls in enumerate(classes, 0):
        # dataset['categories'].append({'id': i, 'name': cls, 'supercategory': 'mark'})
        dataset['categories'].append({'id': i + 1, 'name': cls, 'supercategory': 'mark'})  # 【修复2】COCO 格式的 category_id 必须从 1 开始
    
    # 标注的id
    ann_id_cnt = 0
    for k, index in enumerate(tqdm(indexes)):
        # 支持 png jpg 格式的图片.
        txtFile = f'{index[:index.rfind(".")]}.txt'
        stem = index[:index.rfind(".")]

        # 【修复3】强行将文件名转为 int 格式的 image_id
        try:
            img_id = int(stem)
        except ValueError:
            print(f"警告：图片名 {index} 无法转为纯数字ID，请检查数据集命名。")
            continue

        # 读取图像的宽和高
        try:
            im = cv2.imread(os.path.join(originImagesDir, index))
            height, width, _ = im.shape
        except Exception as e:
            print(f'{os.path.join(originImagesDir, index)} read error.\nerror:{e}')
        # 添加图像的信息
        if not os.path.exists(os.path.join(originLabelsDir, txtFile)):
            # 如没标签，跳过，只保留图片信息.
            continue
        dataset['images'].append({'file_name': index,
                            # 'id': stem,
                            'id': img_id,
                            'width': width,
                            'height': height})
        with open(os.path.join(originLabelsDir, txtFile), 'r') as fr:
            labelList = fr.readlines()
            for label in labelList:
                label = label.strip().split()
                x = float(label[1])
                y = float(label[2])
                w = float(label[3])
                h = float(label[4])

                # convert x,y,w,h to x1,y1,x2,y2
                H, W, _ = im.shape
                x1 = (x - w / 2) * W
                y1 = (y - h / 2) * H
                x2 = (x + w / 2) * W
                y2 = (y + h / 2) * H
                # # 标签序号从0开始计算, coco2017数据集标号混乱，不管它了。
                # cls_id = int(label[0])
                # 【修复4】YOLO的类别0 + 1，强行对齐到刚才定义的类别 1
                cls_id = int(label[0]) + 1

                width = max(0, x2 - x1)
                height = max(0, y2 - y1)
                dataset['annotations'].append({
                    'area': width * height,
                    'bbox': [x1, y1, width, height],
                    'category_id': cls_id,
                    'id': ann_id_cnt,
                    # 'image_id': stem,
                    'image_id': img_id,  # 改为 img_id，它是整数
                    'iscrowd': 0,
                    # mask, 矩形是从左上角点按顺时针的四个顶点
                    'segmentation': [[x1, y1, x2, y1, x2, y2, x1, y2]]
                })
                ann_id_cnt += 1

    # 保存结果
    save_dir = os.path.dirname(arg.save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)  # 如果目录不存在，递归创建它

    # 【修复5】解决乱码问题
    with open(arg.save_path, 'w', encoding='utf-8') as f:
        # json.dump(dataset, f)
        # indent=4: 增加4个空格的缩进，让JSON以标准树状结构换行显示
        # ensure_ascii=False: 允许直接写入中文等非ASCII字符，而不是 \uXXXX
        json.dump(dataset, f, indent=4, ensure_ascii=False)
        print('Save annotation to {}'.format(arg.save_path))
        print(f'成功转换！总计真实框数量(GT): {ann_id_cnt}')

if __name__ == "__main__":
    yolo2coco(arg)
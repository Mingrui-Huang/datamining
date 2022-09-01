import os
import random
from sklearn.model_selection import train_test_split

CLASSES = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
           'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']


def images_name_label(train=True):
    data_root = os.path.abspath(os.path.join(os.getcwd()))
    all_cls_txt_fname = []
    for clsIdx in range(len(CLASSES)):
        cls = CLASSES[clsIdx]
        if train:
            cls_txt_fname = os.path.join(data_root, "VOCdevkit/VOC2012/ImageSets/Main/", cls + '_train.txt')
        else:
            cls_txt_fname = os.path.join(data_root, "VOCdevkit/VOC2012/ImageSets/Main/", cls + '_val.txt')
        all_cls_txt_fname.append(cls_txt_fname)

    if train:
        listText = open('./train_unshuffled.txt', 'a')
        for i in range(len(all_cls_txt_fname)):
            txt_fname = all_cls_txt_fname[i]
            with open(txt_fname, 'r') as f:
                image_name_label = []
                for line in f:
                    line = line.strip('\n')  # 移除字符串首尾的换行符
                    line = line.rstrip()  # 删除末尾空
                    words = line.split()  # 以空格为分隔符 将字符串拆分
                    if int(words[1]) == 1:
                        name = words[0] + ' ' + str(i) + '\n'
                        listText.write(name)
    else:
        listText = open('./val_original.txt', 'a')
        for i in range(len(all_cls_txt_fname)):
            txt_fname = all_cls_txt_fname[i]
            with open(txt_fname, 'r') as f:
                image_name_label = []
                for line in f:
                    line = line.strip('\n')  # 移除字符串首尾的换行符
                    line = line.rstrip()  # 删除末尾空
                    words = line.split()  # 以空格为分隔符 将字符串拆分
                    if int(words[1]) == 1:
                        name = words[0] + ' ' + str(i) + '\n'
                        listText.write(name)

def shuffle_train_txt():
    lines = []
    out_txt = open('./train.txt','w')
    with open("./train_unshuffled.txt", 'r') as f1:
        for line in f1:
            lines.append(line)
    random.shuffle(lines)
    for line in lines:
        out_txt.write(line)
    print('train_num:%d' % len(lines))

def split_val_test():
    original_val_txt = r'./val_original.txt'
    images = []
    with open(original_val_txt, 'r') as file:
        for line in file:
            line = line.strip('\n')  # 移除字符串首尾的换行符
            line = line.rstrip()  # 删除末尾空
            words = line.split()  # 以空格为分隔符 将字符串拆分
            images.append(words[0] + ' ' + str(words[1]))  # imgs中包含有图像路径和标签
    # print(images)
    print('original_val_number:%d' % len(images))
    # 验证集：测试集 为 6:4
    val, test = train_test_split(images, train_size=0.6, random_state=0)

    # print(train)
    print('val_number:%d' % len(val))
    print('test_number:%d' % len(test))
    with open('val.txt', 'w') as f:
        f.write('\n'.join(val))
    with open('test.txt', 'w') as f:
        f.write('\n'.join(test))

    print('txt have been generated')

if __name__ == '__main__':
    # 1.生成未乱序的train.txt和val.txt,生成后注释掉，否则会重复添加到txt里
    images_name_label(train=True)
    images_name_label(train=False)
    # 2.打乱train.txt里的顺序，生成新的train.txt
    shuffle_train_txt()
    # 3.在原本的val.txt里划分出新的val.txt和test.txt
    split_val_test()


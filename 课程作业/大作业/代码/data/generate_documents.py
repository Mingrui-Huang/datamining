import shutil
import os

out_root = './processed_data'
if not os.path.exists(out_root):
    os.mkdir(out_root)

train_or_val = ["train", "val"]

labels = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
          'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']

for i in train_or_val:
    path_train_or_val = os.path.join(out_root, i)
    if not os.path.exists(path_train_or_val):
        os.mkdir(path_train_or_val)
        for j in range(len(labels)):
            path_train_or_val_label = os.path.join(path_train_or_val, labels[j])
            if not os.path.exists(path_train_or_val_label):
                os.mkdir(path_train_or_val_label)

def classify_data(txt_name, train_or_val=None, labels=None):
    txt_path = './' + txt_name
    with open(txt_path, 'r') as fh:
        for line in fh:
            line = line.strip('\n')  # 移除字符串首尾的换行符
            line = line.rstrip()  # 删除末尾空
            words = line.split()  # 以空格为分隔符 将字符串分成两部分
            srcfile = './VOCdevkit/VOC2012/JPEGImages/' + words[0] + '.jpg'
            imgs_label = int(words[1])
            print(srcfile)
            if imgs_label == 0:
                shutil.copy(srcfile, out_root + '/' + train_or_val + '/' + labels[0])
            elif imgs_label == 1:
                shutil.copy(srcfile, out_root + '/' + train_or_val + '/' + labels[1])
            elif imgs_label == 2:
                shutil.copy(srcfile, out_root + '/' + train_or_val + '/' + labels[2])
            elif imgs_label == 3:
                shutil.copy(srcfile, out_root + '/' + train_or_val + '/' + labels[3])
            elif imgs_label == 4:
                shutil.copy(srcfile, out_root + '/' + train_or_val + '/' + labels[4])
            elif imgs_label == 5:
                shutil.copy(srcfile, out_root + '/' + train_or_val + '/' + labels[5])
            elif imgs_label == 6:
                shutil.copy(srcfile, out_root + '/' + train_or_val + '/' + labels[6])
            elif imgs_label == 7:
                shutil.copy(srcfile, out_root + '/' + train_or_val + '/' + labels[7])
            elif imgs_label == 8:
                shutil.copy(srcfile, out_root + '/' + train_or_val + '/' + labels[8])
            elif imgs_label == 9:
                shutil.copy(srcfile, out_root + '/' + train_or_val + '/' + labels[9])
            elif imgs_label == 10:
                shutil.copy(srcfile, out_root + '/' + train_or_val + '/' + labels[10])
            elif imgs_label == 11:
                shutil.copy(srcfile, out_root + '/' + train_or_val + '/' + labels[11])
            elif imgs_label == 12:
                shutil.copy(srcfile, out_root + '/' + train_or_val + '/' + labels[12])
            elif imgs_label == 13:
                shutil.copy(srcfile, out_root + '/' + train_or_val + '/' + labels[13])
            elif imgs_label == 14:
                shutil.copy(srcfile, out_root + '/' + train_or_val + '/' + labels[14])
            elif imgs_label == 15:
                shutil.copy(srcfile, out_root + '/' + train_or_val + '/' + labels[15])
            elif imgs_label == 16:
                shutil.copy(srcfile, out_root + '/' + train_or_val + '/' + labels[16])
            elif imgs_label == 17:
                shutil.copy(srcfile, out_root + '/' + train_or_val + '/' + labels[17])
            elif imgs_label == 18:
                shutil.copy(srcfile, out_root + '/' + train_or_val + '/' + labels[18])
            elif imgs_label == 19:
                shutil.copy(srcfile, out_root + '/' + train_or_val + '/' + labels[19])
        print("Copy files Successfully!")

def pickout_test_images():
    test_image_path = os.path.join(out_root, 'test')
    if not os.path.exists(test_image_path):
        os.mkdir(test_image_path)
    with open('test.txt', 'r') as f:
        for line in f:
            line = line.strip('\n')  # 移除字符串首尾的换行符
            line = line.rstrip()  # 删除末尾空
            words = line.split()  # 以空格为分隔符 将字符串分成两部分
            srcfile = './VOCdevkit/VOC2012/JPEGImages/' + words[0] + '.jpg'
            shutil.copy(srcfile, out_root + '/' + 'test')
    print("Copy test images successfully!")

if __name__ == '__main__':

    classify_data(txt_name='train.txt', train_or_val='train', labels=labels)
    classify_data(txt_name='val.txt', train_or_val='val', labels=labels)
    pickout_test_images()

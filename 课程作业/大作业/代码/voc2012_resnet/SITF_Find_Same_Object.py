import random

import cv2
import os

class_name = 'cat'
image_rootpath = './{}_pictures_you_want'.format(class_name)
targe_picture_name = '2008_002681.jpg'


def generate_same_object_different_pictures():
    for i in range(10):
        # 读取目标图片
        img = cv2.imread(os.path.join(image_rootpath, targe_picture_name))
        # 随机旋转
        rows = img.shape[0]
        cols = img.shape[1]
        angle = random.randint(-180, 180)
        scale = random.random() + 0.5
        M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
        img = cv2.warpAffine(src=img, M=M, dsize=(rows, cols), borderValue=(255, 255, 255))

        # 随机缩放
        X_scale = 0.5 + random.random()
        Y_scale = 0.5 + random.random()
        img = cv2.resize(img, (0, 0), fx=X_scale, fy=Y_scale,
                         interpolation=cv2.INTER_NEAREST)

        # 随机裁剪
        # _ = random.random()
        # rows2 = img.shape[0]
        # cols2 = img.shape[1]
        # rows_bottom = int(rows2/2 - _*(rows2/2))
        # rows_top = int(rows2/2 + _*(rows2/2))
        # cols2_left = int(cols2/2 - _*(cols2/2))
        # cols2_right = int(cols2/2 + _*(cols2/2))
        # img = img[rows_bottom: rows_top, cols2_left: cols2_right]

        # 随机翻转
        flip_type = random.randint(-1, 1)
        img = cv2.flip(img, flip_type)
        new_imgname = targe_picture_name[:-4] + '_' + str(i) + '.jpg'
        cv2.imwrite(os.path.join(image_rootpath, new_imgname), img)
    print('finish generate new pictures!')


def find_related_pictures():
    image_outpath = './related_pictures'
    if not os.path.exists(image_outpath):
        os.mkdir(image_outpath)

    img1 = cv2.imread(os.path.join(image_rootpath, targe_picture_name))
    img1 = cv2.resize(img1, (300, 300))

    num = 0
    for filename in os.listdir(image_rootpath):

        img2 = cv2.imread(os.path.join(image_rootpath, filename))
        img2 = cv2.resize(img2, (300, 300))
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        # 特征点检测
        sift = cv2.SIFT_create()
        keypoints_1, descriptors_1 = sift.detectAndCompute(gray1, None)
        keypoints_2, descriptors_2 = sift.detectAndCompute(gray2, None)
        # 利用knn计算两个描述子的匹配
        bf = cv2.BFMatcher(cv2.NORM_L1)
        matches = bf.knnMatch(descriptors_1, descriptors_2, k=2)
        matches = sorted(matches, key=lambda x: x[0].distance)
        # 比率测试，获得良好匹配
        good_matchs = [[match1] for (match1, match2) in matches if match1.distance < 0.7 * match2.distance]

        # 匹配成功标准，并对符合标准的图片进行保存
        MIN_MATCH_POINTS = 15
        if len(good_matchs) > MIN_MATCH_POINTS and filename != targe_picture_name:
            num = num + 1
            # cv2.imshow('match_picture', img2)
            # cv2.waitKey()
            img1_match_img2 = cv2.drawMatchesKnn(img1, keypoints_1, img2, keypoints_2, good_matchs[:50], None, flags=2)
            cv2.imshow('match', img1_match_img2)
            cv2.waitKey()
            cv2.imwrite(os.path.join(image_outpath, filename), img2)
    print('have found {} related pictures'.format(num))

if __name__ == '__main__':
    # generate_same_object_different_pictures()
    find_related_pictures()

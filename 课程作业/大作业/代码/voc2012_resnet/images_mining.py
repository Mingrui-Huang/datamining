import os
import json

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

from resnet import resnet34

# 选择你想要找的类别
class_name = 'cat'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# transform方法
data_transform = transforms.Compose([
                                     transforms.Resize(256),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
# 从json中读入标签索引字典
json_path = './class_indices.json'
assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)
with open(json_path, "r") as f:
    class_indict = json.load(f)

# 待测图片文件夹
test_img_path = '../data/processed_data/test'
# 该文件夹存放找到的图片
pick_out_path = './{}_pictures_you_want'.format(class_name)
if not os.path.exists(pick_out_path):
    os.mkdir(pick_out_path)

# 开始寻找图片
find_num = 0
test_imglist = os.listdir(test_img_path)
for index in range(len(test_imglist)):
    img1 = Image.open(os.path.join(test_img_path, test_imglist[index]))
    # [N, C, H, W]
    img = data_transform(img1)
    # 增加一个batch维度
    img = torch.unsqueeze(img, dim=0)

    # 选择模型
    net = resnet34()
    in_channel = net.fc.in_features
    net.fc = nn.Sequential(
        nn.Linear(in_channel, 100, bias=True),
        nn.ReLU(),
        nn.Linear(100, 20, bias=True),
    )
    net.to(device)
    # 载入训练好的模型权重
    weights_path = "./best_resNet34.pth"
    assert os.path.exists(weights_path), "file: '{}' dose not exist.".format(weights_path)
    net.load_state_dict(torch.load(weights_path, map_location=device))

    # 预测图片
    net.eval()
    with torch.no_grad():
        output = torch.squeeze(net(img.to(device))).cpu()
        predict = torch.softmax(output, dim=0)
        predict_cla = torch.argmax(predict).numpy()


    # 显示该图片
    plt.ion()
    plt.imshow(img1)
    print_res = "class: {}   prob: {:.3}".format(class_indict[str(predict_cla)],
                                                 predict[predict_cla].numpy())
    plt.title(print_res)
    plt.pause(1)
    plt.close()

    # 显示图片名称
    print('\n----------------------------------'
          '\n{}/{}'
          '\n{}'.format(index, len(test_imglist), test_imglist[index]))

    # 显示各类别预测概率，并由大到小排序
    predict_list = []
    for i in range(len(predict)):
        print("class: {:10}   prob: {:.3}".format(class_indict[str(i)],
                                                  predict[i].numpy()))
        predict_list.append(predict[i].numpy())
        predict_list.sort()
        predict_list.reverse()
    # 如果预测类别与规定类别一致，并且预测概率大于第二大概率的1.5倍，保存该图片到指定文件夹
    if class_indict[str(predict_cla)] == class_name and predict_list[0] > 1.5*predict_list[1]:
        find_num = find_num + 1
        img1.save(os.path.join(pick_out_path, test_imglist[index]))
        print('\nHave save image:{} to {}' .format(test_imglist[index], pick_out_path))
        print('have found {} pictures' .format(find_num))
    if index == len(test_imglist) - 1:
        print('Finish search!')

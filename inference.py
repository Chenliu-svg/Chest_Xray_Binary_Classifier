import argparse
import torch
import torch.nn as nn
from torchvision import models
import os
import shutil
from PIL import Image
from torchvision.models import ResNet50_Weights
from dataset import val_transform


# 加载模型进行预测：
def inference(args):
    # get training device
    useCuda = torch.cuda.is_available()
    device = torch.device("cuda:1" if useCuda else "cpu")
    print('Using CUDA') if useCuda else print('Using CPU')

    # parse arguments
    newDir = args.newDir
    oldDir = args.oldDir

    modelDir = args.modelDir

    count=0
    # 加载预训练的ResNet模型
    model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
    # 替换模型的最后一层
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 1)  # 二分类
    checkpoint=torch.load(os.path.join(modelDir,'bestBinaryClassifier.pt'))
    model.load_state_dict(checkpoint)
    model.to(device)

    # 固定预训练模型的参数
    for param in model.parameters():
        param.requires_grad = False

    # 获取需要分类的数据
    caseName = os.listdir(newDir)
    s1 = len(caseName)
    leftCase = os.listdir(oldDir)[s1:]

    # 根据预测结果拷贝路径,一个文件夹一个文件夹的处理
    for case in leftCase:
        count+=1
        imagePath0 = os.path.join(oldDir, case+'/0.png')
        image_0 = Image.open(imagePath0).convert('RGB')
        image_0 = val_transform(image_0)

        imagePath1 = os.path.join(oldDir, case + '/1.png')
        image_1 = Image.open(imagePath1).convert('RGB')
        image_1 = val_transform(image_1)

        img=torch.stack((image_0,image_1),dim=0).to(device)

        outputs=model(img).squeeze()
        out = (nn.Sigmoid(outputs) > 0.5).to(torch.float32)

        # 必然要有一个0和1
        if out[0]==out[1]:
            print(f'这个case需要人工处理：{oldDir+"/"+case}，预测全部为{out[0]}')

        else:
            for j in range(2):
                name='/frontal.png' if out[j]==1 else '/left.png'
                # 如果目标路径的文件夹不存在，则创建路径
                destination_folder=newDir+'/'+case
                if not os.path.exists(destination_folder):
                    os.makedirs(destination_folder)
                # 将图片从源路径拷贝到目标路径
                shutil.copy(oldDir+'/'+case+f'/{j}.png', destination_folder+name)

        if count%20==0:
            print(f'left:{len(leftCase)-count}')


if __name__=='__main__':
    parser = argparse.ArgumentParser()

    # Data input settings
    parser.add_argument('--oldDir', type=str, default='./iu_xray',
                        help='the path to the directory containing the cases.')
    parser.add_argument('--newDir', type=str, default='./newDir',
                        help='the path to the directory containing the newly-labeled cases.')

    parser.add_argument('--modelDir', typr=str, default='./trainedModels',
                        help='the path to the saved model')

    args = parser.parse_args()

    inference(args)

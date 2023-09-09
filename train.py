import argparse

import torch
import torch.nn as nn
from dataset import ChestXRayDataset,val_transform,train_transform
from torch.utils.data import DataLoader
from torchvision.models import ResNet50_Weights
from torchvision import  models
import os
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim


def train(args):

    # get training device
    useCuda = torch.cuda.is_available()
    device = torch.device("cuda:1" if useCuda else "cpu")
    print('Using CUDA') if useCuda else print('Using CPU')

    # parse arguments
    newDir = args.newDir
    oldDir = args.oldDir
    batchSize= args.batchSize
    lr=args.lr
    saveDir=args.saveDir

    # 获取所有的图像文件夹 7：1：2划分训练，验证，测试
    caseName = os.listdir(newDir)
    s1 = len(caseName)
    leftCase = os.listdir(oldDir)[s1:]

    splitCase = {'train': caseName[:int(s1 * 0.7)], 'val': caseName[int(s1 * 0.7):int(s1 * 0.8)],
                  'test': caseName[int(s1 * 0.8):]}

    # tensorboard 记录loss , acc曲线
    writer1 = SummaryWriter('BClines' + '/loss')
    writer2 = SummaryWriter('BClines' + '/acc')

    train_dataset = ChestXRayDataset(root_dir=newDir ,split_case=splitCase, split='train', transform=train_transform)
    train_dataloader = DataLoader(train_dataset, batch_size=batchSize, shuffle=True)
    val_dataset = ChestXRayDataset(root_dir=newDir ,split_case=splitCase, split='val', transform=val_transform)
    val_dataloader = DataLoader(val_dataset, batch_size=batchSize, shuffle=False)
    test_dataset = ChestXRayDataset(root_dir=newDir ,split_case=splitCase, split='test', transform=val_transform)
    test_dataloader = DataLoader(test_dataset, batch_size=batchSize, shuffle=False)
    print(len(val_dataset))

    # 加载预训练的ResNet模型
    model = models.resnet50(weights=ResNet50_Weights.DEFAULT)

    # 替换模型的最后一层
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 1)  # 二分类问题
    model.to(device)
    # 设置训练参数
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # 训练模型
    best_acc =0
    for epoch in range(100):
        epoch_acc ={'train' :0 ,'val' :0 ,'test' :0}
        epoch_loss ={'train' :0 ,'val' :0 ,'test' :0}
        for i, (images, labels) in enumerate(train_dataloader):
            labels =labels.to(device).to(torch.float32)
            outputs = model(images.to(device)).squeeze()

            loss = criterion(outputs, labels)
            epoch_loss['train']+=loss.item()
            out = (nn.Sigmoid(outputs) > 0.5).to(torch.float32)
            acc = sum(out == labels)
            epoch_acc['train' ]+=acc
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # 验证
        for i, (images, labels) in enumerate(val_dataloader):
            labels = labels.to(device).to(torch.float32)
            outputs = model(images.to(device)).squeeze()

            loss = criterion(outputs, labels)
            epoch_loss['val'] += loss.item()
            out =(nn.Sigmoid(outputs ) >0.5).to(torch.float32)
            acc = sum(out== labels)
            epoch_acc['val'] += acc

        # 测试
        for i, (images, labels) in enumerate(test_dataloader):
            labels = labels.to(device).to(torch.float32)
            outputs = model(images.to(device)).squeeze()

            loss = criterion(outputs, labels)
            epoch_loss['test'] += loss.item()
            out = (nn.Sigmoid(outputs) > 0.5).to(torch.float32)
            acc = sum(out == labels)
            epoch_acc['test'] += acc


        print(f'epoch:{epoch}\nloss: train:{(epoch_loss["train"] ) /len(train_dataloader)},'
              f'val:{epoch_loss["val" ] /len(val_dataloader)},test:{epoch_loss["test" ] /len(test_dataloader)}\n'
              f'acc: train:{epoch_acc["train" ] /(int(s1 *0.7 ) *2)},'
              f'val:{epoch_acc["val" ] /(int(s1 *0.1 ) *2)},test:{epoch_acc["test" ] /(int(s1 *0.2 ) *2)}\n')

        writer1.add_scalars('loss',
                            {'train': (epoch_loss["train"] ) /len(train_dataloader),
                             'val' :epoch_loss["val" ] /len(val_dataloader),
                             'test': epoch_loss["test" ] /len(test_dataloader)}, epoch)
        writer2.add_scalars('acc',
                            {'train': epoch_acc["train" ] /len(train_dataset),
                             'val': epoch_acc["val" ] /len(val_dataset),
                             'test': epoch_acc["test" ] /len(test_dataset)}, epoch)
        cur_acc =epoch_acc["val"] / len(val_dataset)
        if cur_acc >best_acc:
            best_acc =cur_acc
            # 保存最好的模型
            torch.save(model.state_dict(), os.path.join(saveDir,"bestBinaryClassifier.pt"))
    # 保存最后的模型
    torch.save(model.state_dict(), os.path.join(saveDir,"lastBinaryClassifier.pt"))



if __name__=="__main__":
    parser = argparse.ArgumentParser()

    # Data input settings
    parser.add_argument('--oldDir', type=str, default='./iu_xray',
                        help='the path to the directory containing the cases.')
    parser.add_argument('--newDir', type=str, default='./newDir',
                        help='the path to the directory containing the newly-labeled cases.')

    parser.add_argument('--batchSize',type=int,default=16)
    parser.add_argument('--lr', type=float, default=0.0005)
    parser.add_argument('--saveDir',typr=str,default='./trainedModels',
                        help='the path to save the model')

    args = parser.parse_args()

    if not os.path.exists(args.saveDir):
        os.makedirs(args.saveDir)

    train(args)
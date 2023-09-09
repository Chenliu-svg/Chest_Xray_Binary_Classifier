import random
from torch.utils.data import Dataset
import os
from PIL import Image
from torchvision import transforms

# 数据增强
train_transform = transforms.Compose([
    transforms.Resize(256, antialias=True),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406),
                         (0.229, 0.224, 0.225))
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224), antialias=True),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406),
                         (0.229, 0.224, 0.225))
])


# 创建自定义数据集类
class ChestXRayDataset(Dataset):
    def __init__(self,root_dir, split_case, split, transform=None):
        self.root_dir=root_dir
        self.image_files=[]
        for i in split_case[split]:
            self.image_files.append(i+'/frontal.png')
            self.image_files.append(i + '/left.png')
        random.shuffle(self.image_files)
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        image_path = os.path.join(self.root_dir, self.image_files[index])
        image = Image.open(image_path).convert('RGB')
        label = 1 if "frontal" in self.image_files[index] else 0  # 正视图为1，非正视图为0

        if self.transform:
            image = self.transform(image)

        return image, label

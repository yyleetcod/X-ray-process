# coding=utf-8
import torch
from torch.autograd import Variable
import torchvision
from torchvision import transforms, datasets
from torch.utils.data import Dataset, DataLoader
from PIL import Image

import argparse
import os

import time
from torch import nn, optim

def default_loader(path):
    return Image.open(path).convert('RGB')#这里改成以灰度形式读取
class MyDataset(Dataset):
    def __init__(self, txt, transform=None, target_transform=None, loader=default_loader):
        fh = open(txt, 'r')
        imgs = []
        for line in fh:
            line = line.strip('\n')#去掉开头结尾的换行符号
            line = line.rstrip()
            words = line.split()#依据空格进行分割
            imgs.append((words[0],int(words[1])))#word0是路径名称，word1是标签
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        fn, label = self.imgs[index]
        img = self.loader(fn)
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.imgs)


root = os.getcwd() + '/'
# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print (device)
parser = argparse.ArgumentParser("""Image classifical!""")
parser.add_argument('--train_path', type=str, default=root,
                    help="""image dir path default: './xray/⁩'.""") #这里是训练集的路径
parser.add_argument('--test_path', type=str, default=root,
                    help="""image dir path default: './xray/⁩'.""")
parser.add_argument('--epochs', type=int, default=200,
                    help="""Epoch default:1000.""")
parser.add_argument('--batch_size', type=int, default=5,
                    help="""Batch_size default:256.""")
parser.add_argument('--lr', type=float, default=0.001,
                    help="""learing_rate. Default=0.0001""")
parser.add_argument('--num_classes', type=int, default=3,
                    help="""num classes""") #分类的个数
parser.add_argument('--model_path', type=str, default='./model/',
                    help="""Save model path""") #模型存储路径
parser.add_argument('--model_name', type=str, default='real_classification.pth',
                    help="""Model name.""") #模型名称
parser.add_argument('--display_epoch', type=int, default=1) 

args = parser.parse_args()

# Create model
if not os.path.exists(args.model_path):
    os.makedirs(args.model_path)

# 准备数据集并预处理
transform_train = transforms.Compose([
    transforms.Pad(padding = 0),
    transforms.CenterCrop(1000),  #先四周填充0，再把图像随机裁剪成32*32
    transforms.RandomHorizontalFlip(),  #图像一半的概率翻转，一半的概率不翻转
    transforms.ToTensor(),
    #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), #R,G,B每层的归一化用到的均值和方差
])

transform_test = transforms.Compose([
    transforms.Pad(padding = 0),
    transforms.CenterCrop(1000), 
    transforms.ToTensor(),
    #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

train_dataset = MyDataset(txt = args.train_path + 'train_label.txt', transform = transform_train)#注意修改标签文件名
print(len(train_dataset))
for i in train_dataset:
    print (i)
    print (i[0])
    print (len(i[0][0]))
    break
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
print(len(train_loader))
test_dataset = MyDataset(txt = args.test_path + 'train_label.txt', transform = transform_test) #注意修改标签文件名
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)


def train():
    print(f"Train numbers:{len(train_dataset)}")

    # Load model
    # if torch.cuda.is_available():
    #     model = torch.load(args.model_path + args.model_name).to(device)
    # else:
    #     model = torch.load(args.model_path + args.model_name, map_location='cpu')
    model = torchvision.models.resnet18(pretrained = True)
    model.avgpool = nn.AvgPool2d(1, 1)
    fc_in_size = model.fc.in_features
    model.fc = nn.Linear(524288, args.num_classes)
    model = model.to(device)
    # cast
    cast = nn.CrossEntropyLoss().to(device)
    # Optimization
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-8)

    for epoch in range(1, args.epochs + 1):
        model.train()
        # start time
        start = time.time()
        for i, data in enumerate(train_loader):
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            # Forward pass
            outputs = model(images)
            loss = cast(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if epoch % args.display_epoch == 0:
            end = time.time()
            print(f"Epoch [{epoch}/{args.epochs}], "
                  f"Loss: {loss.item():.8f}, "
                  f"Time: {(end-start) * args.display_epoch:.1f}sec!")

            model.eval()

            correct_prediction = 0.0
            total = 0
            for i, (images, labels) in enumerate(test_loader):
                # to GPU
                images = images.to(device)
                labels = labels.to(device)
                # print prediction
                outputs = model(images)
                # equal prediction and acc
                _, predicted = torch.max(outputs.data, 1)
                # val_loader total
                total += labels.size(0)
                # add correct
                correct_prediction += (predicted == labels).sum().item()

            print(f"Acc: {(correct_prediction / total):4f}")

    # Save the model checkpoint
    torch.save(model, args.model_path + args.model_name)
    print(f"Model save to {args.model_path + args.model_name}.")


if __name__ == '__main__':
    train()

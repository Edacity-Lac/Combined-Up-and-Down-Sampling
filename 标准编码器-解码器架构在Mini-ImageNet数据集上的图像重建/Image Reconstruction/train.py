import torch.nn as nn
import torch
from torchvision.transforms import transforms
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
from torch import optim
import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import MultiStepLR
from model import RecoverNet
import random
from evaluate import evaluate
import wandb
from torchvision.datasets import ImageFolder
from argparse import Namespace


wandb.login(key='')#输入账号
cudnn.enabled = True

config = Namespace(
    lr=0.01,
    batch_size=128,
    epochs= 100,
    model_architecture ="RecoverNet",
    image_size=32,
    momentum=0.9,
    downsampler="carafe",
    upsampler="carafe"
)#输入参数与模式


def setup_seed(seed):
   torch.manual_seed(seed)
   os.environ['PYTHONHASHSEED'] = str(seed)
   torch.cuda.manual_seed(seed)
   torch.cuda.manual_seed_all(seed)
   random.seed(seed)
   torch.backends.cudnn.benchmark = False
   torch.backends.cudnn.deterministic = True
   torch.backends.cudnn.enabled = True
#随机种子


#train model
def train(config,seed_number):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = RecoverNet(downsampler=config.downsampler,upsampler=config.upsampler)
    model.to(device=device)
    experiment=wandb.init(project='Image Reconstructed',name=config.downsampler + config.upsampler,save_code=True,config=config)
    cudnn.benchmark = True
    transform = transforms.Compose([
        transforms.Resize(config.image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # 归一化处理
    ])



    trainset = ImageFolder(root='./mini-imagenet/train',transform=transform)
    testset = ImageFolder(root='./mini-imagenet/test',transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=config.batch_size, shuffle=True)  # 训练集
    testloader = torch.utils.data.DataLoader(testset, batch_size=config.batch_size, shuffle=True)  # 测试集

    print(trainloader)
    optimizer = optim.SGD(model.parameters(), lr=config.lr, momentum=config.momentum)
    scheduler = MultiStepLR(optimizer, milestones=[50, 70, 85], gamma=0.1)
    criterion = nn.L1Loss()#优化器设置

    for epoch in range(1, config.epochs + 1):#开始训练
        model.train()
        epoch_loss = 0
        scheduler.step()
        for batch in trainloader:
                images, target = batch
                images = images.to(device=device)
                predict = model(images)
                target = images.clone()
                target = target * 0.3081 + 0.1307
                target = target.to(device=device)
                loss=criterion(predict,target)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
        evaluate(model, testloader, criterion, device,experiment,epoch)
        print('epoch: %d seed_number: %d' % (epoch,seed_number))
    experiment.finish()


if __name__ == '__main__':
for i in range(10, 15):
    setup_seed(i)
    train(config,i)

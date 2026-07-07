# 
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

data_train=torchvision.datasets.MNIST('/data',  #数据地址
                download=True,
                transform=tansforms.Compose([transform.Resize((32,32)),transforme.ToTensor()]))
data_train=torchvision.datasets.MNIST('/data',  #数据地址
                train=False,
                download=True,
                transform=tansforms.Compose([transform.Resize((32,32)),transforme.ToTensor()]))
              
data_train_lodar=DataLoder(data_train,
                           batch_size=256,
                          shuffle=Ture, #随机打乱
                           num_workers=8)
data_test_lodar=DataLoder(data_test,batch_size=1024
                          ,num_workers=8)

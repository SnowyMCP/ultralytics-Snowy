# 数据集
 # 手写体识别数据
'''
1. root (类型: str 或 pathlib.Path):这是一个字符串或路径对象。
幕后：它指定了数据集在你的电脑硬盘上的“家”。必须是一个合法的文件夹路径（绝对路径或相对路径均可）。
常见写法：root='./data' 或 root='/User/dataset/mnist'

2. train (类型: bool, 默认值: True):：这是一个布尔值（只能是 True 或 False）。
幕后：它是一个“开关”。PyTorch 源码内部会根据这个布尔值去读取不同的二进制文件。
传 True 就会去读训练集文件（train-images-idx3-ubyte）。传 False 就会去读测试集文件（t10k-images-idx3-ubyte）。

3. transform (类型: callable, 默认值: None):这是一个可调用对象（函数、匿名函数 lambda，或者实现了 __call__ 方法的类实例）。
幕后：这是最核心的参数之一。MNIST 默认下载下来的是 PIL Image（Python 图像库格式）。如果你不传这个参数（保持 None），你拿到的就是原图，无法直接喂给神经网络。
常见做法：传入 transforms.ToTensor()（它是一个类实例，但它是可调用的）。这个“可调用对象”会在你每次使用 dataset[i] 读取图片时自动触发，把图片实时从 PIL 格式转换成 PyTorch 的 Tensor（张量），并将像素值归一化到 [0.0, 1.0] 之间。

4. target_transform (类型: callable, 默认值: None):同样是一个可调用对象。幕后：transform 是针对图片的，而 target_transform 是针对标签（Label）的。
例如，MNIST 的标签默认是整数 0-9。如果你想做独热编码（One-Hot Encoding），你可以写一个函数传给它，让它把数字 5 转换成 [0,0,0,0,0,1,0,0,0,0]。通常情况下我们不需要修改标签，所以保持默认的 None 即可。

5. download (类型: bool, 默认值: False):又一个布尔值开关。
幕后：控制是否执行下载逻辑。
True：代码会先去 root 路径下找有没有写好的数据集文件。如果有，直接加载；如果没有，就启动网络下载程序。
False：直接去 root 路径下找文件。如果找不到，程序就会直接报错（通常报 RuntimeError: Dataset not found.）。
'''

class torchvision.datasets.MNIST(
    root,            #本地保存数据集的地址
    train=True,        #它是一个“开关”。传 True 就会去读训练集文件（train-images-idx3-ubyte）。传 False 就会去读测试集文件（t10k-images-idx3-ubyte）。
    transform=None,     #这是一个可调用对象，MNIST 默认下载下来的是 PIL Image（Python 图像库格式）。如果你不传这个参数（保持 None），你拿到的就是原图，无法直接喂给神经网络。
    target_transform=None,  #可调用对象。
    download=False
)

#==========================#
#       数据集实例化        #
#==========================#
# DataLoader 会把这个数据集打乱（Shuffle）、分批次（Batch，比如一次打包 64 张图），变成模型真正能成批吞入的训练数据。
'''
分析：
  数据类型统一：
  （1.原始数据格式，类型转化）-->
  （2.数据集整体用法（训练，测试）定义）-->
   (3.定义数据加载器，统一训练，测试数据)
'''
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# 1. 定义图像转换
# MNIST images are PIL images by default; we need them as Tensors
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)) # Optional: Normalize with MNIST mean and std
])

# 2. 加载训练&测试数据
train_dataset = torchvision.datasets.MNIST(
    root='./data', 
    train=True, 
    transform=transform, 
    download=True
)

test_dataset = torchvision.datasets.MNIST(
    root='./data', 
    train=False, 
    transform=transform, 
    download=True
)

# 3. 创建数据加载器 batching and shuffling
train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)

#====================#
#       代码验证      #
#====================#

# 1. 查看数据集大小
print(len(mnist_dataset))  
# 输出: 60000

# 2. 获取第一张图片和它的标签
image, label = mnist_dataset[0]

# 3. 查看它们的类型和形状
print(type(image))  
# 输出: <class 'torch.Tensor'> （因为设置了 transform=ToTensor()）

print(image.shape)  
# 输出: torch.Size([1, 28, 28]) -> [通道数, 长度, 宽度]

print(label)        
# 输出: 5 （表示这张图片手写的数字是 5）


lista=[VOCSegmentstion,ImageNet,CocoDetection,]

class torchvision.datasets.CIFA10(
    root, 
    train=True, 
    transform=None, 
    target_transform=None, 
    download=False
)

#======================#
#    Deep learn        #
#======================#
'''
常见工程文件排列
project/
csrc/
data.py      #数据处理
utils.py     #参数模块，预处理模块
model.py     #网络
train.py     #模型训练
inference.py #模型推断
'''

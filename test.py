import torch
import torchvision
from torch.utils import data
from torchvision import transforms
import matplotlib.pyplot as plt

batch_size = 256
def get_dataloader_workers():
    """用4个进程读取数据"""
    return 4

def load_data_fashion_mnist(batch_size, resize=None):
    """下载数据集并加载到内存中"""
    trans = [transforms.ToTensor()]
    if resize is not None:
        trans.insert(0,transforms.Resize((resize,resize)))
    trans = transforms.Compose(trans)
    mnist_train = torchvision.datasets.FashionMNIST(
        root="./data", train=True, transform=trans,download=True
    )
    mnist_test = torchvision.datasets.FashionMNIST(
        root="./data", train=False, transform=trans,download=True
    )
    return (data.DataLoader(mnist_train, batch_size, shuffle=True,
                           num_workers=get_dataloader_workers()),
           data.DataLoader(mnist_test, batch_size, shuffle=False,
                           num_workers=get_dataloader_workers()))

# resize可以调整图片大小
tran_iter, test_iter = load_data_fashion_mnist(32,resize=64)
for X,y in train_iter:
    print(X.shape, X.dtype, y.shape, y.dtype)
    break
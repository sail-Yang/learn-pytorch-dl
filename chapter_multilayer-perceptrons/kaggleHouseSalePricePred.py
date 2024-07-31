import hashlib
import os
import tarfile
import zipfile
import requests
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils import data
import logging
import sys  
from datetime import datetime
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1" 

### 下载数据集
# 二元组（数据url，sha-1密钥）
DATA_HUB = dict()
DATA_URL = 'http://d2l-data.s3-accelerate.amazonaws.com/'
DATA_HUB['kaggle_house_train'] = ( DATA_URL + 'kaggle_house_pred_train.csv',
    '585e9cc93e70b39160e7921475f9bcd7d31219ce')

DATA_HUB['kaggle_house_test'] = ( DATA_URL + 'kaggle_house_pred_test.csv',
    'fa19780a7b011d9b009e8bff8e99922a8ee2eb90')

logger = None

def download(name, cache_dir=os.path.join('..','data')):
  """下载一共DATA_HUB中的文件，返回本地文件名"""
  assert name in DATA_HUB, f"{name} 不存在于 {DATA_HUB}"
  url, sha1_hash = DATA_HUB[name]
  os.makedirs(cache_dir, exist_ok=True)
  fname = os.path.join(cache_dir, url.split('/')[-1])
  if os.path.exists(fname):
    sha1 = hashlib.sha1()
    with open(fname, 'rb') as f :
      while True:
        data = f.read(1048576)
        if not data:
          break
        sha1.update(data)
    if sha1.hexdigest() == sha1_hash:
      return fname #命中缓存
  logger.info(f'正在从{url}下载{fname}...')
  r = requests.get(url, stream=True, verify=True)
  with open(fname, 'wb') as f:
    f.write(r.content)
  return fname

def download_extract(name, folder=None):
  """下载并解压zip/tar文件"""
  fname = download(name)
  base_dir = os.path.dirname(fname)
  data_dir, ext = os.path.splitext(fname)
  if ext == '.zip':
    fp = zipfile.ZipFile(fname, 'r')
  elif ext in ('.tar', '.gz'):
    fp = tarfile.open(fname, 'r')
  else:
    assert False
  fp.extractall(base_dir)
  return os.path.join(base_dir, folder) if folder else data_dir

def download_all():
  for name in DATA_HUB:
    download(name)
    
def log_rmse(net, features, labels):
  # 为了在取对数的时候进一步稳定该值,将小于1的值设置为1
  clipped_preds = torch.clamp(net(features),1,float('inf'))
  rmse = torch.sqrt(loss(torch.log(clipped_preds),
                         torch.log(labels)))
  return rmse.item()

def load_array(data_arrays, batch_size, is_train=True):
    """构造一个PyTorch数据迭代器"""
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)

def train(net, train_features, train_labels, test_features, test_labels, num_epochs, lr, weight_decay, batch_size):
  train_ls, test_ls = [], []
  train_iter = load_array((train_features, train_labels), batch_size)
  #Adam优化算法
  optimizer = torch.optim.Adam(net.parameters(),
                               lr = lr,
                               weight_decay = weight_decay)
  
  for epoch in range(num_epochs):
    for X,y in train_iter:
      optimizer.zero_grad()
      l = loss(net(X),y)
      l.backward()
      optimizer.step()
    train_ls.append(log_rmse(net, train_features, train_labels))
    if test_labels is not None:
      test_ls.append(log_rmse(net, test_features, test_labels))
  return train_ls, test_ls
  
def get_k_fold_data(k,i,X,y):
  assert k > 1
  fold_size = X.shape[0] // k
  X_train, y_train = None ,None
  for j in range(k):
    idx = slice(j * fold_size, (j+1) * fold_size)
    X_part, y_part = X[idx, :], y[idx]
    if j == i:
      X_valid, y_valid = X_part, y_part
    elif X_train is None:
      X_train, y_train = X_part, y_part
    else:
      X_train = torch.cat([X_train,X_part],0)
      y_train = torch.cat([y_train,y_part],0)
  return X_train, y_train, X_valid, y_valid

def k_fold(k, X_train, y_train, num_epochs, lr, weight_decay, batch_size):
  train_l_sum, valid_l_sum = 0,0
  for i in range(k):
    data = get_k_fold_data(k,i,X_train,y_train)
    net = get_net(X_train.shape[1])
    train_ls, valid_ls = train(net, *data, num_epochs, lr, weight_decay, batch_size)
    train_l_sum += train_ls[-1]
    valid_l_sum += valid_ls[-1]
    logger.info(f'折{i + 1}，训练log rmse:{float(train_ls[-1]):f}, '
              f'验证log rmse:{float(valid_ls[-1]):f}')
  return train_l_sum / k, valid_l_sum / k

def train_and_pred(train_features, test_features, train_labels, test_data, num_epochs, lr, weight_decay, batch_size):
  net = get_net(train_features.shape[1])
  train_ls, _ = train(net, train_features, train_labels, None, None, num_epochs, lr, weight_decay, batch_size)
  logger.info(f'训练log rmse: {float(train_ls[-1]):f}')
  #将网络应用于测试集
  preds = net(test_features).detach().numpy()
  #导出到Kaggle
  test_data['SalePrice'] = pd.Series(preds.reshape(1,-1)[0])
  submission = pd.concat([test_data['Id'], test_data['SalePrice']],axis=1)
  submission.to_csv('submission.csv',index=False)
  
def init_weights(m):
  if type(m) == nn.Linear:
    nn.init.normal_(m.weight, std=0.01)
    
def get_net(in_features):
  net = nn.Sequential(nn.Linear(in_features,8),
                      nn.ReLU(),
                      nn.Linear(8,1),
                      )
  net.apply(init_weights)
  return net

def setup_logging(log_dir, name, level=logging.INFO):  
    """  
    设置日志记录器，将输出重定向到指定的日志文件  
    :param log_dir: 日志文件所在目录
    :param name: 日志记录器的名称，通常设置为程序名  
    :param level: 日志级别  
    """
    # 确保日志目录存在  
    if not os.path.exists(log_dir):  
        os.makedirs(log_dir)
    
    # 构造完整的日志文件路径  
    log_file_path = os.path.join(log_dir, name)
    
    # 创建一个logger  
    logger = logging.getLogger(name)  
    logger.setLevel(level)  
  
    # 创建一个handler，用于写入日志文件  
    handler = logging.FileHandler(log_file_path)  
    handler.setLevel(level)  
  
    # 创建一个formatter，并设置日志的格式  
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')  
    handler.setFormatter(formatter)  
  
    # 给logger添加handler  
    logger.addHandler(handler)  
  
    # 可选：将标准输出和标准错误也重定向到日志  
    # 注意：这可能会干扰正常的交互式使用或脚本中的其他输出  
    # logging.basicConfig(level=level, handlers=[handler], format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')  
  
    return logger  

if __name__ == "__main__":
  # 获取当前脚本所在的目录的上级目录  
  base_dir = os.path.abspath(os.path.dirname(__file__))  
  log_dir = os.path.join(base_dir, '..', 'logs')
    
  # 获取程序名（不包含扩展名）  
  program_name = os.path.splitext(os.path.basename(sys.argv[0]))[0]  
  
  # 创建带时间戳的日志文件名  
  timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')  
  log_file_name = f"{program_name}_{timestamp}.log"  
  
  # 初始化日志记录器  
  logger = setup_logging(log_dir, log_file_name)
  
  train_data = pd.read_csv(download('kaggle_house_train'))
  test_data = pd.read_csv(download('kaggle_house_test'))
  #删除id列
  all_features = pd.concat((train_data.iloc[:,1:-1], test_data.iloc[:,1:]))
  # 若无法获得测试数据，可以用训练数据计算均值和标准差
  numeric_features = all_features.dtypes[all_features.dtypes != 'object'].index
  all_features[numeric_features] = all_features[numeric_features].apply(
    lambda x: (x-x.mean()) / (x.std())
  )
  # 在标准化数据后，所有均值消失，可以将缺失值设置为0
  all_features[numeric_features] = all_features[numeric_features].fillna(0)
  #对于离散值,自动拆成独热编码,比如一个特征M包含RL,RM两种值,那么就拆成特征M_RL和M_RM
  all_features = pd.get_dummies(all_features, dummy_na=True)
  # 转化成张量
  n_train = train_data.shape[0]
  loss = nn.MSELoss()
  train_features = torch.tensor(all_features[:n_train].values, dtype=torch.float32)
  test_features = torch.tensor(all_features[n_train:].values, dtype=torch.float32)
  train_labels = torch.tensor(
    train_data.SalePrice.values.reshape(-1,1), dtype=torch.float32
  )
  
  k , num_epochs, lr, weight_decay , batch_size = 5, 100, 0.2, 1e-3, 64
  train_l, valid_l = k_fold(k, train_features, train_labels, num_epochs, lr, weight_decay, batch_size)
  logger.info(f'{k}-折验证: 平均训练log rmse: {float(train_l):f}, '
      f'平均验证log rmse: {float(valid_l):f}')
  train_and_pred(train_features, test_features, train_labels, test_data, num_epochs, lr,weight_decay,batch_size)
  
  
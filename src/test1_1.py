# 1-1,结构化数据建模流程范例
import torch 
import torchkeras 
print("torch.__version__ = ", torch.__version__)
print("torchkeras.__version__ = ", torchkeras.__version__)

### 一，准备数据
"""
titanic数据集的目标是根据乘客信息预测他们在Titanic号撞击冰山沉没后能否生存。
结构化数据一般会使用Pandas中的DataFrame进行预处理。
"""

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import torch 
from torch import nn 
from torch.utils.data import Dataset,DataLoader,TensorDataset

dftrain_raw = pd.read_csv('C:/Users/WYM/Desktop/eat_pytorch_in_20_days/eat_pytorch_datasets/titanic/train.csv')
dftest_raw = pd.read_csv('C:/Users/WYM/Desktop/eat_pytorch_in_20_days/eat_pytorch_datasets/titanic/test.csv')
dftrain_raw.head(10)

"""
字段说明： 
* Survived:0代表死亡，1代表存活【y标签】
* Pclass:乘客所持票类，有三种值(1,2,3) 【转换成onehot编码】
* Name:乘客姓名 【舍去】
* Sex:乘客性别 【转换成bool特征】
* Age:乘客年龄(有缺失) 【数值特征，添加“年龄是否缺失”作为辅助特征】
* SibSp:乘客兄弟姐妹/配偶的个数(整数值) 【数值特征】
* Parch:乘客父母/孩子的个数(整数值)【数值特征】
* Ticket:票号(字符串)【舍去】
* Fare:乘客所持票的价格(浮点数，0-500不等) 【数值特征】
* Cabin:乘客所在船舱(有缺失) 【添加“所在船舱是否缺失”作为辅助特征】
* Embarked:乘客登船港口:S、C、Q(有缺失)【转换成onehot编码，四维度 S,C,Q,nan】
"""

"""
利用Pandas的数据可视化功能我们可以简单地进行探索性数据分析EDA（Exploratory Data Analysis）。
label分布情况
"""

ax = dftrain_raw['Survived'].value_counts().plot(kind = 'bar',
     figsize = (12,8),fontsize=15,rot = 0)
"""
dftrain_raw['Survived'].value_counts()统计 “未幸存（0）” 和 “幸存（1）” 的人数。
plot(kind='bar')将统计结果绘制成柱状图，设置了图表大小（12x8 英寸）、字体大小（15）、x 轴标签不旋转（rot=0）。
"""
ax.set_ylabel('Counts',fontsize = 15)
ax.set_xlabel('Survived',fontsize = 15)
plt.show()

ax = dftrain_raw['Age'].plot(kind = 'hist',bins = 20,color= 'purple',
                    figsize = (12,8),fontsize=15)
"""
dftrain_raw['Age']：从训练数据集（dftrain_raw）中提取Age列（乘客年龄数据）。
.plot(...)：pandas 的绘图方法，基于 matplotlib 绘制图表：
kind='hist'：指定绘制直方图（直方图用于展示连续型数据的分布特征，通过将数据划分为若干区间，统计每个区间的样本数量）。
bins=20：设置直方图的 “箱数”（即数据划分的区间数量）为 20。箱数越多，对年龄分布的展示越细致（例如 20 箱意味着将年龄范围大致平均分成 20 个区间）。
color='purple'：设置直方图的填充颜色为紫色。
figsize=(12,8)：设置图表的尺寸为宽 12 英寸、高 8 英寸。
fontsize=15：设置图表中文字（如坐标轴刻度）的字体大小。
返回的ax是 matplotlib 的Axes对象，用于后续调整图表细节。
"""
ax.set_ylabel('Frequency',fontsize = 15)
ax.set_xlabel('Age',fontsize = 15)
plt.show()

ax = dftrain_raw.query('Survived == 0')['Age'].plot(kind = 'density',
                      figsize = (12,8),fontsize=15)
"""
dftrain_raw.query('Survived == 0')：使用pandas的query方法筛选数据，只保留Survived列值为 0 的行（即 “未幸存” 的乘客）。
['Age']：从筛选结果中提取Age列（仅关注未幸存乘客的年龄数据）。
.plot(kind='density')：绘制密度图（也称为核密度估计图，Kernel Density Estimation）。密度图用于展示连续数据的概率分布，曲线越高表示该年龄出现的概率（或比例）越大，相比直方图更平滑，能更好地反映分布趋势。
其他参数：figsize=(12,8)设置图表大小，fontsize=15设置字体大小。
返回的ax是 matplotlib 的Axes对象，用于后续统一设置图表属性（如标签、图例）。
"""
dftrain_raw.query('Survived == 1')['Age'].plot(kind = 'density',
                      figsize = (12,8),fontsize=15)
"""
逻辑与上一行类似，但筛选条件为Survived == 1（即 “幸存” 的乘客），同样提取Age列并绘制密度图。
由于未指定新的Axes对象，这条曲线会叠加绘制在之前的图表上，形成两条曲线的对比（未幸存 vs 幸存）。
"""
ax.legend(['Survived==0','Survived==1'],fontsize = 12)
"""
通过ax.legend()为图表添加图例，['Survived==0','Survived==1']分别对应两条曲线的标签，明确区分 “未幸存” 和 “幸存” 的年龄分布。
fontsize=12设置图例字体大小。
"""
ax.set_ylabel('Density',fontsize = 15)
ax.set_xlabel('Age',fontsize = 15)
plt.show()

dftrain_raw # 显示数据集数据和格式

# 1. 预处理函数 preprocessing(dfdata)
def preprocessing(dfdata):
    """
    对泰坦尼克号数据集进行特征预处理，
    将原始数据转换为适合机器学习模型输入的格式，
    并划分出特征矩阵（x_train, x_test）和标签向量（y_train, y_test）
    """

    dfresult= pd.DataFrame() # # 创建一个空的 pandas DataFrame 对象，并将其赋值给变量 dfresult。作为 “容器” 逐步存储预处理后的特征，最终整合为完整的特征集。

    #Pclass 处理 Pclass 特征（船舱等级）
    dfPclass = pd.get_dummies(dfdata['Pclass']).astype(float)
    """
    用 pd.get_dummies() 将 1 个类别特征，拆分为 与类别数量相同的二进制特征（每列对应一个类别）
    用 .astype(float) 将 1/0 转换为 1.0/0.0，确保数据类型统一，避免后续模型因类型不匹配报错。
    """
    dfPclass.columns = ['Pclass_' +str(x) for x in dfPclass.columns ]
    """
    通过 ['Pclass_' + str(x) for x in dfPclass.columns] 生成新列名：
        原始列名 1 → 新列名 Pclass_1（代表 “一等舱”）；
        原始列名 2 → 新列名 Pclass_2（代表 “二等舱”）；
        原始列名 3 → 新列名 Pclass_3（代表 “三等舱”）。
    """
    dfresult = pd.concat([dfresult,dfPclass],axis = 1)

    #Sex
    dfSex = pd.get_dummies(dfdata['Sex']).astype(float)
    dfresult = pd.concat([dfresult,dfSex],axis = 1)
    """
    拼接后，dfresult 最终为：
    Pclass_1	Pclass_2	Pclass_3	female	 male
       1.0	      0.0	      0.0	      1.0 	 0.0
       0.0	      0.0	      1.0	      0.0	 1.0
    """

    #Age
    dfresult['Age'] = dfdata['Age'].fillna(0) # fillna(0) 是 pandas 中用于填充缺失值的方法，这里表示将所有缺失的年龄值（NaN）填充为 0。
    dfresult['Age_null'] = pd.isna(dfdata['Age']).astype(float) # pd.isna() 是 pandas 中判断值是否为缺失值（NaN）的函数，返回布尔值（True 表示缺失，False 表示不缺失）。
    """
    原始Age	        处理后 dfresult['Age']	    处理后 dfresult['Age_null']
    22.0	        22.0	                        0.0（不缺失）
    NaN	            0.0（填充）	                     1.0（缺失）
    35.0	        35.0	                        0.0（不缺失）
    """

    #SibSp,Parch,Fare
    dfresult['SibSp'] = dfdata['SibSp']
    dfresult['Parch'] = dfdata['Parch']
    dfresult['Fare'] = dfdata['Fare']

    #Carbin
    dfresult['Cabin_null'] =  pd.isna(dfdata['Cabin']).astype(float)

    #Embarked
    dfEmbarked = pd.get_dummies(dfdata['Embarked'],dummy_na=True).astype(float)
    dfEmbarked.columns = ['Embarked_' + str(x) for x in dfEmbarked.columns]
    dfresult = pd.concat([dfresult,dfEmbarked],axis = 1)

    return(dfresult)

# 2. 生成模型输入数据
x_train = preprocessing(dftrain_raw).values
y_train = dftrain_raw[['Survived']].values

x_test = preprocessing(dftest_raw).values
y_test = dftest_raw[['Survived']].values
# 3. 输出数据形状 
print("x_train.shape =", x_train.shape )
print("x_test.shape =", x_test.shape )

print("y_train.shape =", y_train.shape )
print("y_test.shape =", y_test.shape )

# 进一步使用DataLoader和TensorDataset封装成可以迭代的数据管道。
dl_train = DataLoader(TensorDataset(torch.tensor(x_train).float(),torch.tensor(y_train).float()),
                     shuffle = True, batch_size = 8)
dl_val = DataLoader(TensorDataset(torch.tensor(x_test).float(),torch.tensor(y_test).float()),
                     shuffle = False, batch_size = 8)
"""
（1）TensorDataset(...)：封装特征和标签
torch.tensor(x_train).float()：
将预处理后的训练集特征（x_train，numpy 数组）转换为 PyTorch 张量（tensor），并指定数据类型为float（PyTorch 模型通常使用 32 位浮点数进行计算，需与模型参数类型匹配）。
torch.tensor(y_train).float()：
同理，将训练集标签（y_train，numpy 数组）转换为float类型的张量。
组合效果：TensorDataset 将特征张量和标签张量按样本索引配对，形成一个包含（特征，标签）元组的数据集。例如，数据集的第 i 个元素是 (x_train[i], y_train[i])。
（2）DataLoader(...)：创建可迭代加载器
shuffle=True：
训练时启用数据打乱，每次迭代前随机打乱样本顺序。这是为了避免模型学习到数据的顺序依赖（如样本按某种规律排列），提高模型的泛化能力。
batch_size=8：
指定每个批次包含 8 个样本。训练时模型不会一次性处理所有数据，而是按批次计算梯度并更新参数（批处理可以平衡计算效率和梯度估计精度，8 是一个常见的中等批次大小）。
"""

# 测试数据管道
for features,labels in dl_train:
    print(features,labels)
    break


### 二，定义模型
"""
使用Pytorch通常有三种方式构建模型：使用nn.Sequential按层顺序构建模型，继承nn.Module基类构建自定义模型，继承nn.Module基类构建模型并辅助应用模型容器进行封装。
此处选择使用最简单的nn.Sequential，按层顺序模型。
"""

def create_net():
    net = nn.Sequential()
    net.add_module("linear1",nn.Linear(15,20))
    net.add_module("relu1",nn.ReLU())
    net.add_module("linear2",nn.Linear(20,15))
    net.add_module("relu2",nn.ReLU())
    net.add_module("linear3",nn.Linear(15,1))
    return net
    
net = create_net()
print(net)


### 三，训练模型
"""
Pytorch通常需要用户编写自定义训练循环，训练循环的代码风格因人而异。
有3类典型的训练循环代码风格：脚本形式训练循环，函数形式训练循环，类形式训练循环。
此处介绍一种较通用的仿照Keras风格的脚本形式的训练循环。
该脚本形式的训练代码与 torchkeras 库的核心代码基本一致。
torchkeras详情:  https://github.com/lyhue1991/torchkeras 
"""

import os,sys,time
import numpy as np
import pandas as pd
import datetime 
from tqdm import tqdm 

import torch
from torch import nn 
from copy import deepcopy
from torchkeras.metrics import Accuracy


def printlog(info):
    nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') # 获取当前时间并格式化
    print("\n"+"=========="*8 + "%s"%nowtime) # 打印分隔线和时间戳
    print(str(info)+"\n") # 打印核心信息
    

loss_fn = nn.BCEWithLogitsLoss()
"""
BCEWithLogitsLoss 是 PyTorch 中用于二元分类任务的损失函数，包含两个核心功能：
内置 Sigmoid 激活函数：将模型输出的原始 logits（未归一化的分数）转换为 [0,1] 区间的概率值（表示 “属于正类” 的概率）。
计算二元交叉熵损失：衡量预测概率与真实标签（0 或 1）之间的差异，损失值越小，预测越准确。
"""
optimizer= torch.optim.Adam(net.parameters(),lr = 0.005) # 根据损失函数的梯度，更新模型参数（如权重、偏置），以最小化损失函数。
metrics_dict = {"acc":Accuracy()} # 以字典形式存储评估指标（这里是准确率）

epochs = 20 
ckpt_path='checkpoint.pt' # 模型状态将保存到名为 checkpoint.pt 的文件中

#early_stopping相关设置
monitor="val_acc"# 早停需要监控的性能指标, val_acc 表示 “验证集准确率”（validation accuracy）。
patience=5 # “容忍次数”：当监控的指标（val_acc）连续多少个 epoch 没有提升时，才触发早停。
mode="max" # 监控指标的优化方向，max 表示 “希望指标越大越好”。

history = {} # history = {} 表示创建一个空的 Python 字典，并将其赋值给变量 history。

for epoch in range(1, epochs+1):
    printlog("Epoch {0} / {1}".format(epoch, epochs))

    # 1，train -------------------------------------------------  
    net.train() # 模型进入训练模式，确保 Dropout、BatchNorm 等层按训练逻辑工作（如 BatchNorm 计算均值和方差）。
    
    total_loss,step = 0,0
    
    loop = tqdm(enumerate(dl_train), total =len(dl_train),file = sys.stdout) # 创建进度条，实时显示训练进度和指标。
    train_metrics_dict = deepcopy(metrics_dict) # 深拷贝指标对象，避免训练和验证阶段的指标计算相互干扰（因为指标会累积计算，需要独立的实例）。
    
    for i, batch in loop: 
        """
        梯度计算流程：
            loss.backward()（求梯度）→ optimizer.step()（更新参数）→ optimizer.zero_grad()（清梯度），这是 PyTorch 参数更新的标准流程。
        """
        features,labels = batch
        """
        batch 是从 dl_train（训练数据管道）中迭代出的一个批次数据，由两部分组成：
        features：当前批次的特征张量（形状为 [batch_size, n_features]，如 [8, 13]，8 个样本，13 个特征）；
        labels：当前批次的标签张量（形状为 [batch_size, 1]，如 [8, 1]，每个样本对应 “是否幸存” 的标签 0 或 1）。
        将批次数据拆分为模型可直接使用的 “输入特征” 和 “监督标签”。
        """
        #forward 前向传播（模型预测与损失计算）
        preds = net(features) # 将特征输入模型，经过多层计算后得到原始预测结果（logits）。
        loss = loss_fn(preds,labels) # 表示当前批次的平均损失（损失越小，预测越准）。
        
        #backward
        loss.backward() # 计算梯度
        optimizer.step() # 更新模型参数
        optimizer.zero_grad() # 清空梯度
            
        #metrics
        step_metrics = {"train_"+name:metric_fn(preds, labels).item() 
                        for name,metric_fn in train_metrics_dict.items()} # 遍历 train_metrics_dict 中的每个键值对，计算当前批次的指标值，并构建新的字典。
        
        step_log = dict({"train_loss":loss.item()},**step_metrics) # 将当前批次的损失和指标整合到一个字典中，形成完整的 “批次日志”。

        total_loss += loss.item()
        
        step+=1
        # 区分中间批次和最后批次
        if i!=len(dl_train)-1:
            loop.set_postfix(**step_log) # 进度条会实时显示 “当前批次的即时指标”（如 train_loss=0.35, train_acc=0.875）。
        else:
            epoch_loss = total_loss/step
            epoch_metrics = {"train_"+name:metric_fn.compute().item() 
                             for name,metric_fn in train_metrics_dict.items()}
            epoch_log = dict({"train_loss":epoch_loss},**epoch_metrics)
            loop.set_postfix(**epoch_log)

            for name,metric_fn in train_metrics_dict.items():
                metric_fn.reset() # 清空这些累积值，避免当前 epoch 的计数影响下一个 epoch 的指标计算（确保每个 epoch 的指标都是独立的）
                
    for name, metric in epoch_log.items(): # 当前 epoch（轮次）的训练 / 验证指标（如损失、准确率）记录到history字典中
        history[name] = history.get(name, []) + [metric]
        

    # 2，validate -------------------------------------------------
    net.eval() #模型进入评估模式，确保 Dropout 不随机丢弃神经元，BatchNorm 使用训练阶段累积的均值和方差（而非实时计算），保证验证结果稳定。
    
    total_loss,step = 0,0
    loop = tqdm(enumerate(dl_val), total =len(dl_val),file = sys.stdout)
    
    val_metrics_dict = deepcopy(metrics_dict) 
    
    with torch.no_grad(): # 禁用梯度计算
        for i, batch in loop: 

            features,labels = batch
            
            #forward
            preds = net(features)
            loss = loss_fn(preds,labels)

            #metrics
            step_metrics = {"val_"+name:metric_fn(preds, labels).item() 
                            for name,metric_fn in val_metrics_dict.items()}

            step_log = dict({"val_loss":loss.item()},**step_metrics)

            total_loss += loss.item()
            step+=1
            if i!=len(dl_val)-1:
                loop.set_postfix(**step_log)
            else:
                epoch_loss = (total_loss/step)
                epoch_metrics = {"val_"+name:metric_fn.compute().item() 
                                 for name,metric_fn in val_metrics_dict.items()}
                epoch_log = dict({"val_loss":epoch_loss},**epoch_metrics)
                loop.set_postfix(**epoch_log)

                for name,metric_fn in val_metrics_dict.items():
                    metric_fn.reset()
                    
    epoch_log["epoch"] = epoch           
    for name, metric in epoch_log.items():
        history[name] = history.get(name, []) + [metric]

    # 3，early-stopping -------------------------------------------------
    arr_scores = history[monitor]
    """
    monitor是之前设置的监控指标（此处为"val_acc"，即验证集准确率）
    history[monitor]是history字典中存储该指标的列表，记录了从第 1 轮到当前轮的所有指标值
    """
    best_score_idx = np.argmax(arr_scores) if mode=="max" else np.argmin(arr_scores) # 定位历史记录中 “最佳指标” 出现的轮次
    if best_score_idx==len(arr_scores)-1:
        torch.save(net.state_dict(),ckpt_path)
        print("<<<<<< reach best {0} : {1} >>>>>>".format(monitor,
             arr_scores[best_score_idx]),file=sys.stderr)
    if len(arr_scores)-best_score_idx>patience:
        print("<<<<<< {} without improvement in {} epoch, early stopping >>>>>>".format(
            monitor,patience),file=sys.stderr)
        break 
    net.load_state_dict(torch.load(ckpt_path,weights_only=True)) # 加载最佳模型参数（为下一轮做准备）
    
dfhistory = pd.DataFrame(history) #  训练记录整理


### 四，评估模型
print(dfhistory) 

import matplotlib.pyplot as plt

def plot_metric(dfhistory, metric):
    train_metrics = dfhistory["train_"+metric]
    val_metrics = dfhistory['val_'+metric]
    epochs = range(1, len(train_metrics) + 1)
    plt.plot(epochs, train_metrics, 'bo--')
    plt.plot(epochs, val_metrics, 'ro-')
    plt.title('Training and validation '+ metric)
    plt.xlabel("Epochs")
    plt.ylabel(metric)
    plt.legend(["train_"+metric, 'val_'+metric])
    plt.show()

plot_metric(dfhistory,"loss")
plot_metric(dfhistory,"acc")


### 五，使用模型
#预测概率
y_pred_probs = torch.sigmoid(net(torch.tensor(x_test[0:10]).float())).data # 使用训练好的模型对测试集中的前 10 个样本进行预测，并将模型输出转换为概率值
print(y_pred_probs)

#预测类别 将预测概率（y_pred_probs）转换为具体的类别标签（0 或 1）
y_pred = torch.where(y_pred_probs>0.5,
        torch.ones_like(y_pred_probs),torch.zeros_like(y_pred_probs))
"""
torch.where(condition, x, y)
这是 PyTorch 中的条件选择函数，作用是：
遍历张量中的每个元素，判断是否满足condition（条件）；
若满足，取x中对应位置的元素；
若不满足，取y中对应位置的元素。
"""
print(y_pred)


### 六，保存模型
"""
Pytorch 有两种保存模型的方式，都是通过调用pickle序列化方法实现的。
    第一种方法只保存模型参数。
    第二种方法保存完整模型。
推荐使用第一种，第二种方法可能在切换设备和目录的时候出现各种问题。
"""

# **1，保存模型参数(推荐)**
print(net.state_dict().keys()) # 打印出模型中所有可学习参数（权重、偏置等）的名称（键）

# 保存模型参数
torch.save(net.state_dict(), "./data/net_parameter.pt")
"""
作用：将训练好的模型net的参数（权重、偏置等）保存到指定路径./data/net_parameter.pt。
细节：
    net.state_dict()返回模型的参数字典（键为参数名称，值为参数张量）；
    torch.save(...)将参数字典序列化并保存为.pt文件（PyTorch 的标准参数文件格式）；
    保存的是 “参数” 而非整个模型结构，因此后续加载时需要先创建相同结构的模型。
"""
net_clone = create_net() # 创建一个与原始模型net结构完全相同的新模型net_clone（但参数是随机初始化的，未训练状态）
net_clone.load_state_dict(torch.load("./data/net_parameter.pt",weights_only=True)) # 将之前保存的参数文件加载到克隆模型net_clone中，使克隆模型拥有与原始模型net完全相同的参数（即具备相同的预测能力）。

print(torch.sigmoid(net_clone.forward(torch.tensor(x_test[0:10]).float())).data) # 用加载了参数的克隆模型对测试集前 10 个样本进行预测，得到与原始模型net完全相同的预测概率。

# **2，保存完整模型(不推荐)**
torch.save(net, './data/net_model.pt')
"""
保存整个模型对象，包括：
    模型的结构（各层的定义，如nn.Linear、nn.ReLU的排列组合）；
    模型的参数（与state_dict内容一致）；
    模型的其他属性（如训练 / 评估模式状态）
"""
net_loaded = torch.load('./data/net_model.pt',weights_only=False)
print(torch.sigmoid(net_loaded(torch.tensor(x_test[0:10]).float())).data)
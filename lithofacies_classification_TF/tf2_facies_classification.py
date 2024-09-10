
import os
import pandas as pd
import numpy as np
import tensorflow as tf
#import keras
import time
import math
import seaborn as sns
import statsmodels.api as sm
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import tensorflow.keras.backend as Kbackend

# pydot_ng 用于绘制网络图
import pydot_ng as pg
from mpl_toolkits.axes_grid1 import make_axes_locatable
# from sklearn.model_selection import train_test_split
from sklearn import model_selection
from sklearn.preprocessing import scale
from sklearn.metrics import mean_absolute_error
from pandas import set_option
# from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.callbacks import ReduceLROnPlateau
from pylab import *
from scipy import interpolate
from pygame import mixer 



# calculate RMSE
from sklearn.metrics import mean_squared_error, confusion_matrix
import matplotlib.colors as colors
# sklearn.metrics.mean_squared_error(y_true, y_pred, sample_weight=None, multioutput='uniform_average', squared=True)
# https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_error.html#sklearn.metrics.mean_squared_error


# tensorflow : 2.2.0  ; keras:  


print(tf.__version__)


# numpy+mkl 版本为17.2  
# tensorboard，tensorflow 版本为2.1.0  
# pydot版本为1.4.1, graphviz 版本为0.13.2

# ## 导入自己的包


import senutil as sen
# from rbflayer import RBFLayer, InitCentersRandom
import senmodels_classification as smsc



mpl.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号
set_option("display.max_rows", 15)
set_option('display.width', 200)
np.set_printoptions(suppress=True, threshold=5000)



# 使用GPU训练时候，打开下面注释
os.environ['CUDA_VISIBLE_DEVICES']='-1'


# # 准备工作（数据处理，结果保存位置定义）

# ## 定义数据集所在位置

# ### 训练集所在位置


TrainDataPath = 'lithofacies_data/train/'


# filename_AB:  
# (1) facies_vectors.csv


filename_AB = 'GY-1_lithofacies_vectors_0.1.csv'      # 地质4口卡奔图-1-58_all_数据均衡.csv  train0413use_all.csv  train0415use_all.csv
# facies_vectors.csv  地质4口卡奔图-1-58_all_R_0.1.csv  古页1_R_0.1m_整段_1500m-2746m.csv（13种岩相）
# 地质4口卡奔图-1-58_all_R_0.02.csv  古页1_all_R_0.02-全.csv  英x58地质岩相-0512_facies_vectors_0.01.csv  古页1地质岩相-0512_facies_vectors_0.1.csv
TrainDataPath = os.path.join(TrainDataPath,filename_AB)
print(TrainDataPath)


# ### 预测阶段测试集所在位置


TestDataPath = 'lithofacies_data/test/'


# data/exp_curve_reconstract/exp_1/test/;   data/exp_curve_reconstract/val/

# filename_A为预测曲线对应的常规曲线数据
# filename_A : nofacies_data.csv


filename_A =  'YX-58_lithofacies_vectors_0.1.csv'
# YX-58_lithofacies_vectors_0.02.csv
# 英x58地质岩相-0512_facies_vectors_0.01.csv  古页19_0428_R_0.1m.csv  古页17_0520_R_0.125m.csv  古页39_0428_R_0.1m.csv
# 奥-34井_facies_vectors_0.1.csv   # Z-2911_lithofacies_vectors_0.1.csv  第三次训练  古页12_0428_R_0.1m.csv
# 无矿物成分：古页12_R_0.1m_2050-2427m.csv    古页15_R_0.1m_整段_2050m-2530m.csv  古页2HC_R_0.1m_整段_2081m-2457m.csv
# 有矿物成分曲线： 古页12_R_0.1m_0415测试.csv  古页15_R_0.1m_0415测试.csv 松页油2_0428_R_0.125m.csv



addR_well_name = filename_A.split(".")[0]
TestDataPath = os.path.join(TestDataPath,filename_A)



use_low_R_data = False


# ### 真实岩相数据集所在位置

# 用于真实岩相数据集与预测数据吻合度，实际中可能没有


use_high_R_data = True #True # False



# 高分辨率相对于低分辨率的倍数, default value = 10 (对应于0.1m); 8 (对应于0.125m)
resolution = 8



HighRDataPath = 'lithofacies_data/test/'


# filename_C_H为高分辨率元素曲线 filename_C_H: (1) YT2_YS_0.1m_shang.csv ; (2) YT2_YS_0.1m_xia.csv 


filename_C_H = 'YX-58_lithofacies_vectors_0.1.csv'
# 古页1地质岩相-0512_facies_vectors_0.1.csv  英x58地质岩相-0512_facies_vectors_0.01.csv
# 奥-34井_facies_vectors_0.02.csv   # 赵-2911井_facies_vectors_0.02.csv
# 奥-34井_facies_vectors_0.1.csv  古页1+英斜58地质岩相-0512_facies_vectors_0.05.csv



HighRDataPath = os.path.join(HighRDataPath,filename_C_H)


# # 模型定义

# ## 定义自变量

# 定义要输入的维度AC、CNL、DEN、GR、RD、RS等


# input_vectors = ["AC","CNL","DEN","GR","RD","RS"]
# input_vectors = ["AC","CNL","DEN","GR","RLLD","RLLS"]
# input_vectors = ["CAL","SP","GR","CNL","DT","DEN","MSFL","RS","RD"]
input_vectors = ["GR","CNL","DT","DEN","MSFL","RS","RD"]

# input_vectors = ["GR","CNL","DT","DEN","RS","RD"]
# input_vectors = ["GR","CNL","DT","MSFL",'DWCALC','DWCLAY','DWDOLO','DWPLAG','DWQUAR']
# input_vectors = ["GR","CNL","DT","MSFL","RD","RS"]
# input_vectors = ["GR","CNL","DT","MSFL"]
# input_vectors = ["GR","ILD_log10","DeltaPHI","PHIND","PE","NM_M","RELPOS"]


# ## 定义因变量

# 定义要训练的参数模型
# 读取岩相所在列训练数据，包括'SS', 'CSiS', 'FSiS', 'SiSh', 'MS','WS', 'D','PS', 'BS'等


# 1=sandstone  2=c_siltstone   3=f_siltstone 
# 4=marine_silt_shale 5=mudstone 6=wackestone 7=dolomite
# 8=packstone 9=bafflestone
# facies_colors = ['#F4D03F', '#F5B041','#DC7633','#6E2C00',
#        '#1B4F72','#2E86C1', '#AED6F1', '#A569BD', '#196F3D']



# facies_colors = ['#632423','#007F00', '#999999','#339966','#99CC00','#00FF00','#7F7F7F','#FFCC99','#FFCC00','#993366','#FF9900', '#FF6600','#00CCFF']

# facies_labels = ['SS', 'CSiS', 'FSiS', 'SiSh', 'MS','WS', 'D','PS', 'BS']

facies_colors = ['#632423', '#0070C0','#00B0F0','#75DAFF','#00B050','#FFC000', '#FFFF00']
facies_labels = ['高有机质层状页岩相', '高有机质纹层状页岩相','中有机质纹层状页岩相','低有机质纹层状页岩相',
                '中低有机质块状白云岩相', '低有机质块状介壳灰岩相', '低有机质块状粉砂岩相']
# 13类
# facies_labels = ['其他','中有机质层状粘土质页岩','低有机质纹层状粘土质页岩','高有机质层状粘土质页岩','中有机质纹层状粘土质页岩','高有机质纹层状粘土质页岩','低有机质层状粘土质页岩', '低有机质纹层状长英质页岩','中有机质层状长英质页岩','高有机质层状长英质页岩','中有机质纹层状长英质页岩','高有机质纹层状长英质页岩', '介壳灰岩']

# 不考虑有机质——6类
# facies_colors = ['#00B0F0','#75DAFF','#00B050','#FFC000', '#FFFF00','#007F00']

# facies_labels = ['泥质粉砂岩', '层状粘土质页岩','纹层状粘土质页岩','层状长英质页岩','纹层状长英质页岩','灰岩、云岩']


element_name = str(len(input_vectors))  # SS | CSiS

# 亚相分类
# adjacent_facies = np.array([[1], [0,2], [1], [4], [3,5], [4,6,7], [5,7], [5,6,8], [6,7]])

# adjacent_facies = np.array([[0],[1], [2], [1,2,3], [4], [5], [5,6]])

# [12],[10], [6], [4],[3,5]



# adjacent_facies = np.array([[12],[11,10,7], [9,8,6], [5,4,2],[1,3] [0]])

adjacent_facies = np.array([[4],[0,1], [2,3], [3] [0]])

# 岩相所在列
facies_labels_col = "Facies"
# 深度列名称
DEPTH_col_name = "DEPTH"

#facies_color_map is a dictionary that maps facies labels
#to their respective colors
facies_color_map = {}
for ind, label in enumerate(facies_labels):
    facies_color_map[label] = facies_colors[ind]

def label_facies(row, labels):
    return labels[ row[facies_labels_col] -1]


# 样本权重
weight_coloum = "sample_weight"
class_weight = None #  {0:0.22, 1:0.01,2:0.02,3:0.03,4:0.30,5:0.38,6:0.04}   
# None |  'auto' | {0:0.22, 1:0.01,2:0.02,3:0.03,4:0.30,5:0.38,6:0.04} 

# ## 选择要使用的模型

# 择要使用的模型类型model_type:  
# (1)'RBF'(flag = 1);   
# (2)'DNN'(flag = 2);  
# (3)'LSTM','GRU','GRU2','DNN_2'(flag = 3),Capsule;  
# (4)'BLSTM', 'BGRU'(flag = 3),'MyWaveNet','BLSTM-Atten','BiLSTM-Atten2'，'BiLSTM-Atten3'，'BiLSTM-Atten4','BiLSTM-Atten5','BiLSTM-Atten6','BiGRU-Atten','BiGRU-Atten2'; 目前最好几种   
# (5)'WaveNet','MyUNet' ,CNN_Atten , 'BiGRU-self-Atten'    
# (6) (暂未完成)  'NAS','IndyGRU','LSTM-GRU','IndyLSTM','UGRNNCell';


# BLSTM ； MyWaveNet; BLSTM-Atten   Bilstm_Capsule
# model_type =  'DNN'  # 'MyWaveNet'   'MyUNet'   'Bilstm_Capsule'
# model_type = 'BiGRU-self-Atten'   # Bigru_Multihead_Self_Atten Bigru_Multihead_Self_Atten   Multihead_Self_Atten(Bilstm)
# model_type = 'RBF'  # Bigru_Multihead_Self_Atten_RBF
# model_type = 'Bigru_Multihead_Self_Atten_RBF'  # (1) MyWaveNet;  (4)BiLSTM-Atten5; (6) BiGRU-Atten2;
# model_type = 'Bigru_Multihead_Self_Atten_DNN'   # Bigru_Multihead_Self_Atten_DNN

model_type = 'Double_Expert_Net'

# ## 设置和模型相关的参数

flag = 0

if model_type == "RBF":
    flag = 1
elif model_type == "DNN":
    flag = 2
elif model_type == "DNN_2":
    flag = 3
elif model_type == "Capsule":
    flag = 3
else:
    flag = 3

# ## 初始化训练模型结构参数

# 网络结构参数来自与senmodels模块


# MAX_SAMPLE_NUM = 2000
# 输入维度
data_dim = len(input_vectors)
seq_length = 10 # 序列长度数 default 10,  TT1:4  J404:8   default:20
hidden_dim = 49 # 隐藏层神经元数 default 49 |   20  16   24  49
# 输出维度
output_dim = len(facies_labels)
n_layers = 3 # LSTM layer 层数 default 4  
dropout_rate = 0.4   # 在训练过程中设置 default 0.2  ,Na,0.4

# 修改下面的参数不改变网络
learning_rate = 0.0002  # default 0.002 0.01, 优选：0.005 ；0.008 可用0.0008 0.0005 0.001
# batch_size = 100 Na:
BATCH_SIZE = 100
# iterations = 300
EPOCHS = 30

# input_vectors_dim = len(input_vectors)
# 是否使用batch_size_strategy default False , | True
batch_size_strategy = False

model_para = smsc.MyModelParameter(data_dim,seq_length, hidden_dim, output_dim,learning_rate,dropout_rate,n_layers,BATCH_SIZE,EPOCHS)
model_para.n_layers


# ## 设定是训练操作还是测试操作
# 模型有两种阶段：  "train"(训练) | "test"(测试)

model_stage = "train"
model_stage = "test"

# 训练模型是否使用样本权重
train_use_weight = False

train_use_class_weight = False


# 是否训练分辨率增加模型，default:False | True


train_add_R_model = False



if model_stage == "train":
    well_name = filename_AB.split(".")[0]
    train_well_name = filename_AB.split(".")[0]
else:
    well_name = filename_A.split(".csv")[0]
    train_well_name = filename_AB.split(".")[0]
print(well_name,train_well_name)



# 训练完成是否播放音乐 True | False
paly_music = True



# 是否保存训练日志：default : False | True， 保存日志可以用tensorboard显示实时计算日志，但是日志文件占用空间
save_logs = False #True
# 训练日志保存位置
log_path = os.path.join("logs/")

# 两种学习率适应方法:  default = 0
#(1)每隔10个epoch，学习率减小为原来的1/10, set value = 1;
#(2)当学习停滞时，减少2倍或10倍的学习率常常能获得较好的效果。value = 2
# (3) 
learning_rate_deacy_policy = 2



use_semi_seqlength = True



# 是否绘制模型图 default = False；  Value：True | False
plot_modelnet = True


# ## 训练阶段模型保存位置


print(model_stage)



Date = sen.tid_date()
child_dir_name = train_well_name + '_Seq_'+ str(seq_length)+  "/"
custom_model_child_dir = "facies_Seq_8_WaveNet/"



# 设置模型保存的文件夹
model_save_path = os.path.join("model/", 'facies_' + model_type.lower() + "_train/")
#model_save_path = os.path.join("model/", 'facies_' + model_type.lower() + "_train/",child_dir_name)
if os.path.exists(model_save_path):
    model_path = model_save_path
else:
    os.mkdir(model_save_path)
    model_path = model_save_path
print(model_path)



model_name = train_well_name + "_" + model_type.lower() + "_" + str(len(input_vectors)) + '-facies_'+ str(n_layers) + "_layers_" +  "_lr_" + str(
        learning_rate) + "h_dim" + str(hidden_dim) + "_seq_length_" + str(seq_length) + "_epoch_" + str(EPOCHS) + ".h5"
model_file = model_path + model_name
print("model_name:",model_name)
print("model_file:",model_file)



# 定义模型保存的json_name
json_name = train_well_name + "_" + model_type.lower() + "_" + str(len(input_vectors)) + '-facies_'+ '_'+ str(n_layers) + "_layers_" +  "_lr_" + str(
        learning_rate) + "h_dim" + str(hidden_dim) + "_seq_length_" + str(seq_length) + "_epoch_" + str(EPOCHS) + ".json"
model_json = model_path + json_name
print(model_json)


# ## 测试操作加载模型存放位置


if model_stage == "test":   
    pred_model_json = model_json
    pred_model_file = model_file
    # custom_model_json =  ""
    # custom_model_file =  ""
    
    # pred_model_json = os.path.join(model_path,custom_model_json)
    # pred_model_file = os.path.join(model_path,custom_model_file)
    
    if not (os.path.exists(pred_model_json) and os.path.exists(pred_model_file)):
        print("预测模型不存在，程序结束，请训练相应模型")
        exit()
        


# ## 定义算法结果图表文字保存位置


if model_stage == "train":
    # well_name = filename_AB.split("_")[0]
    # begin_depth = depth_log[0][0]
    # end_depth = depth_log[-1][0]
    
    training_img_file_saving_path = 'model_training_images/'
    if not os.path.exists(os.path.join(training_img_file_saving_path,child_dir_name)):
        os.mkdir(os.path.join(training_img_file_saving_path,child_dir_name))
    model_training_img_file_saving_path = os.path.join(training_img_file_saving_path, child_dir_name, model_type.lower())
    model_training_img_name =  model_type + "_" + well_name + "_"+ element_name
    
    if not os.path.exists(model_training_img_file_saving_path):
        os.mkdir(model_training_img_file_saving_path)



if model_stage == "test": 

    testing_img_file_saving_path = 'model_testing_images/'
    model_testing_image_name =  model_name.split(".h5")[0] + "_" + well_name + "_" + element_name    # model_type + "_" + well_name + "_" + element_name
    
    model_testing_img_file_saving_path = os.path.join(testing_img_file_saving_path, child_dir_name, model_type.lower())
    if not os.path.exists(os.path.join(testing_img_file_saving_path,child_dir_name)):
        os.mkdir(os.path.join(testing_img_file_saving_path,child_dir_name))
    if not os.path.exists(model_testing_img_file_saving_path):
        os.mkdir(model_testing_img_file_saving_path)



font={'family':'SimHei',
     'style':'italic',
    'weight':'normal',
      'color':'red',
      'size':16
}



csv_file_saving_path = os.path.join("facies_csv_results/")
if model_stage == "test": 
    csv_file_saving_path = os.path.join("facies_csv_results/", model_type.lower() + "_test/")
else:
    csv_file_saving_path = os.path.join("facies_csv_results/", model_type.lower() + "_train/")
if not os.path.exists(csv_file_saving_path):
    os.mkdir(csv_file_saving_path)
print(csv_file_saving_path)


# # 数据加载及处理

# 调用pandas的read_csv()方法时，默认使用C engine作为parser engine，而当文件名中含有中文的时候，用C engine在部分情况下就会出错。所以在调用read_csv()方法时指定engine为Python就可以解决问题了。


if model_stage == "train":
    AB_use = pd.read_csv(TrainDataPath,engine='python',encoding='GBK')
    AB_use  = AB_use.dropna()
    print("Training data loading....")
else:
    AB_use = pd.read_csv(TrainDataPath,engine='python',encoding='GBK')
    AB_use  = AB_use.dropna()
    print("载入训练数据岩相....")
    A_read = pd.read_csv(TestDataPath,engine='python',encoding='GBK')
    A_read  = A_read.dropna()
    print("Testing data loading....")



if model_stage == "train":
    # print(AB_use)
    print(len(AB_use.columns))
    a = AB_use.columns.tolist()
else:
    # print(A_read)
    print(len(A_read.columns))
    a = A_read.columns.tolist()



# if model_stage == "train":
#     print(set(AB_use['Well Name']),DEPTH_col_name)



# AB_use



facies_colors



def make_facies_log_plot(logs, facies_colors):
    #make sure logs are sorted by depth
    logs = logs.sort_values(by = DEPTH_col_name)
    cmap_facies = colors.ListedColormap(
            facies_colors[0:len(facies_colors)], 'indexed')
    
    ztop=logs[DEPTH_col_name].min(); zbot=logs[DEPTH_col_name].max()
    
    cluster = np.repeat(np.expand_dims(logs[facies_labels_col].values,1), 100, 1)
    total_fig_cols = len(input_vectors)+ 1
    f, ax = plt.subplots(nrows=1, ncols = total_fig_cols, figsize=(total_fig_cols * 2,15))
    for i in range(len(input_vectors)):
        ax[i].plot(logs[input_vectors[i]], logs[DEPTH_col_name])
    final_line = len(input_vectors)
    im = ax[final_line].imshow(cluster, interpolation='none', aspect='auto',
                               cmap=cmap_facies,vmin = 0,vmax = len(facies_colors))
    
    divider = make_axes_locatable(ax[final_line])
    cax = divider.append_axes("right", size="25%", pad=0.05)
    # cax = divider.append_axes("right", size="20%", pad=0.05)
    cbar = plt.colorbar(im, cax=cax)
    # cbar.set_label((17*' ').join(facies_labels))
    cbar.set_label((4*' ').join(facies_labels))
    cbar.set_ticks(range(0,1)); cbar.set_ticklabels('')
    
    for i in range(len(ax)-1):
        ax[i].set_ylim(ztop,zbot)
        ax[i].invert_yaxis()
        ax[i].grid()
        ax[i].locator_params(axis='x', nbins=3)
        ax[i].set_xlabel(input_vectors[i])
        ax[i].set_xlim(logs[input_vectors[i]].min(),logs[input_vectors[i]].max())
        
    ax[final_line].set_xlabel('Facies')
    
    for i in range(len(ax)-1):
        ax[i].set_yticklabels([]);
        
    ax[final_line].set_yticklabels([])
    ax[final_line].set_xticklabels([])
    f.suptitle('Well: %s'%logs.iloc[0]['Well Name'], fontsize=14,y=0.94)



use_depth_log = False
if model_stage == "train":
    # 绘制训练段曲线
    Y_ele = AB_use.loc[:, facies_labels_col]
    if "DEPTH" in a:
        use_depth_log = True
        if "Well Name" in a:
            # well = set(AB_use['Well Name'])
            well = np.unique(AB_use['Well Name'])
            for k in well:
                depth_log = AB_use[AB_use['Well Name'] == k].loc[:, ["DEPTH"]]
                depth_log = np.array(depth_log)
                cucao_depth = np.array(depth_log)
                cucao_depth.shape = (len(cucao_depth),)
                begin_depth = depth_log[0][0]
                end_depth = depth_log[-1][0]
                print("Well Name",k)
                print("begin_depth",begin_depth,"end_depth",end_depth)
                make_facies_log_plot(AB_use[AB_use['Well Name'] == k],facies_colors)
        else:
            depth_log = AB_use.loc[:, ["DEPTH"]]
            depth_log = np.array(depth_log)
            cucao_depth = np.array(depth_log)
            cucao_depth.shape = (len(cucao_depth),)
            begin_depth = depth_log[0][0]
            end_depth = depth_log[-1][0]
            # print("Well Name",k)
            print("begin_depth",begin_depth,"end_depth",end_depth)
            make_facies_log_plot(AB_use,facies_colors)
            #plt.show()
        depth_log = np.array(AB_use.loc[:, ["DEPTH"]].reset_index(drop=True))
    else:
        print("No Depth Information")
else:
    if "DEPTH" in a:
        use_depth_log = True
        depth_log = A_read.loc[:, ["DEPTH"]]
        depth_log = np.array(depth_log)
        cucao_depth = np.array(depth_log)
        cucao_depth.shape = (len(cucao_depth),)
        if use_low_R_data == True:
            begin_depth = depth_log[0][0]    
        else:
            begin_depth = depth_log[seq_length][0]
        end_depth = depth_log[-1][0]
        print("begin_depth",begin_depth,"end_depth",end_depth)
    else:
        print("No Depth Information,all method is end!!!")
        # exit()
print("use_depth_log:",use_depth_log)



facies_labels[-1]


# ## 设置自变量应变量数据

# 电阻率曲线取对数值


if model_stage == "train":
    inputY = AB_use.dropna().reset_index(drop = True).loc[:, facies_labels_col]
    inputY_calc = AB_use.dropna().reset_index(drop = True).loc[:, facies_labels_col]
    inputX = AB_use.dropna().reset_index(drop = True).loc[:,input_vectors]
    # print(inputY)
    # print(inputY_calc)
    if train_use_weight == False:
        sample_weight = np.ones(len(inputY))
        # print(AB_Y)
    else:
        # 设定权重矩阵
        sample_weight = AB_use.loc[:, weight_coloum]
else:
    inputX = A_read.dropna().reset_index(drop = True).loc[:,input_vectors]
    inputY = AB_use.dropna().reset_index(drop = True).loc[:, facies_labels_col]


# ### 电阻率取对数


electric_log = ["RD","RS","RLLD","RLLS","MSFL"]
for item in electric_log:
    if item in input_vectors:
        print(item)
        for i in range(len(inputX)):
            if (inputX.loc[:,item][i]) <= 0.01:
                inputX.loc[:,item][i] = 0.01 
        inputX.loc[:,item] = np.log10(inputX.loc[:,item])
    



constant_value = 5


# ### 线性变换渗透率范围


if "渗透率" in a:
    inputY.loc[:,element_name] = np.log10(inputY.loc[:,element_name]) + constant_value
    inputY_calc.loc[:,reference] = np.log10(inputY_calc.loc[:,reference]) + constant_value


# ## 训练数据类别分布


if model_stage == "train":
    AB_use.dropna().reset_index(drop = True)


# ## 归一化操作


# 设置输入曲线范围,由于取了对数，所以RD和RS设置下限-1
DT = [40,200]
# AC = [140,300]
AC = [40,300]
CAL = [4,25]
CNL = [0,50]
CNC = [0,45]
DEN = [1,3]
ZDEN = [1,3]
GR = [50,280]
PE = [0,40]
RD = [0,5]
RS = [0,5]
RLLD = [-5,5]
RLLS = [-5,5]
MSFL = [-1,4]
HRD = [-6,6]
HRM = [-6,6]
ILD_log10 = [-5,5]
DeltaPHI = [-30,30]
PHIND = [0,100]
PE = [0,20]
NM_M = [0,2]
RELPOS = [0,1]


DWCALC = [0.01, 0.8] 
DWCLAY = [0.01, 0.8] 
DWDOLO = [0.01, 0.8] 
DWPLAG = [0.01, 0.8] 
DWQUAR = [0.01, 0.8]



# # 测井曲线计算的物性参数
# POR_CALC = [0,17]
# SW_CALC = [4,100]
# PERM_CALC = [-3 + constant_value, 2 + constant_value]

# # 岩石物理实验测量的物性参数
# POR = [0,17]
# SW = [4,100]
# PERM = [-3+ constant_value, 2 + constant_value]

# DTXX = [100,300]
# DTC = [40,300]
# DTS = [40,600]


# 创建字典，根据输入的数据维度调正归一化的内容


# u_log = {"AC": AC, "CNL": CNL, "DEN": DEN, "GR": GR, "RD": RD, "RS": RS,"RLLD": RLLD, "RLLS": RLLS}
# e_log = {"孔隙度": POR, "饱和度": SW, "渗透率": PERM}
# e_CALC_log = {"POR": POR_CALC, "SW": SW_CALC, "PERM": PERM_CALC}

u_log = {"DT":DT,"AC": AC,"CNL": CNL, "DEN": DEN, "GR": GR, "RD": RD, "RS": RS,"MSFL":MSFL, "RLLD": RLLD, "RLLS": RLLS,"ILD_log10":ILD_log10,"DeltaPHI":DeltaPHI,"PHIND":PHIND,"PE":PE,"NM_M":NM_M,"RELPOS":RELPOS, 
          'DWCALC': DWCALC,'DWCLAY': DWCLAY,'DWDOLO':DWCLAY,'DWPLAG':DWCLAY,'DWQUAR':DWCLAY}
# e_log = {"DTS":DTS,"DTXX":DTXX,"AC": AC, "CNL": CNL, "DEN": DEN, "GR": GR, "RD": RD, "RS": RS,"RLLD": RLLD, "RLLS": RLLS}
# e_CALC_log = {"DT":DT,"DTS":DTS,"DTC":DTC,"AC": AC, "CNL": CNL, "DEN": DEN, "GR": GR, "RD": RD, "RS": RS,"RLLD": RLLD, "RLLS": RLLS}



u_log_name = []
# 关键在于 input_vectors''
for i in input_vectors:
    u_log_name.append(u_log[i])
    
u_log_name



# e_log_name = []
# for i in element:
#     e_log_name.append(e_log[i])
# e_log_name



# e_calc_log_name = []
# for i in reference:
#     e_calc_log_name.append(e_CALC_log[i])
# e_calc_log_name



# def zero_one_scaler(data,log_name):
#     ''' Normalization'''
#     result = data.copy()
#     # result = np.zeros(data.shape)
#     for i in range(len(log_name)):
#                 # 严格控制范围
#         it_data = np.array(data.iloc[:,i])
#         for j in range(len(it_data)):
#             if it_data[j] < log_name[i][0]:
#                 it_data[j] = log_name[i][0]
#             if it_data[j]  > log_name[i][1]:
#                 it_data[j] = log_name[i][1]
        
#         numerator = it_data - log_name[i][0]
#         # numerator_1 = data.iloc[:,i]-log_name[i][1]
#         denominator = log_name[i][1]-log_name[i][0]
#         result.iloc[:,i] = numerator / (denominator + 1e-8)
        
#     return  result



#inputY



# min(inputY-1)


# ## 岩相类别编号变换


AB_G = sen.zero_one_scaler(inputX,u_log_name)
print(AB_G.shape)



set(facies_labels)



class_begin = 0  # default = 0, 默认类别标签从0开始
if model_stage == "train" or model_stage == "test":
#     inputY_train = sen.zero_one_scaler(inputY,e_log_name)
#     inputY_calc_train = sen.zero_one_scaler(inputY_calc,e_calc_log_name)
#     AB_Y_G = inputY_train.loc[:,[element_name]]
#     AB_Y_calc_G = inputY_calc_train.loc[:,[reference_name]]

    if min(inputY) == 1:
        print("类别标签从1开始")
        class_begin = 1
        y = tf.keras.utils.to_categorical(inputY-1,num_classes=len(facies_labels))
    elif min(inputY) == 0:
        print("类别标签从0开始")
        y = tf.keras.utils.to_categorical(inputY,num_classes=len(facies_labels))
    else:
        print("检查类别标签")
        exit()

    AB_Y_G = y #+ 1e-8
    AB_Y_calc_G  = y #+ 1e-8
    print("zero_one_scaler is finished!")
    print(AB_Y_G.shape)
    print(AB_Y_calc_G.shape)



facies_labels_use = []
facies_colors_use = []

facies_counts = AB_use[facies_labels_col].value_counts().sort_index()
#use facies labels to index each count

for i in range(len(facies_counts)):
    # 如果类别从1开始
    if(class_begin ==1):
        use_facies_id = facies_counts.index[i] - 1
    elif(class_begin ==0):
        use_facies_id = facies_counts.index[i]
    print(use_facies_id)
    print(facies_labels[use_facies_id])
    facies_labels_use.append(facies_labels[use_facies_id])
    facies_colors_use.append(facies_colors[use_facies_id])

facies_counts.index = facies_labels_use

facies_counts.plot(kind='bar',color=facies_colors_use, 
                   title='Distribution of Training Data by Facies')
print(facies_counts)



facies_colors_use



facies_labels_use



if model_stage == "train":
    print(np.sum(facies_counts))
    print(np.unique(AB_use[facies_labels_col]))


# ### 绘制归一化后的曲线


facies_labels_col



def make_facies_log_plot_2(logs,sample_index, facies_colors):
    #make sure logs are sorted by depth
    # logs = logs.sort_values(by = DEPTH_col_name)
    cmap_facies = colors.ListedColormap(
            facies_colors[0:len(facies_colors)], 'indexed')
    
    # sample_index = np.arange(len(AB_G))
    # ztop=logs[DEPTH_col_name].min(); zbot=logs[DEPTH_col_name].max()
    ztop = sample_index[0]; zbot = sample_index[-1]
    
    if model_stage == "train":
        label_col = facies_labels_col
    else:
        label_col = "Prediction"
    # print(logs[label_col].values)
    cluster = np.repeat(np.expand_dims(logs[label_col].values,1), 100, 1)
    total_fig_cols = len(input_vectors)+ 1
    f, ax = plt.subplots(nrows=1, ncols=len(input_vectors) + 1, figsize=(total_fig_cols * 2,40))
    for i in range(len(input_vectors)):
        ax[i].plot(logs[input_vectors[i]], sample_index)
    final_line = len(input_vectors)
 
    im = ax[final_line].imshow(cluster, interpolation='none', aspect='auto',
                    cmap=cmap_facies,vmin = 1,vmax = len(facies_colors))
    
    divider = make_axes_locatable(ax[final_line])
    cax = divider.append_axes("right", size="20%", pad=0.05)
    cbar=plt.colorbar(im, cax=cax)
    cbar.set_label((3*' ').join(facies_labels))
    cbar.set_ticks(range(0,1)); cbar.set_ticklabels('')
    
    for i in range(len(ax)-1):
        ax[i].set_ylim(ztop,zbot)
        ax[i].invert_yaxis()
        ax[i].grid()
        ax[i].locator_params(axis='x', nbins=3)
        ax[i].set_xlabel(input_vectors[i])
        ax[i].set_xlim(logs[input_vectors[i]].min(),logs[input_vectors[i]].max())
        
    for i in range(len(ax)-1):
        ax[i].set_yticklabels([]);
        
    if model_stage == "train" :   
        ax[final_line].set_xlabel('Facies')    
        ax[final_line].set_yticklabels([])
        ax[final_line].set_xticklabels([])
    # f.suptitle('Well: %s'%logs.iloc[0]['Well Name'], fontsize=14,y=0.94)
    f.suptitle("岩相分析", fontsize=14,y=0.94)
    if model_stage == "train":
        pred_image_save_path = model_training_img_file_saving_path 
        pred_image_name = model_training_img_name
        plt.savefig(os.path.join(model_training_img_file_saving_path , model_training_img_name +  str(well_name) + '_tendency.png'), dpi=96,  bbox_inches='tight')

    else:
        pred_image_save_path = model_testing_img_file_saving_path 
        pred_image_name = model_testing_image_name
        plt.savefig(os.path.join(pred_image_save_path , pred_image_name + '_PredictionAll.png'), dpi=96,  bbox_inches='tight')



sample_index = np.arange(len(AB_G))
# sample_index



# type(inputY)



if model_stage == "train":
    df_data = pd.concat([AB_G,inputY],axis=1)
    make_facies_log_plot_2(df_data,sample_index,facies_colors) 
 


# ## 输入自变量数据准备


if model_stage == "train":
    AB_Y = np.array(AB_Y_G)
    AB_Y_calc = np.array(AB_Y_calc_G) 
AB_X = np.array(AB_G)



AB_X



# AB_Y[1000]


# ### 训练集验证集划分

# 根据模型需要判定是否需要序列化


if (flag == 1) or (flag == 2):
    print("不需要序列化")
    if model_stage == "train":
        # 训练阶段
        dataX = AB_X
        dataY = AB_Y
        dataY_calc = AB_Y_calc
        sss = model_selection.ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
        for train_index, test_index in sss.split(dataX):
        # print("TRAIN:", train_index, "TEST:", test_index)
            train_X, test_X = dataX[train_index], dataX[test_index]
            train_Y, test_Y = dataY[train_index], dataY[test_index]
            train_Y_calc, test_Y_calc = dataY_calc[train_index], dataY_calc[test_index]
            train_weight, test_weight = sample_weight[train_index],sample_weight[test_index]
    else:
        # 测试阶段
        testALL_A_X = AB_X
    if use_depth_log == True:
        DEPTH_AddReslution = depth_log
    else:
        DEPTH_AddReslution = None
else:
    print("序列化")
    if model_stage == "train":
#         if (flag == 1) or (flag == 2):
#             print("不需要序列化")
#             dataX = AB_X
#             dataY = AB_Y
#             dataY_calc = AB_Y_calc
#         else:
        dataX, dataY = sen.build_All_Train_dataset(AB_X, AB_Y, seq_length)
        dataY_calc = sen.build_All_Y_dataset( AB_Y_calc, seq_length)
        weight_matrix = sen.build_All_Y_dataset(sample_weight,seq_length)
            # 使用model_selection.ShuffleSplit抽取样本
#         sss = model_selection.ShuffleSplit(n_splits = 10, test_size=0.2, random_state=0)
#         for train_index, test_index in sss.split(dataX):
#         # print("TRAIN:", train_index, "TEST:", test_index)
#             trainX, testX = dataX[train_index], dataX[test_index]
#             trainY, testY = dataY[train_index], dataY[test_index]
#             train_Y_calc, test_Y_calc = dataY_calc[train_index], dataY_calc[test_index]
#             train_weight, test_weight = weight_matrix[train_index],weight_matrix[test_index]
        trainX, testX = dataX, dataX
        trainY, testY = dataY, dataY
        train_Y_calc, test_Y_calc = dataY_calc, dataY_calc
        train_weight, test_weight = weight_matrix,weight_matrix
            
    else:
        # 测试阶段
        testALL_A_X = sen.build_All_A_dataset(AB_X, seq_length)
    if use_depth_log == True:
        DEPTH_AddReslution = sen.build_addReslution_DEPTH(depth_log, seq_length)
    else:
        DEPTH_AddReslution = None



# len(depth_log)


# ### 确认输入维度


if (flag == 1) or (flag == 2):
    if model_stage == "train":
        print("input_vectors.length:",len(input_vectors))
        print("train_X.shape:", train_X.shape,"test_X.shape:", test_X.shape)
        print("train_Y.shape:", train_Y.shape,"test_Y.shape:", test_Y.shape)
        print("train_Y_calc.shape:", train_Y_calc.shape,"test_Y_calc.shape:", test_Y_calc.shape)
        print("train_weight.shape:", train_weight.shape,"test_weight.shape:", test_weight.shape)
    else:
        print("testALL_A_X.shape:", testALL_A_X.shape,"\n","input_vectors.length:",len(input_vectors))
else:
    if model_stage == "train":
        print("input_vectors.length:",len(input_vectors))
        print("trainX.shape:", trainX.shape,"testX.shape:", testX.shape) 
        print("trainY.shape:", trainY.shape,"testY.shape:", testY.shape)
        print("train_Y_calc.shape:", train_Y_calc.shape,"test_Y_calc.shape:", test_Y_calc.shape)
        print("train_weight.shape:", train_weight.shape,"test_weight.shape:", test_weight.shape)
    else:
        print("testALL_A_X.shape:", testALL_A_X.shape,"\n","input_vectors.length:",len(input_vectors))



# train_weight


# # 网络实例化

# ## 构建网络或载入模型


def model_type_select(model_type):
    if model_type == 'DNN':
        return smsc.dnn_model(model_para)
    elif model_type == 'DNN_2':
        return smsc.dnn_model_2(model_para)
    elif model_type == 'RBF':
        n_layers = 3
        return smsc.rbf_model(train_X,model_para)
    elif model_type == 'LSTM':
        return smsc.lstm_cell_model(model_para)
    elif model_type == 'GRU':
        return smsc.gru_cell_model(model_para)
    elif model_type == 'GRU2':
        return smsc.gru_block_cell_2(model_para)
    # elif model_type == 'NAS':
    #     return nas_cell()
    elif model_type == 'BLSTM':
        return smsc.bi_lstm_cell_model(model_para)
    elif model_type == 'BGRU':
        return smsc.bi_gru_cell_model(model_para)
    elif model_type == 'BiGRU-Atten':
        return smsc.bigru_atten_model(model_para)
    elif model_type == 'BiGRU-Atten2':
        return smsc.bigru_atten_model_2(model_para)
    elif model_type == 'BiGRU-self-Atten':
        return smsc.bigru_self_atten_model(model_para)
    elif model_type == 'WaveNet':
        return smsc.wavenet_model(model_para)
    elif model_type == 'MyWaveNet':
        return smsc.wavenet_model2(model_para)
    elif model_type == 'Double_Expert_Net':
        return smsc.double_experts_Net_model(model_para)   
    elif model_type == 'MyUNet':
        return smsc.my_unet(model_para)
    elif model_type == 'Bilstm_Capsule':
        return smsc.bilstm_capsule_model(model_para)
    elif model_type == 'BLSTM-Atten':
        return smsc.bilstm_atten_model(model_para)
    elif model_type == 'BiLSTM-Atten2':
        return smsc.bilstm_atten_model_2(model_para)
    elif model_type == 'BiLSTM-Atten3':
        return smsc.bilstm_atten_model_3(model_para)
    elif model_type == 'BiLSTM-Atten4':
        return smsc.bilstm_atten_model_4(model_para)
    elif model_type == 'BiLSTM-Atten5':
        return smsc.bilstm_atten_model_5(model_para)
    elif model_type == 'BiLSTM-Atten6':
        return smsc.bilstm_atten_model_6(model_para)
    elif model_type == 'CNN_Atten':
        return smsc.wavenet_atten_model(model_para)
    elif model_type == 'Multihead_Self_Atten':
        return smsc.multihead_model(model_para)
    elif model_type ==  'Bigru_Multihead_Self_Atten':
        return smsc.bigru_multihead_atten_model(model_para)
    elif model_type ==  'Bigru_Multihead_Self_Atten_RBF':
        return smsc.bigru_multihead_atten_rbf_model(model_para)
    elif model_type ==  'Bigru_Multihead_Self_Atten_DNN':
        return smsc.bigru_multihead_atten_dnn_model(model_para)
    else:
        return smsc.bi_lstm_cell_model(model_para)



if model_stage == "test":
    print(pred_model_file)




# ## 模型可视化


from distutils.util import strtobool
TF_KERAS = strtobool(os.environ.get('TF_KERAS', '0'))
print(TF_KERAS)



# TF_KERAS
if model_stage == "test":
    print(pred_model_json)



model = tf.keras.Model()
if model_stage == "train":
    model =  model_type_select(model_type)
#     model = wavenet_atten_model(model_para)
    # model = sms.bi_lstm_cell_model(model_para)
    # model = bigru_self_atten_model(model_para)
else:
    with open(pred_model_json, "r") as json_file_1:
        json_config_1 = json_file_1.read()
        # 此处加载模型无需判断 
        model = tf.keras.models.model_from_json(json_config_1,custom_objects={'GlorotUniform': tf.keras.initializers.GlorotUniform(),
             'Zeros': tf.keras.initializers.Zeros(),
             'RBFLayer': smsc.RBFLayer,
             'AttentionLayer': smsc.AttentionLayer,
             'Self_Attention_layer': smsc.Self_Attention_layer,
             'MultiHead': smsc.MultiHead,
             'MultiHeadAttention': smsc.MultiHeadAttention})
        model.load_weights(pred_model_file)
print(model.summary())



# 定义结果打印函数
class PrintDot(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs):
        if epoch % 10 == 0:
            print('已经训练完'+ str(epoch) +'Epoch')
            print('.', end='')



#if  pg.find_graphviz() is not None:
if model_stage == "train":
    model_image_path = model_path
else:
    model_image_path = csv_file_saving_path
if plot_modelnet == True:
    tf.keras.utils.plot_model(model,to_file= model_image_path + element_name + '_' + model_type + '_net.png',
               show_shapes=True,
               show_layer_names=True,
               rankdir='TB',
               expand_nested=False,
               dpi=96)
    print(model_image_path + element_name + '_' + model_type + '_net.png')






# ## 日志保存内容设定


current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

if not os.path.exists(log_path):
    # 针对第一次训练
    os.mkdir(log_path)
my_log_dir = os.path.join(log_path,model_type.lower())
print("my_log_dir:",my_log_dir)
if not os.path.exists(my_log_dir):
    os.mkdir(my_log_dir)

str_name = model_name.split(".h5")[0]
# log_name = str_name + "-{}".format(int(time.time()))
log_name = "{}".format(int(time.time()))
curve_reconstract_logs_child_dir = os.path.join(my_log_dir,log_name)
print("curve_reconstract_logs_child_dir:",curve_reconstract_logs_child_dir)
if not os.path.exists(curve_reconstract_logs_child_dir):
    os.mkdir(curve_reconstract_logs_child_dir)






# # 模型训练与测试

# ## 模型训练与验证

# fit函数解析 
# ```
# fit( x, y, batch_size=32, epochs=10, verbose=1, callbacks=None,
# validation_split=0.0, validation_data=None, shuffle=True, 
# class_weight=None, sample_weight=None, initial_epoch=0)
# ```

# * x：输入数据。如果模型只有一个输入，那么x的类型是numpy array，如果模型有多个输入，那么x的类型应当为list，list的元素是对应于各个输入的numpy array  
# * y：标签，numpy array  
# * batch_size：整数，指定进行梯度下降时每个batch包含的样本数。训练时一个batch的样本会被计算一次梯度下降，使目标函数优化一步。  
# * epochs：整数，训练终止时的epoch值，训练将在达到该epoch值时停止，当没有设置initial_epoch时，它就是训练的总轮数，否则训练的总轮数为epochs - inital_epoch  
# * verbose：日志显示，0为不在标准输出流输出日志信息，1为输出进度条记录，2为每个epoch输出一行记录  
# * callbacks：list，其中的元素是keras.callbacks.Callback的对象。这个list中的回调函数将会在训练过程中的适当时机被调用，参考回调函数  
# * validation_split：0~1之间的浮点数，用来指定训练集的一定比例数据作为验证集。验证集将不参与训练，并在每个epoch结束后测试的模型的指标，如损失函数、精确度等。注意，validation_split的划分在shuffle之前，因此如果你的数据本身是有序的，需要先手工打乱再指定validation_split，否则可能会出现验证集样本不均匀。  
# * validation_data：形式为（X，y）的tuple，是指定的验证集。此参数将覆盖validation_spilt。
# * shuffle：布尔值或字符串，一般为布尔值，表示是否在训练过程中随机打乱输入样本的顺序。若为字符串“batch”，则是用来处理HDF5数据的特殊情况，它将在batch内部将数据打乱。  
# * class_weight：字典，将不同的类别映射为不同的权值，该参数用来在训练过程中调整损失函数（只能用于训练）  
# * sample_weight：权值的numpy array，用于在训练时调整损失函数（仅用于训练）。可以传递一个1D的与样本等长的向量用于对样本进行1对1的加权，或者在面对时序数据时，传递一个的形式为（samples，sequence_length）的矩阵来为每个时间步上的样本赋不同的权。这种情况下请确定在编译模型时添加了sample_weight_mode=’temporal’。Timestep-wise sample weighting (use of sample_weight_mode="temporal") is restricted to outputs that are at least 3D, i.e. that have a time dimension.
# * initial_epoch: 从该参数指定的epoch开始训练，在继续之前的训练时有用。  


def scheduler(epoch):
# 每隔10个epoch，学习率减小为原来的1/10
    if epoch % 10 == 0 and epoch != 0:
        lr = Kbackend.get_value(model.optimizer.lr)
        Kbackend.set_value(model.optimizer.lr, lr * 0.1)
        print("lr changed to {}".format(lr * 0.1))
    return Kbackend.get_value(model.optimizer.lr)

# keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, verbose=0, mode='auto', epsilon=0.0001, cooldown=0, min_lr=0)
if learning_rate_deacy_policy == 1:
    reduce_lr = [
        LearningRateScheduler(scheduler),
         # tf.keras.callbacks.ModelCheckpoint(model_file,
          #                      save_best_only=True)
    ]
elif learning_rate_deacy_policy == 2:
    reduce_lr = [
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience = 10,verbose=2, mode='auto'),
        # PrintDot(),  5
    ]
        
else:
     # reduce_lr =  PrintDot()
    reduce_lr = [
        PrintDot(),
    ]
    
# 原文链接：https://blog.csdn.net/zzc15806/article/details/79711114



my_callbacks = [
    tf.keras.callbacks.TensorBoard(log_dir =  curve_reconstract_logs_child_dir),
    tf.keras.callbacks.ModelCheckpoint(model_file,save_best_only=True),
    tf.keras.callbacks.EarlyStopping(patience=8,mode='auto', min_delta=1e-9),
]


# tensorboard = TensorBoard(log_dir = my_log_dir + '\{}'.format(log_name))



# My_X.shape,My_Y.shape



if model_stage == "train":
    if (flag == 1) or (flag == 2):
        print("flag:",flag)
        My_X = train_X
        My_Y = train_Y
        My_Test_X = test_X
        My_Test_Y = test_Y
        My_Weight =  train_weight #
        My_class_Weight =  class_weight# 'auto' # {"0":}
    else:
        print("flag:",flag)
        My_X = trainX # trainX, dataX
        My_Y = trainY # trainY， dataY 
        My_Test_X = testX
        My_Test_Y = testY
        My_Weight = train_weight  # train_weight ,weight_matrix
        My_class_Weight = class_weight # 'auto'
        # history = model.fit(train_X, train_Y, batch_size=BATCH_SIZE,epochs=EPOCHS,
        #                validation_split = 0.1, verbose=1,callbacks=[PrintDot()])
    #     dataset = tf.data.Dataset.from_tensor_slices((train_X, train_Y))
    #     train_dataset  = dataset.shuffle(len(train_Y)).batch(1)
    #     t_dataset = tf.data.Dataset.from_tensor_slices((test_X,test_Y))
    #     val_dataset  = dataset.shuffle(len(test_Y)).batch(1)
    #     history = model.fit(train_dataset, epochs=EPOCHS,
    #                      validation_data = val_dataset, verbose=1,callbacks=[PrintDot()])

    if batch_size_strategy == True:
        if save_logs == False:
            history = model.fit(My_X, My_Y, batch_size = BATCH_SIZE,epochs=EPOCHS,validation_data = (My_Test_X,My_Test_Y),class_weight = My_class_Weight, verbose=1,callbacks=reduce_lr)
        else:
            history = model.fit(My_X, My_Y, batch_size = BATCH_SIZE,epochs=EPOCHS,validation_data = (My_Test_X,My_Test_Y),class_weight = My_class_Weight, verbose=1,callbacks=my_callbacks)
    else:
        if save_logs == False:
            history = model.fit(My_X, My_Y, epochs = EPOCHS,validation_data = (My_Test_X,My_Test_Y), class_weight = My_class_Weight, verbose=1,callbacks = reduce_lr)
        else:
            history = model.fit(My_X, My_Y, epochs = EPOCHS,validation_data = (My_Test_X,My_Test_Y), class_weight = My_class_Weight, verbose=1,callbacks = my_callbacks)
else:
    print("model_stage:", model_stage)


# ## 模型测试

# model.evaluate输入数据(data)和金标准(label),然后将预测结果与金标准相比较,得到两者误差并输出.  
# model.predict输入数据(data),输出预测结果  
# * 是否需要真实标签(金标准)  
# model.evaluate需要,因为需要比较预测结果与真实标签的误差  
# model.predict不需要,只是单纯输出预测结果,全程不需要金标准的参与.  


def plot_history(history,model_training_img_file_saving_path,model_training_img_name):
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch

    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.plot(hist['epoch'], hist['loss'],label='Train Error')
    plt.plot(hist['epoch'], hist['val_loss'],label = 'Val Error')
    # plt.ylim([0,5])
    plt.legend()
    plt.savefig(os.path.join(model_training_img_file_saving_path , model_training_img_name + '_MAE.png'), dpi=96,  bbox_inches='tight')

    plt.figure()
    plt.ylim(0,1.05)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.plot(hist['epoch'], hist['accuracy'],label='Train accuracy')
    plt.plot(hist['epoch'], hist['val_accuracy'],label = 'Val accuracy')
    plt.grid
    # plt.ylim([0,20])
    plt.legend()
    
    plt.savefig(os.path.join(model_training_img_file_saving_path , model_training_img_name +"_" + element_name + '_'+ str(n_layers) + "_layers_" +  "_lr_" + str(
        learning_rate) + "h_dim" + str(hidden_dim) + "_epoch_" + str(EPOCHS) + '_loss.png'), dpi=96,  bbox_inches='tight')
    # plt.show()
    



if model_stage == "train":
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch
    print(hist.tail())
    plot_history(history,model_training_img_file_saving_path,model_training_img_name)



if model_stage == "train":
    if flag != 4:
        train_loss_csv_name = train_well_name + "_" + model_type.lower() + "_" + str(len(input_vectors)) + '-facies_'+ str(n_layers) + "_layers_" +  "_lr_" + str(
        learning_rate) + "h_dim" + str(hidden_dim) + "_epoch_" + str(EPOCHS) + "_trainloss.csv"
    else:
        train_loss_csv_name = train_well_name + "_" + model_type.lower() + "_" + str(len(input_vectors)) + '-facies_'+ str(n_layers) + "_layers_" +  "_lr_" + str(
        learning_rate) + "h_dim" + str(hidden_dim) + "_epoch_" + str(EPOCHS) + "_alltrainloss.csv"
    hist.to_csv(csv_file_saving_path + train_loss_csv_name,mode='w',float_format='%.6f',index=None,header=True)



hist



if model_stage == "train":
    train_loss_csv = model_name.split('.h5')[0] + '_'+ sen.tid_maker() +'.csv'
    hist.to_csv(model_save_path + train_loss_csv,mode='w',float_format='%.4f',index=None,header=True)



print(model_name.split('.h5')[0])


# ## 混淆矩阵


from sklearn.metrics import confusion_matrix
from classification_utilities import display_cm, display_adj_cm



if model_stage == "train":
        predictions = model.predict(My_Test_X)
#         np.testing.assert_allclose(predictions, testY, atol=1e-6)
        loss, accuracy = model.evaluate(My_Test_X, My_Test_Y, verbose=2)
        sample_index = np.arange(len(My_Test_Y))
        max_test = np.argmax(My_Test_Y, axis=1)
        max_predictions = np.argmax(predictions, axis=1)
# else:
#    A_Y_predict = model.predict(testALL_A_X)



np.argmax([0.00000033, 0.00944981, 0.97227716, 0.00000005, 0.01238397,
        0.00106168])



# 待修改
def count_facies(max_predictions_out,facies_labels):
    facies_use = []
    predictions_dataframe = pd.DataFrame(max_predictions_out,columns=[facies_labels_col])
    facies_counts = predictions_dataframe.value_counts().sort_index()
    #use facies labels to index each count   
    for i in range(len(facies_counts)):
        if(class_begin == 1):
            # 如果类别从1开始
            use_facies_id = facies_counts.index[i] - 1
        elif(class_begin == 0):
            use_facies_id = facies_counts.index[i]
        print(use_facies_id)
        print(facies_labels[use_facies_id])
        facies_use.append(facies_labels[use_facies_id])
    return facies_use



if model_stage == "train":
    conf = confusion_matrix(max_test, max_predictions)
    display_cm(conf, facies_labels_use, hide_zeros=True)
    plt.figure(figsize=(10,10))
    sns.heatmap(conf, xticklabels=facies_labels_use, yticklabels=facies_labels_use, annot=True, fmt="d", annot_kws={"size": 20});
    plt.title("Confusion matrix", fontsize=20)
    plt.ylabel('True label', fontsize=20)
    plt.xlabel('Predicted label', fontsize=20)
    plt.savefig(os.path.join(model_training_img_file_saving_path , model_training_img_name +"_" + element_name + '_'+ str(n_layers) + "_layers_" +  "_lr_" + str(
        learning_rate) + "h_dim" + str(hidden_dim) + "_epoch_" + str(EPOCHS) + '_confusion_matrix.png'), dpi=96,  bbox_inches='tight')
    # plt.show()



def accuracy(conf):
    total_correct = 0.
    nb_classes = conf.shape[0]
    for i in np.arange(0,nb_classes):
        total_correct += conf[i][i]
    acc = total_correct/sum(sum(conf)) + 1e-8
    return acc


def accuracy_adjacent(conf, adjacent_facies):
    nb_classes = conf.shape[0]
    total_correct = 0.
    for i in np.arange(0,nb_classes):
        total_correct += conf[i][i]
        for j in adjacent_facies[i]:
            total_correct += conf[i][j]
    return total_correct / sum(sum(conf)) + 1e-8



def per_class_accuracy(conf):
    per_class_correct = 0.
    nb_classes = conf.shape[0]
    per_acc = []
    for i in np.arange(0,nb_classes):
        acc = 0
        per_class_correct = conf[i][i]
        per_class_sum = 0
        per_class_sum = per_class_sum + sum(conf[i])
        # print(per_class_sum)
        acc = per_class_correct/per_class_sum + 1e-8
#         acc = per_class_correct/sum(sum(conf))
        per_acc.append(acc)
    return per_acc



if model_stage == "train":
    print(conf)







if model_stage == "train":
    print('Facies classification accuracy = %f' % accuracy(conf))
#     print('Adjacent facies classification accuracy = %f' % accuracy_adjacent(conf, adjacent_facies))
    per_class_accuracy_list = per_class_accuracy(conf)
    for i in range(len(per_class_accuracy_list)):
        print(i,' class facies classification accuracy = %f' % per_class_accuracy_list[i])


# ## 绘制ROC


from sklearn.metrics import roc_curve, auc



n_classes = len(facies_labels_use)



# Compute ROC curve and ROC area for each class
if model_stage == "train":
    
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(My_Test_Y[:, i], predictions[:, i])   
        #max_test样例真实标签，max_predictions学习器预测的样例的概率 
        roc_auc[i] = auc(fpr[i], tpr[i])   
        #计算ROC曲线下方的面积，fpr假正例率数组(横坐标)，tpr真正例率数组(纵坐标） 



if model_stage == "train":
    fpr["micro"], tpr["micro"], _ = roc_curve(My_Test_Y.ravel(), predictions.ravel())   #ravel函数将矩阵展开成向量
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])



# https://blog.csdn.net/cymy001/article/details/79613787
# First aggregate all false positive rates
if model_stage == "train":
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))   #np.concatenate将“特征维度相同数组”纵向拼接

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)   #np.zeros_like创建一个和参数all_fpr数组维度相同的全0数组
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])   
        #interp一维线性插值，fpr和tpr是插值结点横纵坐标，all_fpr是已知中间节点横坐标(得到插值曲线后，求其纵坐标)
    #https://docs.scipy.org/doc/numpy/reference/generated/numpy.interp.html#numpy.interp

    # Finally average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])



if model_stage == "train":
    print(roc_auc)



# Plot all ROC curves
# from itertools import cycle
if model_stage == "train":
    plt.rcParams['figure.figsize']=(8,6)
    plt.figure()
    plt.plot(fpr["micro"], tpr["micro"], label='micro-average ROC curve (area = {0:0.4f})' ''.format(roc_auc["micro"]),
             color='deeppink', linestyle=':')

    plt.plot(fpr["macro"], tpr["macro"], label='macro-average ROC curve (area = {0:0.4f})' ''.format(roc_auc["macro"]),
             color='navy', linestyle=':')

    # colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])   #python3里的无穷循环器
    line_colors = facies_colors_use
    for i, line_color in zip(range(n_classes), line_colors):
        plt.plot(fpr[i], tpr[i], color=line_color, label='ROC curve of class {0} (area = {1:0.4f})' ''.format(i, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Some extension of Receiver operating characteristic to multi-class')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(model_training_img_file_saving_path , model_training_img_name +"_" + element_name + '_'+ str(n_layers) + "_layers_" +  "_lr_" + str(
        learning_rate) + "h_dim" + str(hidden_dim) + "_epoch_" + str(EPOCHS) + '_ROC.png'), dpi=300,  bbox_inches='tight')
    # plt.show()



import codecs,json







if model_stage == "train":
    if flag != 4:
        train_fpr_json_name = train_well_name + "_" + model_type.lower() + "_" + str(len(input_vectors)) + '-facies_'+ str(n_layers) + "_layers_" +  "_lr_" + str(
        learning_rate) + "h_dim" + str(hidden_dim) + "_epoch_" + str(EPOCHS) + "_fpr.json"
        train_tpr_json_name = train_well_name + "_" + model_type.lower() + "_" + str(len(input_vectors)) + '-facies_'+ str(n_layers) + "_layers_" +  "_lr_" + str(
        learning_rate) + "h_dim" + str(hidden_dim) + "_epoch_" + str(EPOCHS) + "_tpr.json"
    else:
        train_fpr_json_name = train_well_name + "_" + model_type.lower() + "_" + str(len(input_vectors)) + '-facies_'+ str(n_layers) + "_layers_" +  "_lr_" + str(
        learning_rate) + "h_dim" + str(hidden_dim) + "_epoch_" + str(EPOCHS) + "_allfpr.json"
        train_tpr_json_name = train_well_name + "_" + model_type.lower() + "_" + str(len(input_vectors)) + '-facies_'+ str(n_layers) + "_layers_" +  "_lr_" + str(
        learning_rate) + "h_dim" + str(hidden_dim) + "_epoch_" + str(EPOCHS) + "_alltpr.json"
     
    fileObject = open(csv_file_saving_path + train_fpr_json_name, 'w', encoding='utf-8')  
    for i, line_color in zip(range(n_classes), line_colors): 
        fileObject.write(str(fpr[i]))  
        fileObject.write('\n')  
    fileObject.close()
    
    fileObject = open(csv_file_saving_path + train_tpr_json_name, 'w', encoding='utf-8')  
    for i, line_color in zip(range(n_classes), line_colors):
        fileObject.write(str(tpr[i]))  
        fileObject.write('\n')  
    fileObject.close() 
    
    


# ## 和真实岩相对比


def make_facies_log_plot_3(logs,sample_index, facies_colors):
    #make sure logs are sorted by depth
    # logs = logs.sort_values(by = DEPTH_col_name)
    cmap_facies = matplotlib.colors.ListedColormap(
            facies_colors[0:len(facies_colors)], 'indexed')
    
    # ztop=logs[DEPTH_col_name].min(); zbot=logs[DEPTH_col_name].max()
    ztop =0; zbot = len(logs)
    
    cluster_GT = np.repeat(np.expand_dims(logs["Facies"].values,1), 100, 1)
    cluster_Pred = np.repeat(np.expand_dims(logs["Prediction"].values,1), 100, 1)
    
    # total_fig_cols = len(input_vectors)+ 1
    total_fig_cols = 2
    # f, ax = plt.subplots(nrows=1, ncols=len(input_vectors) + 2, figsize=(total_fig_cols * 2,15))
    f, ax = plt.subplots(nrows=1, ncols = 2, figsize=(total_fig_cols * 2,30))
#     for i in range(len(input_vectors)):
#         ax[i].plot(logs[input_vectors[i]], sample_index)
    # final_line = len(input_vectors)
    final_line = 0
    if model_stage == "train":
        im1 = ax[0].imshow(cluster_GT, interpolation='none', aspect='auto',
                        cmap=cmap_facies,vmin = 0,vmax = len(facies_colors))
        im2 = ax[1].imshow(cluster_Pred, interpolation='none', aspect='auto',
                    cmap=cmap_facies,vmin=0,vmax=len(facies_colors))
    
        # divider = make_axes_locatable(ax[final_line])
        divider = make_axes_locatable(ax[1])
        cax = divider.append_axes("right", size="20%", pad=0.05)
        cbar=plt.colorbar(im2, cax=cax)
        cbar.set_label((25*' ').join(facies_labels))
        cbar.set_ticks(range(0,1)); cbar.set_ticklabels('')
    
#     for i in range(len(ax)-1):
#         ax[i].set_ylim(ztop,zbot)
#         ax[i].invert_yaxis()
#         ax[i].grid()
#         ax[i].locator_params(axis='x', nbins=3)
#         ax[i].set_xlabel(input_vectors[i])
#         ax[i].set_xlim(logs[input_vectors[i]].min(),logs[input_vectors[i]].max())
        
#     for i in range(len(ax)-1):
#         ax[i].set_yticklabels([]);
        
    if model_stage == "train" :   
        ax[final_line].set_xlabel('Facies')
        ax[final_line+1].set_xlabel('Prediction')
#         ax[final_line].set_yticklabels([])
#         ax[final_line].set_xticklabels([])
#         ax[final_line+1].set_yticklabels([])
#         ax[final_line+1].set_xticklabels([])
#     # f.suptitle('Well: %s'%logs.iloc[0]['Well Name'], fontsize=14,y=0.94)
    f.suptitle("岩相分析", fontsize=14,y=0.94)



if model_stage == "train":
    Val_GT = pd.DataFrame(max_test, columns=["Facies"], index= np.arange(len(max_test)))
    Val_Pred = pd.DataFrame(max_predictions, columns=["Prediction"],index= np.arange(len(max_test)))







if model_stage == "train":
    pred_data = pd.concat([Val_GT,Val_Pred],axis=1)
    print(pred_data)



if model_stage == "train":
    sample_index = np.arange(len(max_test))
    make_facies_log_plot_3(pred_data,sample_index,facies_colors_use)
    
    plt.savefig(os.path.join(model_training_img_file_saving_path , model_training_img_name +"_" + element_name + '_'+ str(n_layers) + "_layers_" +  "_lr_" + str(
        learning_rate) + "h_dim" + str(hidden_dim) + "_epoch_" + str(EPOCHS)+ '_predError_Distribution.png'), dpi=96,  bbox_inches='tight')
    # plt.show()
    # print('Test RMSE: %.3f' % rmse)


# # 模型保存


if model_stage == "train":
    if paly_music == True:
        mixer.init()
        music_path = "music/"
        file = "MISIA - 星のように.mp3"
        music_file = os.path.join(music_path,file)
        mixer.music.load(music_file)
        mixer.music.play()
        time.sleep(10)
        mixer.music.stop()
        # exit()



if model_stage == "train":
    json_config = model.to_json()
    with open(model_json, 'w') as json_file:
        json_file.write(json_config)

    model.save_weights(model_file)
    print("Model Save is Finished!")
else:
    print("Testing...")


# # 对预测结果后处理

# ## 将预测的概率矩阵转化为实际的标签


if model_stage == "train":
    # A_Y_predict = predictions
    A_Y_predict = model.predict(dataX)
else:
    A_Y_predict = model.predict(testALL_A_X)



A_Y_predict_class = np.argmax(A_Y_predict, axis=1)



A_Y_predict



class_begin



if class_begin == 1:
    testALL_Y_predict_final = A_Y_predict_class + 1
else:
    testALL_Y_predict_final = A_Y_predict_class



if model_stage == "train":
    print("testALL_Y_predict_final.shape:",testALL_Y_predict_final.shape)


# ## 预测曲线可视化


if model_stage == "test":
    if use_depth_log:
        if flag != 3:
            fianl_DEPTH = A_read.loc[:,["DEPTH"]]
        else:
            fianl_DEPTH = A_read.loc[:,["DEPTH"]][seq_length:].reset_index(drop=True)



if model_stage == "test":
    fianl_DEPTH



pred_facies = pd.DataFrame(testALL_Y_predict_final,columns=["Prediction"])
pred_facies



inputX



if model_stage == "train":
    if (flag == 1) or (flag == 2):
        df_pred_data = pd.concat([inputX,pred_facies,inputY],axis=1) 
    else:
        input_AB_X = inputX[seq_length:].reset_index(drop = True)
        input_AB_Y = inputY[seq_length:].reset_index(drop = True)
        df_pred_data = pd.concat([input_AB_X,pred_facies,input_AB_Y],axis=1)
else:
    if (flag == 1) or (flag == 2):
        df_pred_data = pd.concat([fianl_DEPTH,inputX,pred_facies],axis=1) 
    else:
        input_AB_X = inputX[seq_length:].reset_index(drop = True)
        df_pred_data = pd.concat([fianl_DEPTH,input_AB_X,pred_facies],axis=1)



df_pred_data


# # 训练预测阶段的验证操作到此程序结束

# # 总评价模块


# input_AB_Y



add_flag = 0
if model_stage == "train":
    # 训练阶段
    High_R_ALL = AB_use[facies_labels_col]
    High_R = High_R_ALL.copy()
    print("真实标定值为训练数据标签！！！")
    add_flag = 1
    print(add_flag)
else:
    if use_high_R_data == True:
    # 测试阶段
        High_R_ALL = pd.read_csv(HighRDataPath,engine='python',encoding='GBK')
        High_R = High_R_ALL.loc[:, facies_labels_col] 
        add_flag = 2
        print(add_flag)
    else:
        print("无真实标定值！！！")
        add_flag = 3
        print(add_flag)



# max(High_R_ALL)



facies_colors = ['#632423', '#0070C0','#00B0F0','#75DAFF','#00B050','#FFC000', '#FFFF00']

facies_labels = ['高有机质层状页岩相', '高有机质纹层状页岩相','中有机质纹层状页岩相','低有机质纹层状页岩相', '中低有机质块状白云岩相', '低有机质块状介壳灰岩相', '低有机质块状粉砂岩相']

# 不考虑有机质
# facies_colors = ['#00B0F0','#75DAFF','#00B050','#FFC000', '#FFFF00']

# facies_labels = ['层状粘土质页岩', '纹层状粘土质页岩','层状长英质页岩','纹层状长英质页岩','介壳灰岩']

# facies_colors = ['#00B0F0','#75DAFF','#00B050','#FFC000', '#FFFF00','#007F00']

# facies_labels = ['泥质粉砂岩', '层状粘土质页岩','纹层状粘土质页岩','层状长英质页岩','纹层状长英质页岩','灰岩、云岩']



# if use_high_R_data == True:
#     true_facies_counts = High_R_ALL[facies_labels_col].value_counts().sort_index()
#     #use facies labels to index each count 
    
#     true_facies_labels_id = np.unique(High_R_ALL[facies_labels_col])
#     true_facies_labels = []
#     true_facies_colors = []
#     for i in true_facies_labels_id:
#         true_facies_labels.append(facies_labels_use[i])
#         true_facies_colors.append(facies_colors_use[i])
#     true_facies_counts.index = true_facies_labels
#     true_facies_counts.plot(kind='bar',color = true_facies_colors, 
#                        title='Distribution of Training Data by Facies')
#     print(true_facies_counts)
facies_labels


# ## 准确性评估


# facies_colors = ['#632423','#007F00', '#999999','#339966','#99CC00','#00FF00','#7F7F7F','#FFCC99','#FFCC00','#993366','#FF9900', '#FF6600','#00CCFF']

# facies_labels = ['SS', 'CSiS', 'FSiS', 'SiSh', 'MS','WS', 'D','PS', 'BS']
# facies_labels = ['高有机质层状页岩相', '高有机质纹层状页岩相','中有机质纹层状页岩相','低有机质纹层状页岩相',
#                 '中低有机质块状白云岩相', '低有机质块状介壳灰岩相', '低有机质块状粉砂岩相']
# 13类
# facies_labels = ['其他','中有机质层状粘土质页岩','低有机质纹层状粘土质页岩','高有机质层状粘土质页岩','中有机质纹层状粘土质页岩','高有机质纹层状粘土质页岩','低有机质层状粘土质页岩', '低有机质纹层状长英质页岩','中有机质层状长英质页岩','高有机质层状长英质页岩','中有机质纹层状长英质页岩','高有机质纹层状长英质页岩', '介壳灰岩']



if (flag==1 or flag ==2) and (add_flag == 1):
        High_R_Label = High_R.copy()



if add_flag ==1 or add_flag ==2:
    if (flag == 0) or (flag == 1):
        High_R_Label = High_R.copy()
    else:
        High_R_Label = High_R[seq_length:].reset_index(drop=True)



if (add_flag == 1) or (add_flag ==2):
    if min(High_R_Label) == 1:
        print("类别标签从1开始")
        all_y = tf.keras.utils.to_categorical(High_R_Label-1,num_classes=len(facies_labels))
    elif min(High_R_Label) == 0:
        print("类别标签从0开始")
        all_y = tf.keras.utils.to_categorical(High_R_Label,num_classes=len(facies_labels))
    else:
        print("检查类别标签")
        exit()


# ## 绘制曲线岩相图


if model_stage == "train":
    use_depth_log = False



def make_facies_log_plot_4(logs,sample_index, facies_colors):
    #make sure logs are sorted by depth
    # logs = logs.sort_values(by = DEPTH_col_name)
    cmap_facies = colors.ListedColormap(
            facies_colors[0:len(facies_colors)], 'indexed')
    
    # sample_index = np.arange(len(AB_G))
    if use_depth_log == True:
        ztop = logs[DEPTH_col_name].min(); zbot = logs[DEPTH_col_name].max()
        Depth_col = logs[DEPTH_col_name]
        print(ztop,zbot)
    else:
        ztop = 0; zbot = len(logs)
        Depth_col = sample_index
    
    cluster1 = np.repeat(np.expand_dims(logs["Prediction"].values,1), 100, 1)
    cluster2 = np.repeat(np.expand_dims(logs[facies_labels_col].values,1), 100, 1)
    total_fig_cols = len(input_vectors)+ 2
    f, ax = plt.subplots(nrows=1, ncols = total_fig_cols, figsize=(total_fig_cols * 2,15))
    for i in range(len(input_vectors)):
        ax[i].plot(logs[input_vectors[i]], Depth_col)
    final_line = len(input_vectors)
    if (add_flag == 1) or (add_flag ==2):
        im1 = ax[final_line].imshow(cluster1, interpolation='none', aspect='auto',
                        cmap=cmap_facies,vmin = 0,vmax = len(facies_colors))
        im2 = ax[final_line+1].imshow(cluster2, interpolation='none', aspect='auto',
                        cmap=cmap_facies,vmin = 0,vmax = len(facies_colors))
    
        divider = make_axes_locatable(ax[final_line+1])
        cax = divider.append_axes("right", size="20%", pad=0.08)
        cbar=plt.colorbar(im2, cax=cax)
        # 下面一行控制图例文字间距
        cbar.set_label((3*' ').join(facies_labels))
        cbar.set_ticks(range(0,1))
        cbar.set_ticklabels(2*'')
    
    for i in range(len(ax)-2):
        ax[i].invert_yaxis()
        ax[i].grid()
        ax[i].locator_params(axis='x', nbins=3)
        ax[i].set_xlabel(input_vectors[i])
        ax[i].set_xlim(logs[input_vectors[i]].min() - 0.01,logs[input_vectors[i]].max() * 1.01)
        ax[i].set_ylim(math.floor(ztop), math.ceil(zbot))
        
    for i in range(len(ax)-2):
        # ax[i].set_ylim(ztop,zbot)
        ax[i].set_yticklabels([])
    ax[0].set_yticks([])
    
        
    if (add_flag == 1) or (add_flag ==2):
        ax[final_line].set_xlabel('Prediction')
        ax[final_line + 1].set_xlabel('Real Facies')
        ax[final_line].set_yticklabels([])
        ax[final_line].set_xticklabels([])
        ax[final_line + 1].set_yticklabels([])
        ax[final_line + 1].set_xticklabels([])
    # f.suptitle('Well: %s'%logs.iloc[0]['Well Name'], fontsize=14,y=0.94)
    f.suptitle( str(well_name) + "_全部样本预测结果", fontsize=14,y=0.94)
    if model_stage == "train":
        pred_image_save_path = model_training_img_file_saving_path 
        pred_image_name = model_training_img_name
    else:
        pred_image_save_path = model_testing_img_file_saving_path 
        pred_image_name = model_testing_image_name
    print(pred_image_save_path)
    plt.savefig(os.path.join(pred_image_save_path, pred_image_name  + '_PredictionAll.png'), dpi=300,  bbox_inches='tight')



# ztop = df_pred_data[DEPTH_col_name].min(); zbot = df_pred_data[DEPTH_col_name].max()
# math.floor(ztop), math.ceil(zbot)
len(pred_facies),len(testALL_Y_predict_final),testALL_Y_predict_final.shape,pred_facies.shape



samples_index = np.arange(len(testALL_Y_predict_final))
if model_stage == "train":
    make_facies_log_plot_4(df_pred_data,samples_index,facies_colors_use)
else:
    if add_flag ==2:
        reference_facies = pd.DataFrame(High_R_Label, columns=[facies_labels_col])
        df_pred_data = pd.concat([df_pred_data,reference_facies],axis=1)
        make_facies_log_plot_4(df_pred_data,samples_index,facies_colors_use)
    else:
        make_facies_log_plot_2(df_pred_data,samples_index,facies_colors_use)



# make_facies_log_plot_4(df_pred_data,samples_index,facies_colors_use)


# ## 绘制混淆矩阵

# 绘制混淆矩阵，预测结果与真实标定对比


# if class_begin == 1:
#     testALL_Y_predict_final = testALL_Y_predict_final - 1
# else:
#     testALL_Y_predict_final = testALL_Y_predict_final



# len(High_R_Label),len(testALL_Y_predict_final),class_begin



if (add_flag == 1):
    cv_conf = confusion_matrix(High_R_Label, testALL_Y_predict_final)
    display_cm(cv_conf, facies_labels_use, hide_zeros=True)
    plt.figure(figsize=(15,15))
    sns.heatmap(cv_conf, xticklabels=facies_labels_use, yticklabels=facies_labels_use, annot=True, fmt="d", annot_kws={"size": 20});
    plt.title("Confusion matrix", fontsize=20)
    plt.ylabel('True label', fontsize=20)
    plt.xlabel('Predicted label', fontsize=20)
    plt.savefig(model_training_img_file_saving_path + model_training_img_name +"_" + element_name + '_'+ str(n_layers) + "_layers_" +  "_lr_" + str(
            learning_rate) + "h_dim" + str(hidden_dim) + "_epoch_" + str(EPOCHS) + '_all_confusion_matrix.png', dpi=96,  bbox_inches='tight')
    # plt.show()






# ### 是否执行岩相合并

# facies_colors = ['#632423', '#0070C0','#00B0F0','#75DAFF','#00B050','#FFC000', '#FFFF00']
# 
# facies_labels = ['高有机质层状页岩相', '高有机质纹层状页岩相','中有机质纹层状页岩相','低有机质纹层状页岩相',
#                 '中低有机质块状白云岩相', '低有机质块状介壳灰岩相', '低有机质块状粉砂岩相']


# merge_list = {2: 1, 3: 1, 4: 2, 5:3, 6:3}
merge_list = {2: 4, 3: 4, 5: 4}
# 前面的数字为要换的岩相，后面的数字为换的岩相。
# merge_list = {13:13, 8: 12, 11: 12, 9: 10, 7:10,3:6 ,5:6, 2:4, 1:1}



for i in merge_list:
    print(type(i),type(merge_list[i]))
    replace_str = str(i)
    



# np.unique(df_pred_data['Facies'])



execute_merge_label = False   # False  | True
# execute_merge_label = True
# 合并成四类


merge_list = {2: 4, 3: 4, 5: 4}
if (add_flag ==2 and execute_merge_label == True):
#     concanate_facies_labels = ['有机质层状页岩相', '有机质纹层状页岩相', '有机质块状白云岩相',  '有机质块状介壳灰岩粉砂岩相']
#     concanate_facies_labels = ['灰岩、云岩', '纹层状长英质页岩', '层状长英质页岩',  '纹层状粘土质页岩', '层状粘土质页岩','块状粘土质页岩']
    
#     concanate_facies_colors = ['#632423', '#00B0F0','#00B050', '#FFFF00']
#     concanate_facies_colors = ['#632423', '#00B0F0','#00B050', '#FFFF00','#999999']
    
    concanate_facies_colors = ['#00B0F0', '#FFFF00','#007F00']

    concanate_facies_labels = ['砂岩', '页岩','灰岩、云岩']


    for i in merge_list:
        # Series.replace(self, to_replace=None, value=None, inplace=False, limit=None, regex=False, method='pad')
        # 要注意这样的操作并没有改变文档的源数据，要改变源数据需要使用inplace = True
        High_R_Label.replace(to_replace = i, value = merge_list[i],inplace=True)
        # pred_facies.replace(to_replace=r'^replace_str.$', value = merge_list[i], regex=True)
        pred_facies.replace(to_replace= i, value = merge_list[i],inplace=True)
   



# np.unique(pred_facies),np.unique(High_R_Label)



add_flag



def make_facies_log_plot_5(logs,sample_index, facies_colors):
    # 用于绘制合并后的图
    #make sure logs are sorted by depth
    # logs = logs.sort_values(by = DEPTH_col_name)
    cmap_facies = colors.ListedColormap(
            facies_colors[0:len(facies_colors)], 'indexed')
    
    # sample_index = np.arange(len(AB_G))
    if use_depth_log == True:
        ztop = logs[DEPTH_col_name].min(); zbot = logs[DEPTH_col_name].max()
        Depth_col = logs[DEPTH_col_name]
        print(ztop,zbot)
    else:
        ztop = 0; zbot = len(logs)
        Depth_col = sample_index
    
    cluster1 = np.repeat(np.expand_dims(logs["Prediction"].values,1), 100, 1)
    cluster2 = np.repeat(np.expand_dims(logs[facies_labels_col].values,1), 100, 1)
    total_fig_cols = len(input_vectors)+ 2
    f, ax = plt.subplots(nrows=1, ncols=len(input_vectors) + 2, figsize=(total_fig_cols * 2,15))
    for i in range(len(input_vectors)):
        ax[i].plot(logs[input_vectors[i]], Depth_col)
    final_line = len(input_vectors)
    if (add_flag == 1) or (add_flag ==2):
        im1 = ax[final_line].imshow(cluster1, interpolation='none', aspect='auto',
                        cmap=cmap_facies,vmin = 0,vmax = len(facies_colors))
        im2 = ax[final_line+1].imshow(cluster2, interpolation='none', aspect='auto',
                        cmap=cmap_facies,vmin = 0,vmax = len(facies_colors))
    
        divider = make_axes_locatable(ax[final_line+1])
        cax = divider.append_axes("right", size="20%", pad=0.05)
        cbar=plt.colorbar(im2, cax=cax)
        cbar.set_label((25*' ').join(facies_labels))
        cbar.set_ticks(range(0,1)); cbar.set_ticklabels('')
    
    for i in range(len(ax)-2):
        ax[i].invert_yaxis()
        ax[i].grid()
        ax[i].locator_params(axis='x', nbins=3)
        ax[i].set_xlabel(input_vectors[i])
        ax[i].set_xlim(logs[input_vectors[i]].min() - 0.01,logs[input_vectors[i]].max() * 1.02)
        ax[i].set_ylim(math.floor(ztop), math.ceil(zbot))
        
    for i in range(len(ax)-2):
        # ax[i].set_ylim(ztop,zbot)
        ax[i].set_yticklabels([])
    ax[0].set_yticks([])
    
        
    if (add_flag == 1) or (add_flag ==2):
        ax[final_line].set_xlabel('Prediction')
        ax[final_line + 1].set_xlabel('Real Facies')
        ax[final_line].set_yticklabels([])
        ax[final_line].set_xticklabels([])
        ax[final_line + 1].set_yticklabels([])
        ax[final_line + 1].set_xticklabels([])
    # f.suptitle('Well: %s'%logs.iloc[0]['Well Name'], fontsize=14,y=0.94)
    f.suptitle( str(well_name) + "_全部样本预测结果", fontsize=14,y=0.94)
    if model_stage == "train":
        pred_image_save_path = model_training_img_file_saving_path 
        pred_image_name = model_training_img_name
    else:
        pred_image_save_path = model_testing_img_file_saving_path 
        pred_image_name = model_testing_image_name
    print(pred_image_save_path)
    plt.savefig(os.path.join(pred_image_save_path, pred_image_name  + '_Merger_PredictionAll.png'), dpi=96,  bbox_inches='tight')



if (add_flag ==2 and execute_merge_label == True):
    if (flag == 1) or (flag == 2):
        df_pred_data = pd.concat([fianl_DEPTH,inputX,pred_facies],axis=1) 
    else:
        input_AB_X = inputX[seq_length:].reset_index(drop = True)
        df_pred_data = pd.concat([fianl_DEPTH,input_AB_X,pred_facies],axis=1)
        
        reference_facies = pd.DataFrame(High_R_Label, columns=[facies_labels_col])
        df_pred_data = pd.concat([df_pred_data,reference_facies],axis=1)
        
        



if (add_flag ==2 and execute_merge_label == True):
    make_facies_log_plot_5(df_pred_data,samples_index,facies_colors)



if add_flag ==2:
    cv_conf = confusion_matrix(High_R_Label, pred_facies)
    print(cv_conf)



if add_flag ==2:
    new_label_index = np.unique(High_R_Label)

    new_facies_labels = []
    # 关键在于 input_vectors''
    for i in new_label_index:
        new_facies_labels.append(facies_labels[i])

    print(new_facies_labels)



if add_flag ==2:
    cv_conf = confusion_matrix(High_R_Label, pred_facies)
    display_cm(cv_conf, new_facies_labels, hide_zeros=True)
#     plt.figure(figsize=(8,8))
    sns.heatmap(cv_conf, xticklabels = new_facies_labels, yticklabels = new_facies_labels, annot=True, fmt="d", annot_kws={"size": 20});
    plt.title("Confusion matrix", fontsize=20)
    plt.ylabel('True label', fontsize=20)
    plt.xlabel('Predicted label', fontsize=20)
    # plt.savefig(os.path.join(model_testing_img_file_saving_path,model_testing_image_name  + '_all_confusion_matrix.png'), dpi=96,  bbox_inches='tight')
#     plt.savefig(str(model_testing_img_file_saving_path + "/" + model_testing_image_name  + '_all_confusion_matrix.png'), dpi=96,  bbox_inches='tight')
    # plt.show()



# conf,cv_conf







if (add_flag == 1) or (add_flag ==2):
    print('Facies classification accuracy = %f' % accuracy(cv_conf))
#     print('Adjacent facies classification accuracy = %f' % accuracy_adjacent(cv_conf, adjacent_facies))

if (add_flag == 1) or (add_flag ==2):
    per_class_accuracy_list = per_class_accuracy(cv_conf)
    for i in range(len(per_class_accuracy_list)):
        print(i,' class facies classification accuracy = %f' % per_class_accuracy_list[i])


# ## 预测值与真实标定值ROC


if (model_stage == "train") :
    pred_X = dataX
else:
    pred_X = testALL_A_X



if (add_flag == 1) or (add_flag ==2):
    if add_flag == 1:
        loss, accuracy = model.evaluate(pred_X, all_y, verbose=2)
        print("accuracy:",accuracy)
    n_classes = len(facies_labels)
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(all_y[:, i], A_Y_predict[:, i])   
        #max_test样例真实标签，max_predictions学习器预测的样例的概率 
        roc_auc[i] = auc(fpr[i], tpr[i])   
        #计算ROC曲线下方的面积，fpr假正例率数组(横坐标)，tpr真正例率数组(纵坐标） 



if (add_flag == 1) or (add_flag ==2):
    fpr["micro"], tpr["micro"], _ = roc_curve(all_y.ravel(), A_Y_predict.ravel())   #ravel函数将矩阵展开成向量
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])



if (add_flag == 1) or (add_flag ==2):
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))   #np.concatenate将“特征维度相同数组”纵向拼接

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)   #np.zeros_like创建一个和参数all_fpr数组维度相同的全0数组
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])   
            #interp一维线性插值，fpr和tpr是插值结点横纵坐标，all_fpr是已知中间节点横坐标(得到插值曲线后，求其纵坐标)
        #https://docs.scipy.org/doc/numpy/reference/generated/numpy.interp.html#numpy.interp

        # Finally average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])



if (add_flag == 1) or (add_flag ==2):
    plt.rcParams['figure.figsize']=(8,6)
    plt.figure()
    plt.plot(fpr["micro"], tpr["micro"], label='micro-average ROC curve (area = {0:0.2f})' ''.format(roc_auc["micro"]),
                 color='deeppink', linestyle=':')

    plt.plot(fpr["macro"], tpr["macro"], label='macro-average ROC curve (area = {0:0.2f})' ''.format(roc_auc["macro"]),
                 color='navy', linestyle=':')

        # colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])   #python3里的无穷循环器
    line_colors = facies_colors_use
    for i, line_color in zip(range(n_classes), line_colors):
            plt.plot(fpr[i], tpr[i], color=line_color, label='ROC curve of class {0} (area = {1:0.2f})' ''.format(i, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Some extension of Receiver operating characteristic to multi-class')
    plt.legend(loc="lower right")
    if model_stage == "train":
        plt.savefig(os.path.join(model_training_img_file_saving_path, model_training_img_name +"_" + element_name + '_'+ str(n_layers) + "_layers_" +  "_lr_" + str(
            learning_rate) + "h_dim" + str(hidden_dim) + "_epoch_" + str(EPOCHS) + '_all_ROC.png'), dpi=330,  bbox_inches='tight')
    else:
        plt.savefig(os.path.join(model_testing_img_file_saving_path, model_testing_image_name + '_all_ROC.png'), dpi=330,  bbox_inches='tight')
    # plt.show()






# # 曲线结果保存

# ## 结果文件文件头信息


# filename_A ： 'HP1_orginLog_6D_4075m-4280m_R_0.125.csv'
well_name



if use_depth_log == True:
    print("begin_depth,end_depth:",begin_depth,end_depth)



resolution_ratio = 1 / resolution
resolution_ratio


# ## 查看保存的结果曲线


if (add_flag == 1) or (add_flag ==2):
    print(df_pred_data.shape,DEPTH_AddReslution.shape,well_name)







# if use_low_R_data == True:
#     pd_data0 = pd.DataFrame(DEPTH_AddReslution,columns=["DEPTH"])
#     pd_data = pd.concat([pd_data0,df_pred_data],axis=1)
#         # 左右拼接 axis=1, 上下拼接 axis=0
#     pd_data.to_csv(csv_file_saving_path + well_name + "_"+ str(begin_depth) + "_"+ str(end_depth) + "m_" + element_name + '_Add_R_'+ sen.tid_maker() +'.txt',mode='w',float_format='%.4f',sep='\t',index=None,header=True)
#     print("add Resolution Algorithm is Finished!!")
# else:
#     if use_depth_log == True:
#         pd_data0 = pd.DataFrame(DEPTH_AddReslution,columns=["DEPTH"])
#         result_csv_name = well_name + "_"+ str(begin_depth) + "_"+ str(end_depth) + "m_" + element_name + '_Pred_R_'+ sen.tid_maker() +'.txt'
#     else:
#         pd_data0 = pd.DataFrame(all_sample_index,columns=["sample_index"])
#         result_csv_name = well_name + "_" + element_name + '_Pred_R_'+ sen.tid_maker() +'.txt'
#     pd_data1 = pd.DataFrame(testALL_Y_predict_final,columns=[element_name + "_pred"])
#     if use_high_R_data:
#         pd_data2 = pd.DataFrame(High_R_Label,columns=[element_name + "_High_R"])
#         pd_data = pd.concat([pd_data0,pd_data1,pd_data2],axis=1)
#     else:
#         pd_data = pd.concat([pd_data0,pd_data1],axis=1)
if "DEPTH" not in df_pred_data.columns:
    pd_data0 = pd.DataFrame(DEPTH_AddReslution,columns=["DEPTH"])
    pd_data = pd.concat([pd_data0,df_pred_data],axis=1) 
else:
    pd_data = df_pred_data
result_csv_name = well_name + "_" + element_name + '_Pred_R_'+ sen.tid_maker() +'.csv'
# pd_data.to_csv(csv_file_saving_path + result_csv_name,mode='w',float_format='%.4f',sep='\t',index=None,header=True)
pd_data.to_csv(csv_file_saving_path + result_csv_name,mode='w',float_format='%.4f',index=None,header=True)
print("Prediction Algorithm is Finished!!")



if paly_music == True:
    mixer.init()
    music_path = "music/"
    file = "MISIA - 星のように.mp3"
    music_file = os.path.join(music_path,file)
    mixer.music.load(music_file)
    mixer.music.play()
    time.sleep(6)
    mixer.music.stop()
        # exit()







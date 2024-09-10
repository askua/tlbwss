# -*- coding:utf-8 -*-
import numpy as np
import math
import datetime
import pandas as pd
# add
import seaborn as sns
import matplotlib.pylab as plt
import matplotlib
import matplotlib as mpl
import matplotlib.colors as colors
from mpl_toolkits.axes_grid1 import make_axes_locatable
# matplotlib.use('TkAgg')
import tensorflow as tf
import sklearn


def plot_roc(name, labels, predictions, **kwargs):
  fp, tp, _ = sklearn.metrics.roc_curve(labels, predictions)

  plt.plot(100*fp, 100*tp, label=name, linewidth=2, **kwargs)
  plt.xlabel('False positives [%]')
  plt.ylabel('True positives [%]')
  plt.xlim([-0.5,20])
  plt.ylim([80,100.5])
  plt.grid(True)
  ax = plt.gca()
  ax.set_aspect('equal')


def plot_prc(name, labels, predictions, **kwargs):
    precision, recall, _ = sklearn.metrics.precision_recall_curve(labels, predictions)

    plt.plot(precision, recall, label=name, linewidth=2, **kwargs)
    plt.xlabel('Precision')
    plt.ylabel('Recall')
    plt.grid(True)
    ax = plt.gca()
    ax.set_aspect('equal')


# 定义归一化函数
# Define function of Min and Max normalization
def MinMaxScaler(data):
    ''' Min Max Normalization

    Parameters
    ----------
    data : numpy.ndarray
        input data to be normalized
        shape: [Batch size, dimension]

    Returns
    ----------
    data : numpy.ndarry
        normalized data
        shape: [Batch size, dimension]

    References
    ----------
    .. [1] http://sebastianraschka.com/Articles/2014_about_feature_scaling.html

    '''
    numerator = data - np.min(data, 0)
    denominator = np.max(data, 0) - np.min(data, 0)
    # noise term prevents the zero division
    return numerator / (denominator + 1e-7)


# # 定义归一化函数
# def zero_one_scaler(data,log_name):
#     ''' 0-1 Normalization'''
#     result = data.copy()
#     for i in range(len(log_name)):
#         numerator = data.iloc[:,i]-log_name[i][0]
#         denominator = log_name[i][1]-log_name[i][0]
#         result.iloc[:,i]= numerator / (denominator + 1e-7)
#     return  result
# 定义归一化函数
def zero_one_scaler(data,log_name):
    ''' Normalization'''
    result = data.copy()
    # result = np.zeros(data.shape)
    for i in range(len(log_name)):
                # 严格控制范围
        it_data = np.array(data.iloc[:,i])
        for j in range(len(it_data)):
            if it_data[j] < log_name[i][0]:
                it_data[j] = log_name[i][0]
            if it_data[j]  > log_name[i][1]:
                it_data[j] = log_name[i][1]
        
        numerator = it_data - log_name[i][0]
        # numerator_1 = data.iloc[:,i]-log_name[i][1]
        denominator = log_name[i][1]-log_name[i][0]
        result.iloc[:,i] = numerator / (denominator + 1e-8)
        
    return  result


def ZeroOneScaler(data,log_name):
    ''' 0-1 Normalization

    '''
    for i in range(len(log_name)):
        numerator = data.iloc[:,i]-log_name[i][0]
        denominator = log_name[i][1]-log_name[i][0]
#     numerator = data - np.min(data, 0)
#     denominator = np.max(data, 0) - np.min(data, 0)
    # noise term prevents the zero division
    return numerator / (denominator + 1e-7)

# Z-score标准化
def Z_ScoreNormalization(x):
    mu = np.mean(x)
    sigma = np.std(x)
    data = (x - mu) / sigma;

    return data;


# 某目标属性反归一化函数
def revivification_scaler(data,log_name,index):
    result = data.copy()
    for i in range(len(log_name)):
        numerator = log_name[index][0]
        denominator = log_name[index][1]-log_name[index][0]
        result =  data * denominator +  numerator
    return  result


# 反归一化函数
def revivification_all_scaler(data,log_name):
    result = data.copy()
    for i in range(len(log_name)):
        numerator = data.iloc[:,i]-log_name[i][0]
        denominator = log_name[i][1]-log_name[i][0]
        result =  data * denominator +  numerator
    return  result



def multivariate_data(dataset, target, start_index, end_index, history_size,
                      target_size, step, single_step=False):
    data = []
    labels = []

    start_index = start_index + history_size
    if end_index is None:
        end_index = len(dataset) - target_size

    for i in range(start_index, end_index):
        indices = range(i-history_size, i, step)
        data.append(dataset[indices])

    if single_step:
        labels.append(target[i+target_size])
    else:
        labels.append(target[i:i+target_size])

    return np.array(data), np.array(labels)


# 此方法在minBatch中使用
def get_batch(X, Y, batch_size):
    images = tf.cast(X, tf.float32)
    label = tf.cast(Y, tf.float32)
#     input_queue = tf.train.slice_input_producer([images, label], shuffle=False, num_epochs=1)
    input_queue = tf.train.slice_input_producer([images, label], shuffle=False)
    x = input_queue[0]
    y = input_queue[1]
    # 5 调用tf.train.batch生成batch
    # image_batch, label_batch = tf.train.batch([x, y], batch_size=batch_size, num_threads=1, capacity=200)
    image_batch, label_batch = tf.train.batch([x,y], batch_size=batch_size, num_threads=1, capacity=200,allow_smaller_final_batch=True)
    return image_batch, label_batch


# 此方法未用，用了速度慢
def get_net_batch_data(X, Y, batch_size):
    images = tf.cast(X, tf.float32)
    label = tf.cast(Y, tf.float32)
#     input_queue = tf.train.slice_input_producer([images, label], shuffle=False, num_epochs=1)
    input_queue = tf.train.slice_input_producer([images, label], shuffle=False)
    image_batch, label_batch = tf.train.batch(input_queue, batch_size=batch_size, num_threads=1, capacity=200)
    return image_batch, label_batch

# https://blog.csdn.net/silence1214/article/details/77141396

# 原文：https://blog.csdn.net/qq_33373858/article/details/83012236
#train_data训练集特征，train_target训练集对应的标签，batch_size
# 随机取batch_size个训练样本
def next_batch(train_data, train_target, batch_size):
    #打乱数据集
    index = [ i for i in range(0,len(train_target)) ]
    np.random.shuffle(index);
    #建立batch_data与batch_target的空列表
    batch_data = [];
    batch_target = [];
    #向空列表加入训练集及标签
    for i in range(0,batch_size):
        batch_data.append(train_data[index[i]]);
        batch_target.append(train_target[index[i]])
    return batch_data, batch_target #返回


# 随机取batch_size个训练样本
def get_next_batch(train_data, train_target, batch_size, iter_th, iterations):
    data_lenth = len(train_data)
    if(iter_th < iterations):
        # 建立batch_data与batch_target的空列表
        batch_data = []
        batch_target = []
        begin = iter_th * batch_size
        end = (iter_th + 1) * batch_size
        batch_data.append(train_data[begin:end])
        batch_target.append(train_target[begin:end])
    else:
        batch_data = []
        batch_target = []
        index = iter_th * batch_size + 1
        batch_data.append(train_data[index:data_lenth])
        batch_target.append(train_target[index:data_lenth])
    return batch_data, batch_target  # 返回
    # #打乱数据集
    # index = [i for i in range(0,len(train_target))]
    # np.random.shuffle(index);
    # # 建立batch_data与batch_target的空列表
    # batch_data = [];
    # batch_target = [];
    # # 向空列表加入训练集及标签
    # for i in range(0, batch_size):
    #     batch_data.append(train_data[index[i]]);
    #     batch_target.append(train_target[index[i]])
    # return  batch_data, batch_target  # 返回


# build processing datasets with data from A
def build_All_Train_dataset(time_seriesX, time_seriesY, seq_length):
    dataM = []
    dataN = []
    for i in range(0, len(time_seriesX) - seq_length):

        _m = time_seriesX[i:i + seq_length]
        _n = time_seriesY[i + seq_length]
        # print(_z, "->", )
        dataM.append(_m)
        dataN.append(_n)

    return np.array(dataM), np.array(dataN)


# build processing datasets with data from Y
def build_All_Y_dataset(time_series, seq_length):
    dataZ = []
    # dataY = []
    #     for i in range(0, len(time_series)-seq_length):
    for i in range(0, len(time_series) - seq_length):
        # _z = time_series[i:i + seq_length]
        _z = time_series[i + seq_length]
        # _y = time_series[i + seq_length, [-1]]
        # print(_z, "->", )
        dataZ.append(_z)

    return np.array(dataZ)


# build processing datasets with data from A
def build_All_A_dataset(time_series, seq_length):
    dataZ = []
    # dataY = []
    #     for i in range(0, len(time_series)-seq_length):
    for i in range(0, len(time_series) - seq_length):
        # _z = time_series[i:i + seq_length, :]
        _z = time_series[i:i + seq_length]
        # _y = time_series[i + seq_length, [-1]]
        # print(_z, "->", )
        dataZ.append(_z)

    return np.array(dataZ)


# build processing datasets with data from A
def build_All_MLP_dataset(time_series):
    dataZ = []
    # dataY = []
    #     for i in range(0, len(time_series)-seq_length):
    for i in range(0, len(time_series)):
        # _z = time_series[i:i + seq_length, :]
        _z = time_series[i]
        # _y = time_series[i + seq_length, [-1]]
        # print(_z, "->", )
        dataZ.append(_z)

    return np.array(dataZ)


def build_addReslution_DEPTH(DEPTH_use, seq_length):
    dataD = []
    # dataY = []
    #     for i in range(0, len(time_series)-seq_length):
    # for i in range(0, len(DEPTH_use) - seq_length):
    for i in range(seq_length, len(DEPTH_use)):
        _depth = DEPTH_use[i]
        # _depth = DEPTH_use[i:i + seq_length]
        #         _z = DEPTH_use[i:i + seq_length, :]
        # _y = time_series[i + seq_length, [-1]]
        #         print(_depth, "->",)
        dataD.append(_depth)

    return np.array(dataD)

def build_HighReslution_Label(highR_use, seq_length):
    dataD = []
    # dataY = []
    #     for i in range(0, len(time_series)-seq_length):
    # for i in range(0, len(DEPTH_use) - seq_length):
    for i in range(seq_length, len(highR_use)):
        _depth = highR_use[i]
        # _depth = DEPTH_use[i:i + seq_length]
        #         _z = DEPTH_use[i:i + seq_length, :]
        # _y = time_series[i + seq_length, [-1]]
        #         print(_depth, "->",)
        dataD.append(_depth)

    return np.array(dataD)


def dearch_nearest_data(testALL_Y_predict_fianl_GY, index_j):
    for a in range(index_j,len(testALL_Y_predict_fianl_GY)):
        if(testALL_Y_predict_fianl_GY[a] > 0):
            after_current = testALL_Y_predict_fianl_GY[a]
    for b in range(index_j,0,-1):
        if(testALL_Y_predict_fianl_GY[b] > 0):
            before_current = testALL_Y_predict_fianl_GY[b]
    return before_current,after_current


## 2019-0528 10：50 注释
# def build_orgin_log_use(C_element_new, seq_length):
#     dataE = []
#     #     for i in range(0, len(C_element_new)-seq_length):
#     for i in range(seq_length, len(C_element_new)):
#         _element_new = C_element_new[i]
#         if (_element_new < 0):
#             _element_new = 0
#         dataE.append(_element_new)
#
#     return np.array(dataE)

# 方法重写
def build_orgin_log_use(C_element_new, seq_length):
    data_e = []
    # 不用下面的代码是因为低分辨率曲线插值后，比预测的曲线，开始多出一个seq_length长度
    #   for i in range(0, len(C_element_new)-seq_length):
    for i in range(seq_length, len(C_element_new)):
        _element_new = C_element_new[i]
        if (_element_new < 0):
            _element_new = 0
        data_e.append(_element_new)

    return np.array(data_e)



def build_MLP_orgin_log_use(C_element_new):
    data_e = []
    #   for i in range(0, len(C_element_new)-seq_length):
    for i in range(len(C_element_new)):
        _element_new = C_element_new[i]
        if (_element_new < 0):
            _element_new = 0
        data_e.append(_element_new)

    return np.array(data_e)




def build_pred_log_use(testALL_A_Y_chazhi, seq_length):
    dataF = []
    # dataY = []
    #  不用下面的代码是因为最后一段（Seq_length）对应的插值并没有参考预测曲线
    # for i in range(seq_length, len(testALL_A_Y_chazhi)):
    for i in range(0, len(testALL_A_Y_chazhi)-seq_length):
        _testALL_A_Y_chazhi = testALL_A_Y_chazhi[i]
        if (_testALL_A_Y_chazhi < 0):
            _testALL_A_Y_chazhi = 0
        dataF.append(_testALL_A_Y_chazhi)

    return np.array(dataF)

def build_MLP_pred_log_use(testALL_A_Y_chazhi):
    dataF = []
    # dataY = []
    #     for i in range(0, len(testALL_A_Y_chazhi)-seq_length):
    for i in range(len(testALL_A_Y_chazhi)):
        _testALL_A_Y_chazhi = testALL_A_Y_chazhi[i]
        if (_testALL_A_Y_chazhi < 0):
            _testALL_A_Y_chazhi = 0
        dataF.append(_testALL_A_Y_chazhi)

    return np.array(dataF)


# k 表示窗长，k个近邻
# index 表示当前窗口
def calc_mean_with_window(t, k):
    #     current_window = math.ceil(i / k)

    total_window_num = math.ceil(len(t) / k)
    window_mean = np.zeros(total_window_num)
    # range（0， 5） 是[0, 1, 2, 3, 4]
    for index in range(0, total_window_num):
        sum = 0
        #         print(index)
        if (index < total_window_num):
            # 索引窗口没到最后
            # 如第一个窗口，[0,9]
            start = (index - 1) * k
            end = (index * k) - 1 + 1
            for c in range(start, end, 1):
                sum = sum + t[c]
            _window_mean = sum / k

        else:
            # 在最后一个窗长内
            start = (index - 1) * k
            end = (index - 1) * k + (len(t) % k)
            for c in range(start, end, 1):
                sum = sum + t[c]
            _window_mean = sum / k

        window_mean[index] = _window_mean
    return window_mean


def calc_mean_with_nearest_window(t, window_width):
    # t 是一个一维数据组
    data_length = len(t)
    nearest_mean = np.zeros(data_length)
    for index in range(data_length):
    # 位于第一个窗长之后的
        t_list = np.zeros(window_width)
        if index <= window_width :
             for i in range(window_width):
                t_list[i] = t[index]      
        elif((index > window_width) & (index < (data_length - window_width))):
            for i in range(window_width):
                if i <= int(window_width/2):
                    t_list[i] = t[index-i]
                else:
                    t_list[i] = t[index+i]
            # nearest_mean = (t[index] + t[index - 1] + t[index - 2] + t[index - 3] + t[index - 4] + t[index - 5] + + t[index - 6]+ t[index - 7] ) / 8
        else:
            #index in range(data_length - window_width-1,data_length):
            for i in range(window_width):
                t_list[i] = t[index]
        nearest_mean[index] = np.average(t_list)
            # nearest_mean = (t[index] + t[index + 1] + t[index + 2] + t[index + 3] + t[index + 4] + t[index + 5] + + t[index + 6]+ t[index + 7] ) / 8
    return nearest_mean

# 等同于MATLAB中的smooth函数，
# 但是平滑窗口必须为奇数。
# yy = smooth(y) smooths the data in the column vector y ..
# The first few elements of yy are given by
# yy(1) = y(1)
# yy(2) = (y(1) + y(2) + y(3))/3
# yy(3) = (y(1) + y(2) + y(3) + y(4) + y(5))/5
# yy(4) = (y(2) + y(3) + y(4) + y(5) + y(6))/5
# ...
# https://blog.csdn.net/weixin_40532625/article/details/91950668


# 下面的方法与calc_mean_with_nearest_window 效果差不多
def calc_mean_with_nearest_smooth(a, WSZ):
    # a:原始数据，NumPy 1-D array containing the data to be smoothed
    # 必须是1-D的，如果不是，请使用 np.ravel()或者np.squeeze()转化 
    # WSZ: smoothing window size needs, which must be odd number,
    # as in the original MATLAB implementation
    # default WSZ =5
    WSZ = 5
    out0 = np.convolve(a,np.ones(WSZ,dtype=int),'valid')/WSZ
    r = np.arange(1,WSZ-1,2)
    start = np.cumsum(a[:WSZ-1])[::2]/r
    stop = (np.cumsum(a[:-WSZ:-1])[::2]/r)[::-1]
    return np.concatenate((  start , out0, stop  ))


# 滑动平均滤波
# http://www.3qphp.com/python/pybase/4167.html
# https://www.cnblogs.com/xiaosongshine/p/10874644.html
# N为窗口宽度
def fast_moving_average(x, N):
    # return np.convolve(x, np.ones((N,))/float(N))[(N-1):]
    return(np.convolve(x, np.ones((N,))/float(N), mode="same"))
    # return np.convolve(x, np.ones((N,))/N)[(N-1):]


# ###########################################----------滑动窗口提升分辨率函数------------#####################################
def slide_windows_add_resolution(C_orgin_log_use,C_orgin_log_use_mean,testALL_Y_predict_final,Y_predict_final_mean,proportion_list,var_prop_list,final_log_E):
    # 参数说明
    # C_orgin_log_use         --------  低分辨率曲线线性插值为高分辨率后的曲线
    # C_orgin_log_use_mean    --------  低分辨率曲线线性插值为高分辨率后的曲线平滑求均值结果
    # testALL_Y_predict_final --------  模型预测的高分辨率曲线
    # Y_predict_final_mean    --------  模型预测的高分辨率曲线平滑求均值结果
    # proportion_list         --------  权重因子1，取值为一维列表
    # var_prop_list           --------  权重因子2，取值为一维列表
    # final_log_E             --------  结果保存的ndarray，是一个三维数组，第一维度是权重因子1索引，第二维度是权重因子2索引，第三维度是深度值对应索引，
   
    # example
    # proportion_list = [0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.618, 0.8, 0.9999]
    # var_prop_list = [0.001,0.1,0.2,0.618]
    # final_log_E = np.zeros([len(proportion_list),len(var_prop_list),len(testALL_Y_predict_final)])

    diff_current = np.zeros(len(C_orgin_log_use))
    
    log_E_max = max(np.max(testALL_Y_predict_final),np.max(C_orgin_log_use))
    log_E_min = min(np.min(testALL_Y_predict_final),np.min(C_orgin_log_use))
    print("log_E_max:",log_E_max)
    print("log_E_min:",log_E_min)
    
    diff_current = C_orgin_log_use_mean - Y_predict_final_mean
    # 设定权重
    # proportion_list = [0.3, 0.5, 0.618, 0.8, 0.9999]
    # 设定窗口宽
    new_k = 5
    for i in range(len(proportion_list)):
        proportion = proportion_list[i]
        for j in range(len(var_prop_list)):
            var_prop = var_prop_list[j]
            for k in range(0,len(C_orgin_log_use)):
                modified_value = 0
                # 计算观测窗口内两条曲线的差值
                # 定位当前窗口
                # math.ceil(f) #向上取整, math.floor(f) #向下取整
                #current_window = math.ceil(k / new_k)
                # diff_current[i] = C_orgin_log_use_mean[current_window - 1] - testALL_A_Y_chazhi_use_mean[current_window - 1] 
                # diff_current[i] =  C_orgin_log_use_mean[current_window - 1] - Y_predict_final_mean[current_window - 1] 
                # diff_current = C_orgin_log_use_mean - Y_predict_final_mean
                # print(diff_current[i])
                # default proportion = 1 | 黄金分割比
                if(diff_current[k] >= 0):
                    # 如果大于零，那么该段曲线每一个值都加上diff
                    # final_log_E[i] = Y_predict_final_mean[i] - diff_current[i]
                    final_log_E[i][j][k] = testALL_Y_predict_final[k] + proportion * diff_current[k]
                    if(final_log_E[i][j][k] > log_E_max):
                        # final_log_E[i] = C_orgin_log_use[i] + 0.1 * diff_current[i]
                        final_log_E[i][j][k] = C_orgin_log_use_mean[k] - var_prop * diff_current[k]
                else:
                    # 如果小于零，那么该段曲线每一个值都加上diff
                    final_log_E[i][j][k] = testALL_Y_predict_final[k] + proportion * diff_current[k]
                    if(final_log_E[i][j][k] < log_E_min):
                        # 就不改动了
                        # final_log_E[i] = C_orgin_log_use[i] + 0.1 * diff_current[i]
                        final_log_E[i][j][k] = C_orgin_log_use_mean[k] + var_prop * diff_current[k]

    return final_log_E

# ###################################################################----------滑动窗口提升分辨率函数------------##############################


# ############################################均值方差变化提升分辨率涉及的两个函数###############################################################
# 函数第一个变量为标准井，第二个变量为分辨率待提升井
def calc_ab(well_data_1, well_data_2):
    # 第一步：引入标准井和分辨率待提升井
    # 第二步：计算标准井的均值和方差
    well_data_1_mean = np.mean(well_data_1)
    well_data_1_var  = np.var(well_data_1)
    
    # 第三步：计算分辨率待提升井的方法
    well_data_2_mean = np.mean(well_data_2)
    well_data_2_var  = np.var(well_data_2)
    
    # 第四步： 计算变换系数
    core_index_a = np.sqrt(well_data_1_var/(well_data_2_var + 1e-7))
    core_index_b = well_data_1_mean - core_index_a * well_data_2_mean
    
    return core_index_a,core_index_b 

def addR_nearest_window_mean_var(well_1, well_2, window_width):
    # well_1,well_2 是一个一维数据组，二者等长度,window_width是一个整数
    data_length = len(well_1)
    # E_Hist = np.zeros(data_length)
    E_local_Hist = well_2.copy()
    for index in range(data_length):
    # 建立数据用于装well_1
        t_list = np.zeros(window_width)
        s_list = np.zeros(window_width)
        if index <= window_width :
        # 第一个窗口内
            for i in range(window_width):
                t_list[i] = well_1[index]
                s_list[i] = well_2[index]
        elif((index > window_width) and (index < (data_length - window_width))):
            for i in range(window_width):
                if i <= int(window_width/2):
                    t_list[i] = well_1[index-i]
                    s_list[i] = well_2[index-i]
                else:
                    t_list[i] = well_1[index+i]
                    s_list[i] = well_2[index+i]
        else:
            # z最后一个窗口
            for i in range(window_width):
                t_list[i] = well_1[index]
                s_list[i] = well_2[index]
        # 当t_list和s_list取值完毕，调用calc_ab函数
        transform_a, transform_b = calc_ab(t_list,s_list)
        E_local_Hist[index] = transform_a * well_2[index] + transform_b
    return E_local_Hist

# ##############################################################################################################



def tid_maker():
    # return '{0:%Y%m%d%H%M}'.format(datetime.datetime.now())
    return '{0:%m%d%H%M}'.format(datetime.datetime.now())

def tid_date():
    return '{0:%m%d}'.format(datetime.datetime.now())





# 曲线绘制函数
def make_log_plot(logs, element_name, e_colors):
    # reset to original matplotlib style
    # seaborn doesn't look as good for this
    import matplotlib as mpl
    mpl.rcParams.update(mpl.rcParamsDefault)

    # make sure logs are sorted by depth
    logs = logs.sort_values(by='DEPTH')
    cmap_facies = colors.ListedColormap(
        e_colors[0:len(e_colors)], 'indexed')

    ztop = logs.DEPTH.min();
    zbot = logs.DEPTH.max()

    #     cluster=np.repeat(np.expand_dims(logs[label].values,1), 100, 1)

    f, ax = plt.subplots(nrows=1, ncols=4, figsize=(50, 100))

    curve_0 = element_name + "_orgin_log_chazhi"
    curve_1 = element_name + "_pred"
    curve_2 = element_name + "_pred_chazhi"
    curve_3 = element_name + "_add_Resolution"

    ax[0].plot(logs[curve_0], logs.DEPTH, '-g')
    ax[1].plot(logs[curve_1], logs.DEPTH, '-')
    ax[2].plot(logs[curve_2], logs.DEPTH, '-', color='0.5')
    ax[3].plot(logs[curve_3], logs.DEPTH, '-', color='r')
    # ax[4].plot(logs['RD'], logs.DEPTH, '-', color='b')
    # ax[5].plot(logs['RS'], logs.DEPTH, '-', color='g')
    #     im=ax[4].imshow(cluster, interpolation='none', aspect='auto',
    #                     cmap=cmap_facies,vmin=0,vmax=6)

    #     divider = make_axes_locatable(ax[4])
    #     cax = divider.append_axes("right", size="20%", pad=0.05)
    #     cbar=plt.colorbar(im, cax=cax)
    #     cbar.set_label((10*' ').join(['Clust 1', 'Clust 2', 'Clust 3',
    #                                 'Clust 4', 'Clust 5', 'Clust 6']), fontsize=14)
    #     cbar.set_ticks(range(0,1)); cbar.set_ticklabels('')

    for i in range(len(ax) - 1):
        ax[i].set_ylim(ztop, zbot)
        ax[i].invert_yaxis()
        ax[i].grid()
        ax[i].locator_params(axis='x', nbins=3)

    ax[0].set_xlabel(curve_0, fontsize=16)
    ax[0].set_xlim(logs[curve_0].min(), logs[curve_0].max())
    ax[0].set_ylabel('MD [m]', fontsize=24)
    ax[0].tick_params(labelsize=12)
    ax[0].grid(b=False)
    ax[1].set_xlabel(curve_1, fontsize=16)
    ax[1].set_xlim(logs[curve_1].min(), logs[curve_1].max())
    ax[1].grid(b=False)
    ax[1].tick_params(labelsize=12)
    ax[2].set_xlabel(curve_2, fontsize=16)
    ax[2].set_xlim(logs[curve_2].min(), logs[curve_2].max())
    ax[2].grid(b=False)
    ax[2].tick_params(labelsize=12)
    ax[3].set_xlabel(curve_3, fontsize=16)
    ax[3].set_xlim(logs[curve_3].min(), logs[curve_3].max())
    ax[3].grid(b=False)
    ax[3].tick_params(labelsize=12)
    # ax[4].set_xlabel("RD", fontsize=16)
    # ax[4].set_xlim(logs['RD'].min(), logs['RD'].max())
    # ax[4].grid(b=False)
    # ax[4].tick_params(labelsize=12)
    # ax[5].set_xlabel("RS", fontsize=16)
    # ax[5].set_xlim(logs['RS'].min(), logs['RS'].max())
    # ax[5].grid(b=False)
    # ax[5].tick_params(labelsize=12)
    #     ax[4].set_xlabel('Facies', fontsize=16)

    ax[0].set_yticklabels([]);
    ax[1].set_yticklabels([]);
    ax[2].set_yticklabels([]);
    ax[3].set_yticklabels([])
    # ax[4].set_yticklabels([]);
    # ax[5].set_yticklabels([])
    # ax[3].set_xticklabels([])
    # f.suptitle(logs.iloc[0]['Well Name'], fontsize=14,y=0.9)
    return f



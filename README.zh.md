# TLBWSS

### 描述
这是一种关于“基于相似性样本权重的测井评估方法的迁移学习”的实现和应用。

#### 软件架构
软件架构基于 python 3.10+、tesorflow 2.4+、, xgboost, scikit-learn
需要先安装Grpahviz（https://graphviz.org/download/）用于可视化绘制神经网络模型
#### 要求
matplotlib>=3.5.2  
numpy>=1.22.4  
pandas==2.2.2  
pydot_ng==2.0.0  
pygame==2.1.2  
scikit_learn==1.1.2  
scipy==1.14.1  
seaborn==0.11.0  
statsmodels==0.14.2  
tensorflow_gpu==2.9.0  
graphviz  
scikit-learn==1.1.2  
xgboost==2.1.1  
#### 说明

1. prepare_data_utils_notebooks 文件夹中有关于数据预处理的代码。
2. curve_reconstract_TF文件夹展示关于 double experts network 的模型，其他新模型可以添加到 senmodels.py 中。
3. lithofacies_classification_TF文件夹是 BiGRU-MHSA 模型，有助于岩相分类，其他新模型可以添加到senmodels_classification.py
4. data 文件夹展示示例数据

![workflow_chart](figure/workflow_chart_new.png "workflow_chart")
**图 1 工作流程图**

![double_experts_network](figure/double_experts_network_en.png "double_experts_network")
**图 2 双专家网**

![BiGRU-MHSA_network](figure/BiGRU-MHSA_network.png "BiGRU-MHSA_network")
**图 3 BiGRU-MHSA 网络**

电子邮件： xbs150@163.com
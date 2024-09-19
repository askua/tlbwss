# TLBWSS

### Description
It's an implements about "Transfer learning based on weights of similarity sample for well logging evaluation method 
and application".

#### Software Architecture
Software architecture is based on python 3.10+, tesorflow 2.4+, 
You need to install Graphviz （https://graphviz.org/download/） to visualize the neural network model

#### Requirement
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



#### Instructions

1. prepare_data_utils_notebooks folder has the code about data preprocessing. You can process custom data according comment.   
2. curve_reconstract_TF show the model about double experts network, and other new models could be added into senmodels.py.
3. lithofacies_classification_TF is BiGRU-MHSA model to help to lithofacies classification, and other new models could be added into senmodels_classification.py
4. data folder is example data. The default data file format is csv file, input data type is numpy's ndarray. 

If training models, the variable 'model_stage  == "train"' in code; when inference the testing dataset, 
change 'model_stage  == "test"'. This operation is vaild in curve_reconstract_TF as well as lithofacies_classification_TF, 
such as tf2_curve_reconstract.ipynb and tf2_facies_classification.ipynb.


![workflow_chart](figure/workflow_chart_new.png "workflow_chart")
**Figure 1 workflow chart**

![double_experts_network](figure/double_experts_network_en.png "double_experts_network")
**Figure 2 double experts network**

![BiGRU-MHSA_network](figure/BiGRU-MHSA_network.png "BiGRU-MHSA_network")
**Figure 3 BiGRU-MHSA network**

Email: xbs150@163.com


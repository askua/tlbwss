# -*- coding:utf-8 -*-

import tensorflow as tf
# import keras
from tensorflow.keras.layers import Layer
from tensorflow.keras import initializers, regularizers, constraints
from tensorflow.keras.initializers import RandomUniform, Initializer, Constant
from tensorflow import keras
from tensorflow.keras import layers
# from keras_self_attention import SeqSelfAttention
import numpy as np
from sklearn.cluster import KMeans
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, Dense, Activation, Dropout, Lambda, Multiply, Add, Concatenate
from tensorflow.keras.layers import GlobalAveragePooling1D, MaxPool1D, GlobalMaxPool1D, UpSampling1D
from tensorflow.keras.layers import Concatenate,concatenate
# from capsule_layer import Capsule
# from attention_layer import AttentionLayer

from tensorflow.keras import backend as Kbackend

# from __future__ import absolute_import
# import numpy as np

# # BLSTM_atten_CRF模型
# from crf_layer import CRF
# from attention_layer import AttentionSelf


# # # SRU
# from keras import backend as K
# from keras import activations
# from keras import initializers
# from keras import regularizers
# from keras import constraints
# from keras.engine import InputSpec
# from keras.legacy import interfaces
# from keras.layers import Recurrent


# #-------------------model_parameter--------------------------------
# 该类在方法中定义
class MyModelParameter:
     # 参数类的定义�?以只用下面的属性，也可以利用__init__()的构造方�?
#     MAX_SAMPLE_NUM = 2000
#     # 输入维度
#     data_dim = input_vectors_dim
#     seq_length = 5
#     hidden_dim = 72
#     # 输出维度
#     # output_dim = 1
#     output_dim = 1
#     learning_rate = 0.01
#     dropout_rate = 0.1    # 在�??练过程中设置
#     n_layers = 3  # LSTM layer 层数
#     # batch_size = 100
#     BATCH_SIZE = 800
#     # iterations = 300
#     EPOCHS = 15
    
    def __init__(self,data_dim,seq_length, hidden_dim, output_dim,learning_rate,dropout_rate,n_layers,BATCH_SIZE,EPOCHS):
        self.data_dim = data_dim
        self.seq_length = seq_length
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.learning_rate = learning_rate
        self.dropout_rate = dropout_rate
        self.n_layers = n_layers
        self.batch_size = BATCH_SIZE
        self.EPOCHS = EPOCHS
# #----------------------------------------------------


# def bilstm_capsule_model(model_parameter):
#     model = tf.keras.Sequential()
#     model.add(layers.LSTM(model_parameter.hidden_dim, input_shape=(model_parameter.seq_length, model_parameter.data_dim), return_sequences=True))
#     # model.add(layers.InputLayer(input_shape=(model_parameter.seq_length,model_parameter.data_dim),batch_size= None))
#     # model.add(layers.InputLayer(input_shape=(model_parameter.data_dim,),batch_size= None))
#     model.add(layers.BatchNormalization())
#     # model.add(layers.Bidirectional(layers.LSTM(model_parameter.hidden_dim,return_sequences=True)))
#     # model.add(layers.Dropout(model_parameter.dropout_rate))
#     for i in range(model_parameter.n_layers):
#         capsule = Capsule(num_capsule = model_parameter.hidden_dim//2,
#                              dim_capsule = model_parameter.output_dim,
#                              routings = model_parameter.seq_length * 2,
#                              # activation = 'relu',
#                              share_weights = True)
#         model.add(capsule)
   
#     # model.add(layers.Dense(model_parameter.output_dim))
#     # model.add(layers.Dropout(model_parameter.dropout_rate))
# #     capsule = Capsule(num_capsule=model_parameter.seq_length,
# #                              dim_capsule=model_parameter.output_dim,
# #                              routings=4,
# #                              activation='relu',
# #                              share_weights=True)
# #     model.add(capsule)
#     model.add(layers.GRU(model_parameter.hidden_dim,return_sequences=False))
#     # model.add(layers.Flatten())
#     model.add(layers.Dropout(model_parameter.dropout_rate))
#     model.add(layers.Dense(model_parameter.output_dim))

#     optimizer = tf.keras.optimizers.Adam(model_parameter.learning_rate)
#     # loss = 'mse', 'mean_squared_error', 'huber_loss'
#     my_loss = 'mse'
#     model.compile(optimizer=optimizer,
#                   loss = my_loss,
#                   metrics=['mae', 'mse'])
#     return model


# # BLSTM_atten_CRF模型
# def bi_lstm_atten_crf_model(model_parameter):
#     model = tf.keras.Sequential()
#     model.add(layers.LSTM(model_parameter.hidden_dim, input_shape=(model_parameter.seq_length, model_parameter.data_dim), return_sequences=True))
#     model.add(layers.BatchNormalization())
#     # model.add(TimeDistributed(keras.layers.Dropout(model_parameter.dropout_rate)))
#     # model.add(layers.Dropout(model_parameter.dropout_rate))
#     for i in range(model_parameter.n_layers-1):
#         model.add(layers.Bidirectional(layers.LSTM(model_parameter.hidden_dim,return_sequences=True)))
#         model.add(layers.Dropout(model_parameter.dropout_rate))
#     model.add(layers.Bidirectional(layers.LSTM(model_parameter.hidden_dim,return_sequences=True)))
#     # use learn_mode = 'join', test_mode = 'viterbi', sparse_target = True (label indice output)
#     # model.add(TimeDistributed(Dropout(model_parameter.dropout_rate)))

#     # model.add(layers.Dropout(model_parameter.dropout_rate))
#     # model.add(layers.Dense(model_parameter.output_dim))
    
#     # crf = CRF(1)
#     atten = AttentionSelf(model_parameter.output_dim)
#     model.add(atten)
#     model.add(layers.Dropout(model_parameter.dropout_rate))
#     model.add(layers.Dense(model_parameter.output_dim))
#     model.add(CRF(model_parameter.output_dim))
#     optimizer = tf.keras.optimizers.Adam(model_parameter.learning_rate)
#     # loss = 'mse', 'mean_squared_error', 'huber_loss'
#     my_loss = 'mse'
#     model.compile(optimizer=optimizer,
#                   loss = my_loss,
#                   metrics=['mae', 'mse'])
#     return model



def gru_cell_model(model_parameter):
    # cell = tf.contrib.rnn.GRUCell(hidden_dim)
    # # ======增加GRU单元�?20170710�?�?=========
    # with tf.name_scope('gru_dropout'):
    #     gru_cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=keep_prob)
    # return gru_cell
    model = tf.keras.Sequential()
    # model.add(layers.Embedding(input_dim=MAX_SAMPLE_NUM, output_dim=output_dim, input_length=data_dim))
    # model.add(layers.Embedding(input_dim=seq_length, output_dim=output_dim, input_length=data_dim))
    # model.add(layers.InputLayer(input_shape=(model_parameter.seq_length,model_parameter.data_dim),batch_size= model_parameter.batch_size))

    model.add(layers.GRU(model_parameter.hidden_dim, input_shape=(model_parameter.seq_length, model_parameter.data_dim), return_sequences=True))
    model.add(layers.Dropout(model_parameter.dropout_rate))
    for i in range(model_parameter.n_layers-1):
        model.add(layers.GRU(model_parameter.hidden_dim,return_sequences=True))
        model.add(layers.Dropout(model_parameter.dropout_rate))
    model.add(layers.GRU(model_parameter.output_dim, return_sequences=False))
    model.add(layers.Dropout(model_parameter.dropout_rate))
    model.add(layers.Dense(model_parameter.output_dim))
    # model.add(layers.Dense(model_parameter.output_dim, activation='sigmoid'))
#     model.add(Dense(1))
#     model.add(Activation('sigmoid'))
    optimizer = tf.keras.optimizers.RMSprop(model_parameter.learning_rate)
    model.compile(optimizer=optimizer,
                  loss='mse',
                  metrics=['mae', 'mse'])
    return model


def gru_block_cell_2(model_parameter):
    model = tf.keras.Sequential()
#     model.add(layers.Embedding(input_dim=MAX_SAMPLE_NUM, output_dim=output_dim, input_length=data_dim))
    # model.add(layers.Embedding(input_dim=seq_length, output_dim=output_dim, input_length=data_dim))
    # model.add(layers.InputLayer(input_shape=(seq_length,data_dim),batch_size= None))
    model.add(layers.InputLayer(input_shape=(model_parameter.seq_length,model_parameter.data_dim),batch_size= None))
    model.add(tf.keras.layers.BatchNormalization())
    for i in range(model_parameter.n_layers):
        model.add(layers.GRU(model_parameter.hidden_dim * 2,return_sequences=True))
        # model.add(tf.keras.layers.BatchNormalization())
        model.add(layers.Dropout(model_parameter.dropout_rate))
    # model.add(layers.GRU(model_parameter.output_dim, activation='sigmoid', return_sequences=False))
    model.add(layers.GRU(model_parameter.output_dim, return_sequences=False))
    # model.add(tf.keras.layers.BatchNormalization())
    # model.add(layers.Dropout(dropout_rate))
    # model.add(layers.Dense(output_dim, activation='sigmoid'))
#     model.add(Dense(1))
#     model.add(Activation('sigmoid'))
    optimizer = tf.keras.optimizers.RMSprop(model_parameter.learning_rate)
    model.compile(optimizer=optimizer,
                  loss='mse',
                  metrics=['mae', 'mse'])
    return model


def lstm_cell_model(model_parameter):
    model = tf.keras.Sequential()
    # model.add(layers.Embedding(input_dim=MAX_SAMPLE_NUM, output_dim=output_dim, input_length=data_dim))
    # model.add(layers.Embedding(input_dim=seq_length, output_dim=output_dim, input_length=data_dim))

    # model.add(layers.InputLayer(input_shape=(model_parameter.seq_length,model_parameter.data_dim),batch_size= None))
    # ##################### Using Lstm layer as input layer
    model.add(layers.LSTM(model_parameter.hidden_dim, input_shape=(model_parameter.seq_length, model_parameter.data_dim), return_sequences=True))
    model.add(layers.Dropout(model_parameter.dropout_rate))
    # model.add(layers.Dense(model_parameter.hidden_dim, activation='relu',input_shape=(model_parameter.seq_length,model_parameter.data_dim)))
    # for i in range(model_parameter.n_layers):
    for i in range(model_parameter.n_layers-1):
        # Bidirectional(LSTM())
        # model.add(layers.LSTM(model_parameter.hidden_dim, kernel_regularizer=keras.regularizers.l2(0.01),
                # activity_regularizer=keras.regularizers.l1(0.01) , return_sequences=True))
        model.add(layers.LSTM(model_parameter.hidden_dim, return_sequences=True))
        model.add(layers.Dropout(model_parameter.dropout_rate))
    # the last lstm layer doesn't need return_sequences
    model.add(layers.LSTM(model_parameter.output_dim, return_sequences=False))
    model.add(layers.Dropout(model_parameter.dropout_rate))
    # the last output layer
    model.add(layers.Dense(model_parameter.output_dim))
    # model.add(layers.Dense(model_parameter.output_dim, activation='sigmoid'))
    # optimizer = tf.keras.optimizers.RMSprop(model_parameter.learning_rate)
    optimizer = tf.keras.optimizers.Adam(model_parameter.learning_rate)
    model.compile(optimizer=optimizer,
                #   loss='mse',
                  loss='mse',
                  metrics=['mae', 'mse'])
    return model


def dnn_model(model_parameter):
    model = tf.keras.Sequential()
    model.add(layers.Dense(model_parameter.hidden_dim, input_shape=(model_parameter.data_dim,)))
    model.add(layers.Dropout(model_parameter.dropout_rate))
    for i in range(model_parameter.n_layers-2):
        model.add(layers.Dense(model_parameter.hidden_dim))
        model.add(layers.Dropout(model_parameter.dropout_rate))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(layers.Dense(model_parameter.output_dim, activation='relu'))
    optimizer = tf.keras.optimizers.Adam(model_parameter.learning_rate)
    model.compile(optimizer=optimizer,
                  loss='mse',
                  metrics=['mae', 'mse'])
    return model


def dnn_model_2(model_parameter):
    model = tf.keras.Sequential()
    model.add(layers.Dense(model_parameter.hidden_dim, activation='relu',input_shape=(model_parameter.seq_length, model_parameter.data_dim)))
    model.add(layers.Flatten())
    model.add(tf.keras.layers.BatchNormalization())
    # model.add(layers.Embedding(input_dim=seq_length, output_dim=output_dim, input_length=data_dim))
    for i in range(model_parameter.n_layers):
        model.add(layers.Dense(model_parameter.hidden_dim))
        model.add(layers.Dropout(model_parameter.dropout_rate))
    model.add(layers.Dense(model_parameter.output_dim))
    optimizer = tf.keras.optimizers.Adam(model_parameter.learning_rate)
    model.compile(optimizer=optimizer,
                  loss='mse',
                  metrics=['mae', 'mse'])
    return model

# BLSTM模型
def bi_lstm_cell_model(model_parameter):
    input_data = tf.keras.layers.Input(shape=(model_parameter.seq_length,model_parameter.data_dim))
    input_data_G = tf.keras.layers.BatchNormalization()(input_data)
    bilstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(model_parameter.hidden_dim, return_sequences=True))(input_data_G)  # lstm입력은 (N, X, Y) 3�?원이어여한다
    for i in range(model_parameter.n_layers-1):
        bilstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(model_parameter.hidden_dim * 2,return_sequences=True))(bilstm)
        bilstm = tf.keras.layers.Dropout(model_parameter.dropout_rate)(bilstm)
    bilstm = tf.keras.layers.Bidirectional(layers.LSTM(model_parameter.hidden_dim * 4,return_sequences=False))(bilstm)
    bilstm = tf.keras.layers.Dropout(model_parameter.dropout_rate)(bilstm)
    
    bilstm_output = Dense(model_parameter.output_dim)(bilstm)
    my_optimizer = tf.keras.optimizers.Adam(model_parameter.learning_rate)

    model = tf.keras.models.Model(inputs=[input_data],
                outputs=[bilstm_output])
    # loss = 'mse', 'mean_squared_error', 'huber_loss'
    my_loss = 'huber_loss'
    model.compile(loss='mean_squared_error', optimizer = my_optimizer,metrics=['mae', 'mse'])

    return model


# bilstm_atten_model
def bilstm_atten_model(model_parameter):
    lstm_dim = 50  # 替换model_parameter.hidden_dim
    input_data = tf.keras.layers.Input(shape=(model_parameter.seq_length,model_parameter.data_dim))
    # input_data_G = tf.keras.layers.BatchNormalization()(input_data)
    bilstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(model_parameter.hidden_dim, return_sequences=True))(input_data)  # lstm입력은 (N, X, Y) 3�?원이어여한다
    for i in range(model_parameter.n_layers-1):
        bilstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(model_parameter.hidden_dim,return_sequences=True))(bilstm)
        bilstm = tf.keras.layers.Dropout(model_parameter.dropout_rate)(bilstm)
    # bilstm = tf.keras.layers.Bidirectional(layers.LSTM(model_parameter.hidden_dim,return_sequences=False))(bilstm)
    # bilstm = tf.keras.layers.Dropout(model_parameter.dropout_rate)(bilstm)
    
    # bilstm = LSTM(2*lstm_dim, return_sequences=True)(input_data)  # lstm입력은 (N, X, Y) 3�?원이어여한다
    # bilstm_output = Dense(1)(bilstm)

    attention_layer = AttentionLayer()(bilstm)
    print(attention_layer)

    repeated_word_attention = tf.keras.layers.RepeatVector(model_parameter.hidden_dim * 2)(attention_layer)
    repeated_word_attention = tf.keras.layers.Permute([2, 1])(repeated_word_attention)
    sentence_representation = tf.keras.layers.Multiply()([bilstm, repeated_word_attention])
    sentence_representation = tf.keras.layers.Lambda(lambda x: Kbackend.sum(x, axis=1))(sentence_representation)

    bilstm_output = tf.keras.layers.Dense(model_parameter.output_dim)(sentence_representation)

    my_optimizer = tf.keras.optimizers.Adam(model_parameter.learning_rate)

    model = tf.keras.models.Model(inputs=[input_data],
                outputs=[bilstm_output])
    # 3. �?�? 학습과정 설정하기
    # model.compile(loss='mean_squared_error', optimizer='adam',metrics=['mae', 'mse'])
    model.compile(loss='mean_squared_error', optimizer = my_optimizer,metrics=['mae', 'mse'])

    return model


# bilstm_atten_model_2
def bilstm_atten_model_2(model_parameter):
    lstm_dim = 50  # 替换model_parameter.hidden_dim
    input_data = tf.keras.layers.Input(shape=(model_parameter.seq_length,model_parameter.data_dim))
    # input_data_G = tf.keras.layers.BatchNormalization()(input_data)
    bilstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(model_parameter.hidden_dim, return_sequences=True))(input_data)  # lstm입력은 (N, X, Y) 3�?원이어여한다
    for i in range(model_parameter.n_layers-1):
        bilstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(model_parameter.hidden_dim,return_sequences=True))(bilstm)
        bilstm = tf.keras.layers.Dropout(model_parameter.dropout_rate)(bilstm)
        attention_layer = AttentionLayer()(bilstm)
        print(attention_layer)

        repeated_word_attention = tf.keras.layers.RepeatVector(model_parameter.hidden_dim * 2)(attention_layer)
        repeated_word_attention = tf.keras.layers.Permute([2, 1])(repeated_word_attention)
        sentence_representation = tf.keras.layers.Multiply()([bilstm, repeated_word_attention])
    
    sentence_representation = tf.keras.layers.Lambda(lambda x: Kbackend.sum(x, axis=1))(sentence_representation)

    bilstm_output = tf.keras.layers.Dense(model_parameter.output_dim)(sentence_representation)

    my_optimizer = tf.keras.optimizers.Adam(model_parameter.learning_rate)

    model = tf.keras.models.Model(inputs=[input_data],
                outputs=[bilstm_output])
    # 3. �?�? 학습과정 설정하기
    # model.compile(loss='mean_squared_error', optimizer='adam',metrics=['mae', 'mse'])
    model.compile(loss='mean_squared_error', optimizer = my_optimizer,metrics=['mae', 'mse'])

    return model


# bilstm_atten_model_3
def bilstm_atten_model_3(model_parameter):
    lstm_dim = 50  # 替换model_parameter.hidden_dim
    input_data = tf.keras.layers.Input(shape=(model_parameter.seq_length,model_parameter.data_dim))
    input_data_G = tf.keras.layers.BatchNormalization()(input_data)
    bilstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(model_parameter.hidden_dim, return_sequences=True))(input_data_G)  # lstm입력은 (N, X, Y) 3�?원이어여한다
    skips = []

    for i in range(model_parameter.n_layers-1):
        bilstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(model_parameter.hidden_dim,return_sequences=True))(bilstm)
        # bilstm = tf.keras.layers.Dropout(model_parameter.dropout_rate)(bilstm)
        attention_layer = AttentionLayer()(bilstm)
        print(attention_layer)

        repeated_word_attention = tf.keras.layers.RepeatVector(model_parameter.hidden_dim * 2)(attention_layer)
        repeated_word_attention = tf.keras.layers.Permute([2, 1])(repeated_word_attention)
        sentence_representation = tf.keras.layers.Multiply()([bilstm, repeated_word_attention])
    
        sentence_representation = tf.keras.layers.Lambda(lambda x: Kbackend.sum(x, axis=1))(sentence_representation)
        
        skips.append(sentence_representation)
    
    out_block = Activation('relu')(Add()(skips))
    bilstm_output = tf.keras.layers.Dense(model_parameter.output_dim)(out_block)

    my_optimizer = tf.keras.optimizers.Adam(model_parameter.learning_rate)

    model = tf.keras.models.Model(inputs=[input_data],
                outputs=[bilstm_output])
    # model.compile(loss='mean_squared_error', optimizer='adam',metrics=['mae', 'mse'])
    model.compile(loss='mean_squared_error', optimizer = my_optimizer,metrics=['mae', 'mse'])

    return model

# bilstm_atten_model_4
def bilstm_atten_model_4(model_parameter):
    lstm_dim = 50  # 替换model_parameter.hidden_dim
    input_data = tf.keras.layers.Input(shape=(model_parameter.seq_length,model_parameter.data_dim))
    input_data_G = tf.keras.layers.BatchNormalization()(input_data)
    bilstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(model_parameter.hidden_dim, return_sequences=True))(input_data_G)  # lstm입력은 (N, X, Y) 3�?원이어여한다
    skips = []

    for i in range(model_parameter.n_layers-1):
        bilstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(model_parameter.hidden_dim,return_sequences=True))(bilstm)
        # bilstm = tf.keras.layers.Dropout(model_parameter.dropout_rate)(bilstm)
        attention_layer = AttentionLayer()(bilstm)
        print(attention_layer)

        repeated_word_attention = tf.keras.layers.RepeatVector(model_parameter.hidden_dim * 2)(attention_layer)
        repeated_word_attention = tf.keras.layers.Permute([2, 1])(repeated_word_attention)
        sentence_representation = tf.keras.layers.Multiply()([bilstm, repeated_word_attention])
    
        sentence_representation = tf.keras.layers.Lambda(lambda x: Kbackend.sum(x, axis=1))(sentence_representation)
        
        skips.append(sentence_representation)
    
    # out_block = Activation('relu')(Add()(skips))
#     for j in range(len(skips)):
#         out_block = tf.keras.layers.Concatenate()([skips[j],x])
    out_block = Activation('relu')(Concatenate()(skips))
    
    bilstm_output = tf.keras.layers.Dense(model_parameter.output_dim)(out_block)

    my_optimizer = tf.keras.optimizers.Adam(model_parameter.learning_rate)

    model = tf.keras.models.Model(inputs=[input_data],
                outputs=[bilstm_output])
    # model.compile(loss='mean_squared_error', optimizer='adam',metrics=['mae', 'mse'])
    model.compile(loss='mean_squared_error', optimizer = my_optimizer,metrics=['mae', 'mse'])

    return model

# bilstm_atten_model_5
def bilstm_atten_model_5(model_parameter):
    lstm_dim = 50  # 替换model_parameter.hidden_dim
    input_data = tf.keras.layers.Input(shape=(model_parameter.seq_length,model_parameter.data_dim))
    input_data_G = tf.keras.layers.BatchNormalization()(input_data)
    bilstm = input_data_G  # lstm입력은 (N, X, Y) 3�?원이어여한다
    skips = []

    for i in range(model_parameter.n_layers):
        bilstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(model_parameter.hidden_dim,return_sequences=True))(bilstm)
        # bilstm = tf.keras.layers.Dropout(model_parameter.dropout_rate)(bilstm)
        attention_layer = AttentionLayer()(bilstm)
        print(attention_layer)

        repeated_word_attention = tf.keras.layers.RepeatVector(model_parameter.hidden_dim * 2)(attention_layer)
        repeated_word_attention = tf.keras.layers.Permute([2, 1])(repeated_word_attention)
        sentence_representation = tf.keras.layers.Multiply()([bilstm, repeated_word_attention])
    
        sentence_representation = tf.keras.layers.Lambda(lambda x: Kbackend.sum(x, axis=1))(sentence_representation)
        
        skips.append(sentence_representation)
    
    out_block = Activation('relu')(Add()(skips))
#     for j in range(len(skips)):
#         out_block = tf.keras.layers.Concatenate()([skips[j],x])
    # out_block = Activation('relu')(Concatenate()(skips))
    
    bilstm_output = tf.keras.layers.Dense(model_parameter.output_dim)(out_block)

    my_optimizer = tf.keras.optimizers.Adam(model_parameter.learning_rate)

    model = tf.keras.models.Model(inputs=[input_data],
                outputs=[bilstm_output])
    # model.compile(loss='mean_squared_error', optimizer='adam',metrics=['mae', 'mse'])
    model.compile(loss='mean_squared_error', optimizer = my_optimizer,metrics=['mae', 'mse'])

    return model


# bilstm_atten_model_6
def bilstm_atten_model_6(model_parameter):
    lstm_dim = 50  # 替换model_parameter.hidden_dim
    input_data = tf.keras.layers.Input(shape=(model_parameter.seq_length,model_parameter.data_dim))
    input_data_G = tf.keras.layers.BatchNormalization()(input_data)
    bilstm = input_data_G  # lstm입력은 (N, X, Y) 3�?원이어여한다
    skips = []

    for i in range(model_parameter.n_layers):
        bilstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(model_parameter.hidden_dim,return_sequences=True))(bilstm)
        # bilstm = tf.keras.layers.Dropout(model_parameter.dropout_rate)(bilstm)
        attention_layer = AttentionLayer()(bilstm)
        print(attention_layer)

        repeated_word_attention = tf.keras.layers.RepeatVector(model_parameter.hidden_dim * 2)(attention_layer)
        repeated_word_attention = tf.keras.layers.Permute([2, 1])(repeated_word_attention)
        sentence_representation = tf.keras.layers.Multiply()([bilstm, repeated_word_attention])
    
        sentence_representation = tf.keras.layers.Lambda(lambda x: Kbackend.sum(x, axis=1))(sentence_representation)
        
        skips.append(sentence_representation)
    
    # out_block = Activation('relu')(Add()(skips))
#     for j in range(len(skips)):
#         out_block = tf.keras.layers.Concatenate()([skips[j],x])
    out_block = Activation('relu')(Concatenate()(skips))
    
    bilstm_output = tf.keras.layers.Dense(model_parameter.output_dim)(out_block)

    my_optimizer = tf.keras.optimizers.Adam(model_parameter.learning_rate)

    model = tf.keras.models.Model(inputs=[input_data],
                outputs=[bilstm_output])
    # model.compile(loss='mean_squared_error', optimizer='adam',metrics=['mae', 'mse'])
    model.compile(loss='mean_squared_error', optimizer = my_optimizer,metrics=['mae', 'mse'])

    return model



# BiGRU模型
def bi_gru_cell_model(model_parameter):
    model = tf.keras.Sequential()
    # model.add(layers.Embedding(input_dim=MAX_SAMPLE_NUM, output_dim=output_dim, input_length=data_dim))
    # model.add(layers.Embedding(input_dim=seq_length, output_dim=output_dim, input_length=data_dim))
    # model.add(layers.InputLayer(input_shape=(model_parameter.seq_length,model_parameter.data_dim),batch_size= None))
    model.add(layers.GRU(model_parameter.hidden_dim, input_shape=(model_parameter.seq_length, model_parameter.data_dim), return_sequences=True))
    model.add(tf.keras.layers.BatchNormalization())
    # model.add(layers.Dropout(model_parameter.dropout_rate))
    for i in range(model_parameter.n_layers-1):
        model.add(layers.Bidirectional(layers.GRU(model_parameter.hidden_dim,return_sequences=True)))
        model.add(layers.Dropout(model_parameter.dropout_rate))
    model.add(layers.Bidirectional(layers.GRU(model_parameter.hidden_dim,return_sequences=False)))
    model.add(layers.Dropout(model_parameter.dropout_rate))
    model.add(layers.Dense(model_parameter.output_dim))
    optimizer = tf.keras.optimizers.Adam(model_parameter.learning_rate)
    # loss = 'mse', 'mean_squared_error', 'huber_loss'
    my_loss = 'huber_loss'
    model.compile(optimizer=optimizer,
                  loss='huber_loss',
                  metrics=['mae', 'mse'])
    return model

# bigru_atten_model
def bigru_atten_model(model_parameter):
    lstm_dim = 50  # 替换model_parameter.hidden_dim
    input_data = tf.keras.layers.Input(shape=(model_parameter.seq_length,model_parameter.data_dim))
    input_data_G = tf.keras.layers.BatchNormalization()(input_data)
    bigru = input_data_G  # lstm입력은 (N, X, Y) 3�?원이어여한다
    skips = []

    for i in range(model_parameter.n_layers):
        bigru = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(model_parameter.hidden_dim,return_sequences=True))(bigru)
        # bigru = tf.keras.layers.Dropout(model_parameter.dropout_rate)(bigru)
        attention_layer = AttentionLayer()(bigru)
        print(attention_layer)

        repeated_word_attention = tf.keras.layers.RepeatVector(model_parameter.hidden_dim * 2)(attention_layer)
        repeated_word_attention = tf.keras.layers.Permute([2, 1])(repeated_word_attention)
        sentence_representation = tf.keras.layers.Multiply()([bigru, repeated_word_attention])
    
        sentence_representation = tf.keras.layers.Lambda(lambda x: Kbackend.sum(x, axis=1))(sentence_representation)
        
        skips.append(sentence_representation)
    
    out_block = Activation('relu')(Add()(skips))
#     for j in range(len(skips)):
#         out_block = tf.keras.layers.Concatenate()([skips[j],x])
    # out_block = Activation('relu')(Concatenate()(skips))
    
    bigru_output = tf.keras.layers.Dense(model_parameter.output_dim)(out_block)

    my_optimizer = tf.keras.optimizers.Adam(model_parameter.learning_rate)

    model = tf.keras.models.Model(inputs=[input_data],
                outputs=[bigru_output])
    # model.compile(loss='mean_squared_error', optimizer='adam',metrics=['mae', 'mse'])
    model.compile(loss='mean_squared_error', optimizer = my_optimizer,metrics=['mae', 'mse'])

    return model


# bigru_atten_model_2
def bigru_atten_model_2(model_parameter):
    lstm_dim = 50  # 替换model_parameter.hidden_dim
    input_data = tf.keras.layers.Input(shape=(model_parameter.seq_length,model_parameter.data_dim))
    input_data_G = tf.keras.layers.BatchNormalization()(input_data)
    bigru = input_data_G  # lstm입력은 (N, X, Y) 3�?원이어여한다
    skips = []

    for i in range(model_parameter.n_layers):
        bigru = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(model_parameter.hidden_dim * 2,return_sequences=True))(bigru)
        # bigru = tf.keras.layers.Dropout(model_parameter.dropout_rate)(bigru)
        attention_layer = AttentionLayer()(bigru)
        print(attention_layer)

        repeated_word_attention = tf.keras.layers.RepeatVector(model_parameter.hidden_dim * 4)(attention_layer)
        repeated_word_attention = tf.keras.layers.Permute([2, 1])(repeated_word_attention)
        sentence_representation = tf.keras.layers.Multiply()([bigru, repeated_word_attention])
    
        sentence_representation = tf.keras.layers.Lambda(lambda x: Kbackend.sum(x, axis=1))(sentence_representation)
        
        skips.append(sentence_representation)
    
    # out_block = Activation('relu')(Add()(skips))
#     for j in range(len(skips)):
#         out_block = tf.keras.layers.Concatenate()([skips[j],x])
    out_block = Activation('relu')(Concatenate()(skips))
    
    bigru_output = tf.keras.layers.Dense(model_parameter.output_dim)(out_block)

    my_optimizer = tf.keras.optimizers.Adam(model_parameter.learning_rate)

    model = tf.keras.models.Model(inputs=[input_data],
                outputs=[bigru_output])
    # model.compile(loss='mean_squared_error', optimizer='adam',metrics=['mae', 'mse'])
    model.compile(loss='mean_squared_error', optimizer = my_optimizer,metrics=['mae', 'mse'])

    return model



# wavenet模型
def wavenet_model(model_parameter):
    # convolutional operation parameters
    n_filters = model_parameter.hidden_dim # 32 
    filter_width = 2
    # dilation_rates = [2**i for i in range(8)] * 2   # 
    dilation_rates = [2**i for i in range(model_parameter.seq_length)] * 2 
    
    # define an input history series and pass it through a stack of dilated causal convolution blocks. 
    history_seq = Input(shape=(model_parameter.seq_length, model_parameter.data_dim))
    x = history_seq

    skips = []
    for dilation_rate in dilation_rates:
        
        # preprocessing - equivalent to time-distributed dense
        x = Conv1D(16, 1, padding='same', activation='relu')(x) 
        
        # filter convolution
        x_f = Conv1D(filters=n_filters,
                    kernel_size=filter_width, 
                    padding='causal',
                    dilation_rate=dilation_rate)(x)
        
        # gating convolution
        x_g = Conv1D(filters=n_filters,
                    kernel_size=filter_width, 
                    padding='causal',
                    dilation_rate=dilation_rate)(x)
        
        # multiply filter and gating branches
        z = Multiply()([Activation('tanh')(x_f),
                        Activation('sigmoid')(x_g)])
        
        # postprocessing - equivalent to time-distributed dense
        z = Conv1D(16, 1, padding='same', activation='relu')(z)
        
        # residual connection
        x = Add()([x, z])    
        
        # collect skip connections
        skips.append(z)

    # add all skip connection outputs 
    out = Activation('relu')(Add()(skips))

    # final time-distributed dense layers 
    out = Conv1D(128, 1, padding='same')(out)
    out = Activation('relu')(out)
    out = Dropout(model_parameter.dropout_rate)(out)
    out = Conv1D(1, 1, padding='same')(out)
    out = GlobalAveragePooling1D()(out)
    pred_seq_train = layers.Dense(model_parameter.output_dim)(out)

    # extract the last 60 time steps as the training target
    # def slice(x, seq_length):
    #     return x[:,-seq_length:,:]

    # pred_seq_train = Lambda(slice, arguments={'seq_length':model_parameter.seq_length})(out)

    model = Model(history_seq, pred_seq_train)
    
    optimizer = tf.keras.optimizers.Adam(model_parameter.learning_rate)
    # optimizer = tf.keras.optimizers.RMSprop(model_parameter.learning_rate)
    # loss = 'mse', 'mean_squared_error', 'huber_loss'
    my_loss = 'mse'
    model.compile(optimizer=optimizer,
                  loss = my_loss,
                  metrics=['mae', 'mse'])
    return model


# wavenet2模型
def wavenet_model2(model_parameter):
    # convolutional operation parameters
    n_filters = model_parameter.hidden_dim # 32 
    filter_width = 2
    # dilation_rates = [2**i for i in range(8)] * 2   # 
    # dilation_rates = [2**i for i in range(model_parameter.seq_length)] * 2
    dilation_rates = [2**i for i in range(model_parameter.seq_length)] * 2 
    
    # define an input history series and pass it through a stack of dilated causal convolution blocks. 
    history_seq = Input(shape=(model_parameter.seq_length, model_parameter.data_dim))
    x = history_seq
    
    # 增加正则化层-2020-03-20
    x = tf.keras.layers.BatchNormalization()(x)
    
    skips = []
    for dilation_rate in dilation_rates:
        
        # preprocessing - equivalent to time-distributed dense
        # x = Conv1D(16, 1, padding='same')(x) data_dim
        # x = Conv1D(model_parameter.seq_length * 2, 1, padding='same')(x)
        x = Conv1D(model_parameter.data_dim * 4, 1, padding='same')(x)
 
        
        # filter convolution
        x_f = Conv1D(filters=n_filters,
                    kernel_size=filter_width, 
                    padding='causal',
                    dilation_rate=dilation_rate)(x)
        
        # gating convolution
        x_g = Conv1D(filters=n_filters,
                    kernel_size=filter_width, 
                    padding='causal',
                    dilation_rate=dilation_rate)(x)
        
        # multiply filter and gating branches
        # z = Multiply()([Activation('tanh')(x_f),
        #                Activation('sigmoid')(x_g)])
        z = Multiply()([Activation('tanh')(x_f),
                        Activation('relu')(x_g)])
        
        # postprocessing - equivalent to time-distributed dense
        # z = Conv1D(16, 1, padding='same')(z)
        # z = Conv1D(model_parameter.seq_length * 2, 1, padding='same')(z)
        z = Conv1D(model_parameter.data_dim * 4, 1, padding='same')(z)
        
        
        # residual connection
        x = Add()([x, z])    
        
        # collect skip connections
        skips.append(z)

    # add all skip connection outputs 
    # out = Activation('relu')(Add()(skips))
    out = Add()(skips)

    # final time-distributed dense layers 
    # out = Conv1D(128, 1, padding='same')(out)
    out = Conv1D(model_parameter.hidden_dim * 5, 1, padding='same')(out)
    # out = Activation('relu')(out)
    out = Dropout(model_parameter.dropout_rate)(out)
    out = Conv1D(1, 1, padding='same')(out)
    out = GlobalAveragePooling1D()(out)
    pred_seq_train = layers.Dense(model_parameter.output_dim)(out)

    # extract the last 60 time steps as the training target
    # def slice(x, seq_length):
    #     return x[:,-seq_length:,:]

    # pred_seq_train = Lambda(slice, arguments={'seq_length':model_parameter.seq_length})(out)

    model = Model(history_seq, pred_seq_train)
    
    optimizer = tf.keras.optimizers.Adam(model_parameter.learning_rate)
    # optimizer = tf.keras.optimizers.RMSprop(model_parameter.learning_rate)
    # loss = 'mse', 'mean_squared_error', 'huber_loss'
    my_loss = 'mse'
    model.compile(optimizer=optimizer,
                  loss = my_loss,
                  metrics=['mae', 'mse'])
    return model


# wavenet_atten_model模型
def wavenet_atten_model(model_parameter):
    # convolutional operation parameters
    n_filters = model_parameter.hidden_dim # 32 
    filter_width = 2
    # dilation_rates = [2**i for i in range(8)] * 2   # 
    # dilation_rates = [2**i for i in range(model_parameter.seq_length)] * 2
    dilation_rates = [2**i for i in range(model_parameter.seq_length)] * 2 
    
    # define an input history series and pass it through a stack of dilated causal convolution blocks. 
    history_seq = Input(shape=(model_parameter.seq_length, model_parameter.data_dim))
    x = history_seq
    
    # 增加正则化层-2020-03-20
    x = tf.keras.layers.BatchNormalization()(x)
    
    skips = []
    for dilation_rate in dilation_rates:
        
        # preprocessing - equivalent to time-distributed dense
        # x = Conv1D(16, 1, padding='same')(x) data_dim
        # x = Conv1D(model_parameter.seq_length * 2, 1, padding='same')(x)
        x = Conv1D(model_parameter.data_dim * 4, 1, padding='same')(x)
 
        
        # filter convolution
        x_f = Conv1D(filters=n_filters,
                    kernel_size=filter_width, 
                    padding='causal',
                    dilation_rate=dilation_rate)(x)
        
        # gating convolution
        x_g = Conv1D(filters=n_filters,
                    kernel_size=filter_width, 
                    padding='causal',
                    dilation_rate=dilation_rate)(x)
        
        # multiply filter and gating branches
        # z = Multiply()([Activation('tanh')(x_f),
        #                Activation('sigmoid')(x_g)])
        z = Multiply()([Activation('tanh')(x_f),
                        Activation('relu')(x_g)])
        
        # postprocessing - equivalent to time-distributed dense
        # z = Conv1D(16, 1, padding='same')(z)
        # z = Conv1D(model_parameter.seq_length * 2, 1, padding='same')(z)
        z = Conv1D(model_parameter.data_dim * 4, 1, padding='same')(z)
        
        
        # residual connection
        x = Add()([x, z])

        attention_layer = AttentionLayer()(z)
        # print(attention_layer)

        repeated_word_attention = tf.keras.layers.RepeatVector(model_parameter.hidden_dim * 4)(attention_layer)
        repeated_word_attention = tf.keras.layers.Permute([2, 1])(repeated_word_attention)
        sentence_representation = tf.keras.layers.Multiply()([z, repeated_word_attention])
    
        # sentence_representation = tf.keras.layers.Lambda(lambda x: Kbackend.sum(x, axis=1))(sentence_representation)
        # skips.append(sentence_representation) 
        z = tf.keras.layers.Lambda(lambda x: Kbackend.sum(x, axis=1))(sentence_representation)
        # collect skip connections
        skips.append(z)

    # add all skip connection outputs 
    out = Activation('relu')(Concatenate()(skips))
    # out = Add()(skips)

    # final time-distributed dense layers 
    # out = Conv1D(128, 1, padding='same')(out)
    out = Conv1D(model_parameter.hidden_dim * 5, 1, padding='same')(out)
    # out = Activation('relu')(out)
    out = Dropout(model_parameter.dropout_rate)(out)
    out = Conv1D(1, 1, padding='same')(out)
    out = GlobalAveragePooling1D()(out)
    pred_seq_train = layers.Dense(model_parameter.output_dim)(out)

    # extract the last 60 time steps as the training target
    # def slice(x, seq_length):
    #     return x[:,-seq_length:,:]

    # pred_seq_train = Lambda(slice, arguments={'seq_length':model_parameter.seq_length})(out)

    model = Model(history_seq, pred_seq_train)
    
    optimizer = tf.keras.optimizers.Adam(model_parameter.learning_rate)
    # optimizer = tf.keras.optimizers.RMSprop(model_parameter.learning_rate)
    # loss = 'mse', 'mean_squared_error', 'huber_loss'
    my_loss = 'mse'
    model.compile(optimizer=optimizer,
                  loss = my_loss,
                  metrics=['mae', 'mse'])
    return model



# https://www.evergreeninnovations.co/blog-quantile-loss-function-for-machine-learning/
def quantile_loss(q, y, y_p):
    e = y-y_p
    return tf.keras.backend.mean(tf.keras.backend.maximum(q*e, (q-1)*e))

'''
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(100, activation='relu', input_dim=1))
model.add(tf.keras.layers.Dense(100, activation='relu'))
model.add(tf.keras.layers.Dense(100, activation='relu'))
model.add(tf.keras.layers.Dense(1, activation='linear'))
# The lambda function is used to input the quantile value to the quantile
# regression loss function. Keras only allows two inputs in user-defined loss
# functions, actual and predicted values.
quantile = 0.977
model.compile(optimizer='adam', loss=lambda y, y_p: quantile_loss(quantile, y, y_p))
model.fit(x_train, y_train, epochs=20)
prediction = model.predict(x_test)
'''

# Double_Expert_Net
def double_experts_Net_model(model_parameter):
    ##double_ experts_Net 
    # 杈撳叆鏁版嵁
    input_data = tf.keras.layers.Input(shape=(model_parameter.seq_length, model_parameter.data_dim))
    input_data_G = tf.keras.layers.BatchNormalization()(input_data)
    ##############涓擄�??锟斤�??1############################
    bilstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(model_parameter.hidden_dim, return_sequences=True))(
        input_data_G)
    bilstm_output = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(model_parameter.hidden_dim, return_sequences=False))(bilstm)
    expert_1_output = tf.keras.layers.Dense(model_parameter.output_dim)(bilstm_output)

    ##############涓擄�??锟斤�??2############################
    n_filters = model_parameter.hidden_dim  # 闅愯棌灞傦拷?锟界粡鍏冩暟锟�?
    filter_width = 2
    dilation_rates = [2 ** i for i in range(model_parameter.seq_length)]

    x = input_data_G  # 杈撳叆鏁版嵁姝ｅ�?鍖栧悗鐨�?粨锟�??
    x = tf.keras.layers.BatchNormalization()(x)

    skips = []
    for dilation_rate in dilation_rates:
        # preprocessing - equivalent to time-distributed dense
        x = tf.keras.layers.Conv1D(model_parameter.seq_length, 1, padding='same', activation='relu')(x)
        # filter convolution
        x_f = tf.keras.layers.Conv1D(filters=n_filters,
                                     kernel_size=filter_width,
                                     padding='causal',
                                     dilation_rate=dilation_rate)(x)
        # gating convolution
        x_g = tf.keras.layers.Conv1D(filters=n_filters,
                                     kernel_size=filter_width,
                                     padding='causal',
                                     dilation_rate=dilation_rate)(x)
        # multiply filter and gating branches鈥斺�?斿浘4锟�?宸︿笂瑙掔殑鍚堝苟鎿嶄綔
        z = tf.keras.layers.Multiply()([tf.keras.layers.Activation('tanh')(x_f),
                                        tf.keras.layers.Activation('sigmoid')(x_g)])
        # postprocessing - equivalent to time-distributed dense
        z = tf.keras.layers.Conv1D(model_parameter.seq_length, 1, padding='same', activation='relu')(z)
        # residual connection鈥斺�?旀�?锟�?鏈哄�?
        x = tf.keras.layers.Add()([x, z])

        # collect skip connections鈥斺�?旇烦灞傝繛锟�??
        skips.append(z)

    # add all skip connection outputs 
    out = tf.keras.layers.Activation('relu')(tf.keras.layers.Add()(skips))

    # final time-distributed dense layers 
    out = tf.keras.layers.Conv1D(model_parameter.hidden_dim, 1, padding='same')(out)
    out = tf.keras.layers.Activation('relu')(out)
    out = tf.keras.layers.Conv1D(1, 1, padding='same')(out)
    out = tf.keras.layers.GlobalAveragePooling1D()(out)
    expert_2_output = tf.keras.layers.Dense(model_parameter.output_dim)(out)

    ##############final_output##############################
    # double_experts =  tf.keras.layers.add([expert_1_output, expert_2_output])
    # final_output = double_ experts * 0.5  # 鏀规垚涓�??潰鐨�?綉缁滃疄锟�?

    double_experts = tf.keras.layers.Concatenate()([expert_1_output, expert_2_output])
    final_output = tf.keras.layers.Dense(model_parameter.output_dim)(double_experts)
        # optimizer = tf.keras.optimizers.RMSprop(model_parameter.learning_rate)
    # loss = 'mse', 'mean_squared_error', 'huber_loss'
    # huber_loss:
    huberLoss = tf.keras.losses.Huber(delta=1,
                # reduction=tf.keras.losses.Reduction.SUM,
                reduction=tf.keras.losses.Reduction.AUTO,
                name='huber_loss')
    #  quantileLoss  # 
    quantile = 0.5
    quantileLoss=lambda y, y_p: quantile_loss(quantile, y, y_p)
    # tf.keras.losses.log_cosh
    LogCosh_Loss = tf.keras.losses.LogCosh(
                   reduction=tf.keras.losses.Reduction.AUTO,
                   name='log_cosh')
    my_optimizer = tf.keras.optimizers.Adam(model_parameter.learning_rate)
    model = tf.keras.models.Model(inputs=[input_data],
                                  outputs=[final_output])
    model.compile(loss=quantileLoss, optimizer=my_optimizer, metrics=['mae',  'mse'])

    return model




class ScaledDotProductAttention(Layer):
    r"""The attention layer that takes three inputs representing queries, keys and values.

    \text{Attention}(Q, K, V) = \text{softmax}(\frac{Q K^T}{\sqrt{d_k}}) V

    See: https://arxiv.org/pdf/1706.03762.pdf
    """

    def __init__(self,
                 return_attention=False,
                 history_only=False,
                 **kwargs):
        """Initialize the layer.

        :param return_attention: Whether to return attention weights.
        :param history_only: Whether to only use history data.
        :param kwargs: Arguments for parent class.
        """
        super(ScaledDotProductAttention, self).__init__(**kwargs)
        self.supports_masking = True
        self.return_attention = return_attention
        self.history_only = history_only
        self.intensity = self.attention = None

    def get_config(self):
        config = {
            'return_attention': self.return_attention,
            'history_only': self.history_only,
        }
        base_config = super(ScaledDotProductAttention, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        if isinstance(input_shape, list):
            query_shape, key_shape, value_shape = input_shape
        else:
            query_shape = key_shape = value_shape = input_shape
        output_shape = query_shape[:-1] + value_shape[-1:]
        if self.return_attention:
            attention_shape = query_shape[:2] + (key_shape[1],)
            return [output_shape, attention_shape]
        return output_shape

    def compute_mask(self, inputs, mask=None):
        if isinstance(mask, list):
            mask = mask[0]
        if self.return_attention:
            return [mask, None]
        return mask

    def call(self, inputs, mask=None, **kwargs):
        if isinstance(inputs, list):
            query, key, value = inputs
        else:
            query = key = value = inputs
        if isinstance(mask, list):
            mask = mask[1]
        feature_dim = Kbackend.shape(query)[-1]
        e = Kbackend.batch_dot(query, key, axes=2) / Kbackend.sqrt(Kbackend.cast(feature_dim, dtype=Kbackend.floatx()))
        if self.history_only:
            query_len, key_len = Kbackend.shape(query)[1], Kbackend.shape(key)[1]
            indices = Kbackend.expand_dims(Kbackend.arange(0, key_len), axis=0)
            upper = Kbackend.expand_dims(Kbackend.arange(0, query_len), axis=-1)
            e -= 10000.0 * Kbackend.expand_dims(Kbackend.cast(indices > upper, Kbackend.floatx()), axis=0)
        if mask is not None:
            e -= 10000.0 * (1.0 - Kbackend.cast(Kbackend.expand_dims(mask, axis=-2), Kbackend.floatx()))
        self.intensity = e
        e = Kbackend.exp(e - Kbackend.max(e, axis=-1, keepdims=True))
        self.attention = e / Kbackend.sum(e, axis=-1, keepdims=True)
        v = Kbackend.batch_dot(self.attention, value)
        if self.return_attention:
            return [v, self.attention]
        return v



class MultiHeadAttention(tf.keras.layers.Layer):
    """Multi-head attention layer.

    See: https://arxiv.org/pdf/1706.03762.pdf
    """

    def __init__(self,
                 head_num,
                 activation='relu',
                 use_bias=True,
                 kernel_initializer=tf.keras.initializers.GlorotUniform(),
                 bias_initializer=tf.keras.initializers.Zeros(),
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 history_only=False,
                 **kwargs):
        """Initialize the layer.

        :param head_num: Number of heads.
        :param activation: Activations for linear mappings.
        :param use_bias: Whether to use bias term.
        :param kernel_initializer: Initializer for linear mappings.
        :param bias_initializer: Initializer for linear mappings.
        :param kernel_regularizer: Regularizer for linear mappings.
        :param bias_regularizer: Regularizer for linear mappings.
        :param kernel_constraint: Constraints for linear mappings.
        :param bias_constraint: Constraints for linear mappings.
        :param history_only: Whether to only use history in attention layer.
        """
        self.supports_masking = True
        self.head_num = head_num
        self.activation = keras.activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.bias_initializer = keras.initializers.get(bias_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.bias_regularizer = keras.regularizers.get(bias_regularizer)
        self.kernel_constraint = keras.constraints.get(kernel_constraint)
        self.bias_constraint = keras.constraints.get(bias_constraint)
        self.history_only = history_only

        self.Wq = self.Wk = self.Wv = self.Wo = None
        self.bq = self.bk = self.bv = self.bo = None

        self.intensity = self.attention = None
        super(MultiHeadAttention, self).__init__(**kwargs)

    def get_config(self):
        config = {
            'head_num': self.head_num,
            'activation': keras.activations.serialize(self.activation),
            'use_bias': self.use_bias,
            'kernel_initializer': keras.initializers.serialize(self.kernel_initializer),
            'bias_initializer': keras.initializers.serialize(self.bias_initializer),
            'kernel_regularizer': keras.regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': keras.regularizers.serialize(self.bias_regularizer),
            'kernel_constraint': keras.constraints.serialize(self.kernel_constraint),
            'bias_constraint': keras.constraints.serialize(self.bias_constraint),
            'history_only': self.history_only,
        }
        base_config = super(MultiHeadAttention, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        if isinstance(input_shape, list):
            q, k, v = input_shape
            return q[:-1] + (v[-1],)
        return input_shape

    def compute_mask(self, inputs, input_mask=None):
        if isinstance(input_mask, list):
            return input_mask[0]
        return input_mask

    def build(self, input_shape):
        if isinstance(input_shape, list):
            q, k, v = input_shape
        else:
            q = k = v = input_shape
        feature_dim = int(v[-1])
        if feature_dim % self.head_num != 0:
            raise IndexError('Invalid head number %d with the given input dim %d' % (self.head_num, feature_dim))
        self.Wq = self.add_weight(
            shape=(int(q[-1]), feature_dim),
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            name='%s_Wq' % self.name,
        )
        if self.use_bias:
            self.bq = self.add_weight(
                shape=(feature_dim,),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                name='%s_bq' % self.name,
            )
        self.Wk = self.add_weight(
            shape=(int(k[-1]), feature_dim),
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            name='%s_Wk' % self.name,
        )
        if self.use_bias:
            self.bk = self.add_weight(
                shape=(feature_dim,),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                name='%s_bk' % self.name,
            )
        self.Wv = self.add_weight(
            shape=(int(v[-1]), feature_dim),
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            name='%s_Wv' % self.name,
        )
        if self.use_bias:
            self.bv = self.add_weight(
                shape=(feature_dim,),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                name='%s_bv' % self.name,
            )
        self.Wo = self.add_weight(
            shape=(feature_dim, feature_dim),
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            name='%s_Wo' % self.name,
        )
        if self.use_bias:
            self.bo = self.add_weight(
                shape=(feature_dim,),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                name='%s_bo' % self.name,
            )
        super(MultiHeadAttention, self).build(input_shape)

    @staticmethod
    def _reshape_to_batches(x, head_num):
        input_shape = Kbackend.shape(x)
        batch_size, seq_len, feature_dim = input_shape[0], input_shape[1], input_shape[2]
        head_dim = feature_dim // head_num
        x = Kbackend.reshape(x, (batch_size, seq_len, head_num, head_dim))
        x = Kbackend.permute_dimensions(x, [0, 2, 1, 3])
        return Kbackend.reshape(x, (batch_size * head_num, seq_len, head_dim))

    @staticmethod
    def _reshape_attention_from_batches(x, head_num):
        input_shape = Kbackend.shape(x)
        batch_size, seq_len, feature_dim = input_shape[0], input_shape[1], input_shape[2]
        x = Kbackend.reshape(x, (batch_size // head_num, head_num, seq_len, feature_dim))
        return Kbackend.permute_dimensions(x, [0, 2, 1, 3])

    @staticmethod
    def _reshape_from_batches(x, head_num):
        input_shape = Kbackend.shape(x)
        batch_size, seq_len, feature_dim = input_shape[0], input_shape[1], input_shape[2]
        x = Kbackend.reshape(x, (batch_size // head_num, head_num, seq_len, feature_dim))
        x = Kbackend.permute_dimensions(x, [0, 2, 1, 3])
        return Kbackend.reshape(x, (batch_size // head_num, seq_len, feature_dim * head_num))

    @staticmethod
    def _reshape_mask(mask, head_num):
        if mask is None:
            return mask
        seq_len = Kbackend.shape(mask)[1]
        mask = Kbackend.expand_dims(mask, axis=1)
        mask = Kbackend.tile(mask, [1, head_num, 1])
        return Kbackend.reshape(mask, (-1, seq_len))

    def call(self, inputs, mask=None):
        if isinstance(inputs, list):
            q, k, v = inputs
        else:
            q = k = v = inputs
        if isinstance(mask, list):
            q_mask, k_mask, v_mask = mask
        else:
            q_mask = k_mask = v_mask = mask
        q = Kbackend.dot(q, self.Wq)
        k = Kbackend.dot(k, self.Wk)
        v = Kbackend.dot(v, self.Wv)
        if self.use_bias:
            q += self.bq
            k += self.bk
            v += self.bv
        if self.activation is not None:
            q = self.activation(q)
            k = self.activation(k)
            v = self.activation(v)
        scaled_dot_product_attention = ScaledDotProductAttention(
            history_only=self.history_only,
            name='%s-Attention' % self.name,
        )
        y = scaled_dot_product_attention(
            inputs=[
                self._reshape_to_batches(q, self.head_num),
                self._reshape_to_batches(k, self.head_num),
                self._reshape_to_batches(v, self.head_num),
            ],
            mask=[
                self._reshape_mask(q_mask, self.head_num),
                self._reshape_mask(k_mask, self.head_num),
                self._reshape_mask(v_mask, self.head_num),
            ],
        )
        self.intensity = self._reshape_attention_from_batches(scaled_dot_product_attention.intensity, self.head_num)
        self.attention = self._reshape_attention_from_batches(scaled_dot_product_attention.attention, self.head_num)
        y = self._reshape_from_batches(y, self.head_num)
        y = Kbackend.dot(y, self.Wo)
        if self.use_bias:
            y += self.bo
        if self.activation is not None:
            y = self.activation(y)
        # if TF_KERAS:
            # Add shape information to tensor when using `tf.keras`
        input_shape = [Kbackend.int_shape(q), Kbackend.int_shape(k), Kbackend.int_shape(v)]
        output_shape = self.compute_output_shape(input_shape)
        if output_shape[1] is not None:
            output_shape = (-1,) + output_shape[1:]
            y = Kbackend.reshape(y, output_shape)
        return y



class Self_Attention_layer(tf.keras.layers.Layer):
    """
        Attention operation, with a context/query vector, for temporal data.
        Supports Masking.
        Follows the work of Yang et al. [https://www.cs.cmu.edu/~diyiy/docs/naacl16.pdf]
        "Hierarchical Attention Networks for Document Classification"
        by using a context vector to assist the attention
        # Input shape
            3D tensor with shape: `(samples, steps, features)`.
        # Output shape
            2D tensor with shape: `(samples, features)`.
        :param kwargs:
        Just put it on top of an RNN Layer (GRU/LSTM/SimpleRNN) with return_sequences=True.
        The dimensions are inferred based on the output shape of the RNN.
        Example:
            model.add(LSTM(64, return_sequences=True))
            model.add(AttentionWithContext())
        """

    def __init__(self,
                 W_regularizer=None, b_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, supports_masking=True, initializer=None, initializer_bias=None, **kwargs):

        self.supports_masking = True
        # self.initializer = initializers.get('glorot_uniform')
        self.initializer = tf.keras.initializers.GlorotUniform()
        # self.initializer_bias = initializers.get('zero')
        self.initializer_bias = tf.keras.initializers.Zeros()

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        super(Self_Attention_layer, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight(name='{}_W'.format(self.name),
                                 shape=(input_shape[-1], input_shape[-1]),
                                 initializer=self.initializer,
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        if self.bias:
            self.b = self.add_weight(name='{}_b'.format(self.name),
                                     shape=(input_shape[-1],),
                                     initializer=self.initializer_bias,
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)

        super(Self_Attention_layer, self).build(input_shape)

    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def call(self, x, mask=None):
        uit = Kbackend.dot(x, self.W)

        if self.bias:
            uit += self.b

        uit = Kbackend.tanh(uit)

        a = Kbackend.exp(uit)

        # apply mask after the exp. will be re-normalized next
        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting in theano
            a *= Kbackend.cast(mask, Kbackend.floatx())

        # in some cases especially in the early stages of training the sum may be almost zero
        # and this results in NaN's. A workaround is to add a very small positive number to the sum.
        # a /= K.cast(K.sum(a, axis=1, keepdims=True), K.floatx())
        a /= Kbackend.cast(Kbackend.sum(a, axis=1, keepdims=True) + Kbackend.epsilon(), Kbackend.floatx())
        # print(a)
        # a = K.expand_dims(a)
        # print(x)
        weighted_input = x * a
        # print(weighted_input)
        return Kbackend.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1])

    # https://zhuanlan.zhihu.com/p/86886620
    # ????????????????????????config??????��?????????????????????????��?????????????????????��?????get_config
    # ??????base_config??????????????????????????��???��??????????��???????????????????????��?????????config????????????????????????????��???????��??��?????
    def get_config(self):
        config = {
            "supports_masking": self.supports_masking,
            "initializer": self.initializer,
            "initializer_bias": self.initializer_bias,
            "W_regularizer": self.W_regularizer,
            "b_regularizer": self.b_regularizer,
            "W_constraint": self.W_constraint,
            "b_constraint": self.b_constraint,
            "bias": self.bias,
        }
        base_config = super(Self_Attention_layer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))



# Bigru_Multihead_Self_Atten_DNN
def bigru_multihead_atten_dnn_model(model_parameter):
    input_data = tf.keras.layers.Input(shape=(model_parameter.seq_length,model_parameter.data_dim))
    input_data_G = input_data
    bigru = input_data_G  
    skips = []

    for i in range(model_parameter.n_layers):
        bigru = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(model_parameter.hidden_dim,return_sequences=True))(bigru)
        bigru = tf.keras.layers.Dropout(model_parameter.dropout_rate)(bigru)
        multihead_atten_layer =  MultiHeadAttention(head_num=model_parameter.hidden_dim * 2,name='Multi-Head' + str(i),)(bigru)
        # print(attention_layer)        
        skips.append(multihead_atten_layer)
    
    # out_block = Activation('relu')(Add()(skips))
#     for j in range(len(skips)):
#         out_block = tf.keras.layers.Concatenate()([skips[j],x])
    out_block = Activation('relu')(tf.keras.layers.Concatenate()(skips))
    self_atten_layer = Self_Attention_layer(name='SelfAttenLayer')(out_block)
    dnnlayer_output = tf.keras.layers.Dense(model_parameter.hidden_dim * 10,  name='fianl_class_input')(self_atten_layer)            
    bigru_output = tf.keras.layers.Dense(model_parameter.output_dim, activation='softmax', name='fianl_class_output')(dnnlayer_output)

    my_optimizer = tf.keras.optimizers.Adam(model_parameter.learning_rate)

    model = tf.keras.models.Model(inputs=[input_data],
                outputs=[bigru_output])
    # loss = 'mse', 'mean_squared_error', 'huber_loss'
    # huber_loss:
    huberLoss = tf.keras.losses.Huber(delta=1,
                # reduction=tf.keras.losses.Reduction.SUM,
                reduction=tf.keras.losses.Reduction.AUTO,
                name='huber_loss')
    #  quantileLoss  # 
    quantile = 0.5
    quantileLoss=lambda y, y_p: quantile_loss(quantile, y, y_p)
    # tf.keras.losses.log_cosh
    LogCosh_Loss = tf.keras.losses.LogCosh(
                   reduction=tf.keras.losses.Reduction.AUTO,
                   name='log_cosh')
    model.compile(loss='mse', optimizer=my_optimizer,metrics=['mse','mae'])
    # model.compile(loss=FocalLoss(alpha=1), optimizer = my_optimizer,metrics=['accuracy'])

    return model



# u-net改�?
def my_unet(model_parameter):
    data_format = "channels_last"   # "channels_first"  | "channels_last"
    # inputs = Input((3, img_w, img_h))
    inputs = Input(shape=(model_parameter.seq_length, model_parameter.data_dim))
    x = inputs
    depth = model_parameter.n_layers
    n_filters = model_parameter.data_dim * 2 # model_parameter.hidden_dim  # 32 
    filter_width = 3
    features = n_filters
    skips = []
    # 增加正则化层-2020-03-20
    x = tf.keras.layers.BatchNormalization()(x)
    for i in range(depth):
        # print("�?",i,"�?")
        x =  Conv1D(filters=features,
                kernel_size=filter_width,
                   padding='same', data_format=data_format)(x)
        # print(x.shape)
        x = Dropout(model_parameter.dropout_rate)(x)
        x =  Conv1D(filters=features,
                kernel_size=filter_width,
                   padding='same', data_format=data_format)(x)
        # print(x.shape)
        skips.append(x)
        x = MaxPool1D(pool_size = 1, strides=None,  padding='same', data_format=data_format)(x)
        features = features * 2
    # print("过渡")
    x =  Conv1D(filters = features,
                kernel_size = filter_width,
                   padding = 'same', data_format=data_format)(x)
    # print(x.shape)
    x = Dropout(model_parameter.dropout_rate)(x)
    x =  Conv1D(filters = features,
                kernel_size = filter_width,
                   padding = 'same', data_format=data_format)(x)
    # print(x.shape)
    # print("过渡")
    # print("reversed")
    for i in reversed(range(depth)):
        # print("�?",i,"�?")
        features = features // 2
        # attention_up_and_concate(x,[skips[i])
        # #######不做上采�?
        x = UpSampling1D(size=1)(x)
        # print(x.shape)
        # print(skips[i].shape)
        x = tf.keras.layers.Concatenate()([skips[i],x])
        # 与上述操作效果一�?
        # x = concatenate([skips[i],x],axis=2)
        
        # 注意concatenate区别于residual connection �? Add
        #x = Add()([x, z])    

        x =  Conv1D(filters = features,
                kernel_size = filter_width,
                   padding = 'same', data_format=data_format)(x)
        x = Dropout(model_parameter.dropout_rate)(x)
        # x = Conv2D(features, (3, 3), activation='relu', padding='same', data_format=data_format)(x)
        x =  Conv1D(filters = features,
                kernel_size = filter_width,
                   padding = 'same', data_format=data_format)(x)
        
    
    conv6 = Conv1D(1, 1, padding='same', data_format=data_format)(x)
    out = Dropout(model_parameter.dropout_rate)(conv6)
    
    # conv7 = core.Activation('sigmoid')(conv6)
    # conv7 = GlobalAveragePooling1D()(conv6)
    # conv7 = Activation('relu')(conv6)
    # out = Dropout(model_parameter.dropout_rate)(conv7)
    # out = Dropout(model_parameter.dropout_rate)(conv6)
    out = GlobalAveragePooling1D()(out)
    pred = layers.Dense(model_parameter.output_dim)(out)
    
    model = Model(inputs=inputs, outputs=pred)
    
    optimizer = tf.keras.optimizers.Adam(model_parameter.learning_rate)
    # optimizer = tf.keras.optimizers.RMSprop(model_parameter.learning_rate)
    # loss = 'mse', 'mean_squared_error', 'huber_loss'
    my_loss = 'mse'
    model.compile(optimizer=optimizer,
                  loss = my_loss,
                  metrics=['mae', 'mse'])
    #model.compile(optimizer=Adam(lr=1e-5), loss=[focal_loss()], metrics=['accuracy', dice_coef])
    return model




# def rbf_model(X,model_parameter):
def rbf_model(model_parameter):
      # creating RBF network-径向基函数�?�经网络(Radical Basis Function)
    model = tf.keras.Sequential()
    # rbflayer = RBFLayer(model_parameter.hidden_dim, initializer=InitCentersRandom(X), betas=2.0,
    #                   input_shape=(model_parameter.data_dim,))
    model.add(tf.keras.layers.BatchNormalization())
    rbflayer = RBFLayer(model_parameter.hidden_dim, initializer=tf.keras.initializers.GlorotUniform(), betas=2.0,
                      input_shape=(model_parameter.data_dim,))
#     rbflayer = RBFLayer(10,initializer=InitCentersKMeans(X), betas=2.0,
#                       input_shape=(num_inputs,))
    model.add(rbflayer)
    ## 有待商榷
    # model.add(layers.Dropout(model_parameter.dropout_rate))
    model.add(layers.Dense(model_parameter.output_dim, use_bias=False, activation='relu'))
    optimizer = tf.keras.optimizers.Adam(model_parameter.learning_rate)
    model.compile(optimizer=optimizer,
                  loss='mse',
                  metrics=['mae', 'mse'])
    return model








# --------------------------------下面为模型依赖类--------------------------------------------------


class AttentionLayer(Layer):
    """
    Attention operation, with a context/query vector, for temporal data.
    Supports Masking.
    Follows the work of Yang et al. [https://www.cs.cmu.edu/~diyiy/docs/naacl16.pdf]
    "Hierarchical Attention Networks for Document Classification"
    by using a context vector to assist the attention
    # Input shape
        3D tensor with shape: `(samples, steps, features)`.
    # Output shape
        2D tensor with shape: `(samples, features)`.
    How to use:
    Just put it on top of an RNN Layer (GRU/LSTM/SimpleRNN) with return_sequences=True.
    The dimensions are inferred based on the output shape of the RNN.
    Note: The layer has been tested with Keras 2.0.6
    Example:
        model.add(LSTM(64, return_sequences=True))
        model.add(AttentionWithContext())
        # next add a Dense layer (for classification/regression) or whatever...
    """
    def __init__(self, **kwargs):
        self.init = initializers.get('glorot_uniform')
        super(AttentionLayer, self).__init__(**kwargs)
    
    def build(self, input_shape):
        assert len(input_shape) == 3
        
        self.W = self.add_weight(name='Attention_Weight',
                                 shape=(input_shape[-1], input_shape[-1]),
                                 initializer=self.init,
                                 trainable=True)
        self.b = self.add_weight(name='Attention_Bias',
                                 shape=(input_shape[-1], ),
                                 initializer=self.init,
                                 trainable=True)
        self.u = self.add_weight(name='Attention_Context_Vector',
                                 shape=(input_shape[-1], 1),
                                 initializer=self.init,
                                 trainable=True)
        super(AttentionLayer, self).build(input_shape)

    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None
    
    def call(self, x):
        # refer to the original paper
        # link: https://www.cs.cmu.edu/~hovy/papers/16HLT-hierarchical-attention-networks.pdf
        
        # RNN �?조를 거쳐�? 나온 hidden states (x)�? single layer perceptron (tanh activation)
        # 적용하여 나온 벡터가 uit 
        u_it = Kbackend.tanh(Kbackend.dot(x, self.W) + self.b)
        
        # uit와 uw (혹은 us) 간의 similarity�? attention으�?? �?�?
        # softmax�? 통해 attention 값을 확�?? 분포�? 만듬
        a_it = Kbackend.dot(u_it, self.u)
        a_it = Kbackend.squeeze(a_it, -1)
        a_it = Kbackend.softmax(a_it)
        
        return a_it
        
    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1])




# or using KMeans clustering for RBF centers
class InitCentersKMeans(Initializer):
    """ Initializer for initialization of centers of RBF network
        by clustering the given data set.
    # Arguments
        X: matrix, dataset
    """

    def __init__(self, X, max_iter=100):
        self.X = X
        self.max_iter = max_iter
        super().__init__()

    def __call__(self, shape, dtype=None):
        assert shape[1:] == self.X.shape[1:]

        n_centers = shape[0]
        km = KMeans(n_clusters=n_centers, max_iter=self.max_iter, verbose=0)
        km.fit(self.X)
        return km.cluster_centers_


class InitCentersRandom(Initializer):
    """ Initializer for initialization of centers of RBF network
        as random samples from the given data set.

    # Arguments
        X: matrix, dataset to choose the centers from (random rows
          are taken as centers)
    """

    def __init__(self, X):
        self.X = X
        super().__init__()

    def __call__(self, shape, dtype=None):
        assert shape[1:] == self.X.shape[1:]  # check dimension

        # np.random.randint returns ints from [low, high) !
        idx = np.random.randint(self.X.shape[0], size=shape[0])

        return self.X[idx, :]


class RBFLayer(Layer):
    """ Layer of Gaussian RBF units.

    # Example

    ```python
        model = Sequential()
        model.add(RBFLayer(10,
                           initializer=InitCentersRandom(X),
                           betas=1.0,
                           input_shape=(1,)))
        model.add(Dense(1))
    ```


    # Arguments
        output_dim: number of hidden units (i.e. number of outputs of the
                    layer)
        initializer: instance of initiliazer to initialize centers
        betas: float, initial value for betas

    """

    def __init__(self, output_dim, initializer=None, betas=1.0, **kwargs):

        self.output_dim = output_dim

        # betas is either initializer object or float
        if isinstance(betas, Initializer):
            self.betas_initializer = betas
        else:
            self.betas_initializer = Constant(value=betas)

        self.initializer = initializer if initializer else RandomUniform(
            0.0, 1.0)

        super().__init__(**kwargs)

    def build(self, input_shape):

        self.centers = self.add_weight(name='centers',
                                       shape=(self.output_dim, input_shape[1]),
                                       initializer=self.initializer,
                                       trainable=True)
        self.betas = self.add_weight(name='betas',
                                     shape=(self.output_dim,),
                                     initializer=self.betas_initializer,
                                     # initializer='ones',
                                     trainable=True)

        super().build(input_shape)

    def call(self, x):

        C = tf.expand_dims(self.centers, -1)  # inserts a dimension of 1
        H = tf.transpose(C-tf.transpose(x))  # matrix of differences
        return tf.exp(-self.betas * tf.math.reduce_sum(H**2, axis=1))

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)

    def get_config(self):
        # have to define get_config to be able to use model_from_json
        config = {
            'output_dim': self.output_dim
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))
   

'''rbfnet = load_model("some_fency_file_name.h5", custom_objects={'RBFLayer': RBFLayer})'''




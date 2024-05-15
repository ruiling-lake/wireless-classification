import h5py
import numpy as np
import os,random
from keras.layers import Input,Reshape,ZeroPadding2D,Conv2D,Dropout,Flatten,Dense,Activation,MaxPooling2D,AlphaDropout
from keras import layers
import keras.models as Model
from keras.regularizers import *
from keras.optimizers import adam
import seaborn as sns
import keras
import tensorflow as tf
import matplotlib.pyplot as plt
# %matplotlib inline
os.environ["KERAS_BACKEND"] = "tensorflow"

f = h5py.File('E:/pycharm file/fedml_Project/ResNet_Model_60w.wts.h5')
sample_num = f['X'].shape[0]
idx = np.random.choice(range(0,sample_num),size=30000)
X = f['X'][:][idx]
Y = f['Y'][:][idx]
Z = f['Z'][:][idx]
f.close()

for i in range(1,11):
    # if i%2 != 0:
        # free -m
    '''if i == 10:
        continue'''
    # filename = 'drive/RadioModulationRecognition/Data_dir/part'+str(i) + '.h5'
    filename = 'E:/pycharm file/fedml_Project'+str(i) + '.h5'
    print(filename)
    f = h5py.File(filename,'r')
    X = np.vstack((X,f['X'][:][idx]))
    Y = np.vstack((Y,f['Y'][:][idx]))
    Z = np.vstack((Z,f['Z'][:][idx]))
    f.close()


print('X维度：',X.shape)
print('Y维度：',Y.shape)
print('Z维度：',Z.shape)

"""数据预处理，并获取训练集和测试集"""
n_examples = X.shape[0]
n_train = int(n_examples * 0.7)   #70%训练样本
train_idx = np.random.choice(range(0,n_examples), size=n_train, replace=False)  #随机选取训练样本下标
test_idx = list(set(range(0,n_examples))-set(train_idx)) #测试样本下标
X_train = X[train_idx]  #训练样本
X_test =  X[test_idx]  #测试样本
Y_train = Y[train_idx]
Y_test = Y[test_idx]
print("X_train:",X_train.shape)
print("Y_train:",Y_train.shape)
print("X_test:",X_test.shape)
print("Y_test:",Y_test.shape)

"""建立模型"""
classes = ['8PSK',
 'AM-DSB',
 'AM-SSB',
 'BPSK',
 'CPFSK',
 'GFSK',
 'PAM4',
 'QAM16',
 'QAM64',
 'QPSK',
 'WBFM']


def residual_stack(X,Filters,Seq,max_pool):
    #1*1 Conv Linear
    X = Conv2D(Filters, (1, 1), padding='same', name=Seq+"_conv1", init='glorot_uniform',data_format="channels_first")(X)
    #Residual Unit 1
    X_shortcut = X
    X = Conv2D(Filters, (3, 2), padding='same',activation="relu",name=Seq+"_conv2", init='glorot_uniform',data_format="channels_first")(X)
    X = Conv2D(Filters, (3, 2), padding='same', name=Seq+"_conv3", init='glorot_uniform',data_format="channels_first")(X)
    X = layers.add([X,X_shortcut])
    X = Activation("relu")(X)
    #Residual Unit 2
    X_shortcut = X
    X = Conv2D(Filters, (3, 2), padding='same',activation="relu",name=Seq+"_conv4", init='glorot_uniform',data_format="channels_first")(X)
    X = Conv2D(Filters, (3, 2), padding='same', name=Seq+"_conv5", init='glorot_uniform',data_format="channels_first")(X)
    X = layers.add([X,X_shortcut])
    X = Activation("relu")(X)
    #MaxPooling
    if max_pool:
        X = MaxPooling2D(pool_size=(2, 1), strides=(2, 1), padding='valid', data_format="channels_first")(X)
    return X


in_shp = X_train.shape[1:]   #每个样本的维度
#input layer
X_input = Input(in_shp)
X = Reshape([1,1024,2], input_shape=in_shp)(X_input)
#Residual Srack 1
X = residual_stack(X,32,"ReStk1",True)  #shape:(1,128,32)
#Residual Srack 2
X = residual_stack(X,32,"ReStk2",True)  #shape:(1,64,32)
#Residual Srack 3
X = residual_stack(X,32,"ReStk3",True)  #shape:(1,32,32)
#Residual Srack 4
X = residual_stack(X,32,"ReStk4",True)  #shape:(1,16,32)
#Full Con 1
X = Flatten()(X)
X = Dense(128, activation='selu', init='he_normal', name="dense1")(X)
X = AlphaDropout(0.3)(X)
#Full Con 2
X = Dense(128, activation='selu', init='he_normal', name="dense2")(X)
X = AlphaDropout(0.3)(X)
#Full Con 3
X = Dense(len(classes), init='he_normal', name="dense3")(X)
#SoftMax
X = Activation('softmax')(X)
#Create Model
model = Model.Model(inputs=X_input,outputs=X)
model.compile(loss='categorical_crossentropy', optimizer='adam')
model.summary()
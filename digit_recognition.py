#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
import os
import cv2
import tensorflow as tf


# In[2]:


def normalize(img):
    img=(img-np.mean(img))/(np.std(img))
    return img


# In[3]:


base_dir=r"C:\Users\Aakriti Poudel\Desktop\digit\database"
files=os.listdir(base_dir)


# In[4]:


x_train=[]
y_train=[]
x_test=[]
y_test=[]
size=30
for i in tqdm(files):
    x=[]
    y=[]
    path=os.path.join(base_dir,i)
    file_name=os.listdir(path)
    for j in file_name:
        one_hot=np.zeros((1,10))
        one_hot[0,int(i)]=1
        img=cv2.imread(os.path.join(path,j),cv2.IMREAD_GRAYSCALE)
        img=cv2.resize(img,(size,size))
        img=np.array(img).reshape(1,-1)
        img=normalize(img)
        x.append(img)
        y.append(one_hot)
    x=np.array(x)
    y=np.array(y)
    x_test.append(x[:100,:,:])
    y_test.append(y[:100,:,:])
    x_train.append(x[100:,:,:])
    y_train.append(y[100:,:,:])

        


# In[5]:


x_train=np.array(x_train).reshape(-1,size*size)
y_train=np.array(y_train).reshape(-1,10)
x_test=np.array(x_train).reshape(-1,size*size)
y_test=np.array(y_train).reshape(-1,10)


# In[6]:


print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)


# In[7]:


model=tf.keras.Sequential([
    tf.keras.layers.Dense(size*size,activation='relu',input_shape=(size*size,),kernel_regularizer='l2'),
    tf.keras.layers.Dense(160,activation='relu',kernel_regularizer='l2'),
    tf.keras.layers.Dense(80,activation='relu',kernel_regularizer='l2'),
    tf.keras.layers.Dense(40,activation='relu',kernel_regularizer='l2'),
    tf.keras.layers.Dense(10,activation='softmax',kernel_regularizer='l2'),
])


# In[8]:


model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])


# In[9]:


model.summary()


# In[10]:


history=model.fit(x_train,y_train,batch_size=64,epochs=20)


# In[11]:


plt.plot(history.history['loss'])
plt.show()


# In[12]:


for i in range(x_test.shape[0]):
    test=x_test[np.random.randint(0,x_test.shape[0])]
    v=test.reshape(size,size)
    plt.imshow(v)
    plt.show()
    pre=model.predict(test.reshape(1,size*size))
    p=np.argmax(pre)
    print("Neural Network Prediction",p)
    print('continue?')
    c=input()
    if(c=='y' or c=='Y'):
        continue
    else:
        print('network exitting.')
        break


# In[ ]:





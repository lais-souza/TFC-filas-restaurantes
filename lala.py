# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 19:47:07 2019

@author: luis.lima
"""

#Importando as bibliotecas
import os
import numpy as np
from skimage import data
import random
import tensorflow as tf
import pickle
import dill
import matplotlib.pyplot as plt

# Root directory where files are contained
ROOT_PATH = os.getcwd()
# Number of network outputs
NUMBER_OF_OUTPUTS = 62

################################################################################

# Function that organize and catch the data from image database and classify them
def load_data(data_directory,y_cim,y_bai,x_esq,x_dir,color):
    directories = [d for d in os.listdir(data_directory)
                   if os.path.isdir(os.path.join(data_directory, d))]
    labels = []
    images = []
    for d in directories:
        label_directory = os.path.join(data_directory, d)
        file_names = [os.path.join(label_directory, f)
                      for f in os.listdir(label_directory)
                      if f.endswith(".jpeg")]
        for fi in range(0,len(file_names),2):
            f=file_names[fi]
            a=data.imread(f)
            image=np.array(a,dtype=np.float32)
            image=img_resize(image,y_cim,y_bai,x_esq,x_dir,color)
            images.append(image)
            labels.append(int(d))
    return images, labels

################################################################################

#Função do Xandim
def xandico(img):
    r,g,b=img[:,:,:,0],img[:,:,:,1],img[:,:,:,2]
    gray=0.2989*r+0.5870*g+0.1140*b
    return gray

def img_resize(img,y_cim,y_bai,x_esq,x_dir,color):
    img=img[y_cim:y_bai , x_esq:x_dir , color]
    return img

# Loading training image data
    
#Tamanho e cor da Fig
x_esq=500
x_dir=1300
y_cim=400
y_bai=900
color=1

#Define o diretorio
train_data_directory = os.path.join(ROOT_PATH, "Training")

#Chama a funcao que le, arruma o tamanho e escolhe a cor
images_raw, labels = load_data(train_data_directory,y_cim,y_bai,x_esq,x_dir,color)

#Cria um np array 
images_raw = np.array(images_raw,dtype=np.float32)


"""LALA.py"""

#Plot das cores
#images_raw_0 = images_raw[: , y_cim:y_bai , x_esq:x_dir , 0]
#images_raw_1 = images_raw[: , y_cim:y_bai , x_esq:x_dir , 1]
#images_raw_2 = images_raw[: , y_cim:y_bai , x_esq:x_dir , 2]
#images_raw_x = xandico( images_raw[: , y_cim:y_bai , x_esq:x_dir , :] )
#fig,ax=plt.subplots(1,4)
#ax[0].imshow(images_raw_0[0,:,:],cmap='gray')
#ax[1].imshow(images_raw_1[0,:,:],cmap='gray')
#ax[2].imshow(images_raw_2[0,:,:],cmap='gray')
#ax[3].imshow(images_raw_x[0,:,:],cmap='gray')
#ax[0].set_title("R")
#ax[1].set_title("G")
#ax[2].set_title("B")
#ax[3].set_title("X")

#Codigo para subtrair as imagens
images=images_raw-images_raw[0,:,:]

#Plot da subtracao
fig,ax=plt.subplots(1,3)
ax[0].imshow(images_raw[320,:,:],cmap='gray')
ax[1].imshow(images_raw[0,:,:],cmap='gray')
ax[2].imshow(images[320,:,:],cmap='gray')
ax[0].set_title("Imagem com fila")
ax[1].set_title("Imagem sem fila")
ax[2].set_title("Imagem subtraidas")





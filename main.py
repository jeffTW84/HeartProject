## -*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, model_from_json#, model_from_json
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.applications import Xception
from tensorflow.python.client import device_lib
import tensorflow as tf
import numpy as np
import os
import cv2

# heatmap
class GradCAM:
    # Adapted with some modification from https://www.pyimagesearch.com/2020/03/09/grad-cam-visualize-class-activation-maps-with-keras-tensorflow-and-deep-learning/
    def __init__(self, model, layerName=None):
        """
        model: pre-softmax layer (logit layer)
        """
        self.model = model
        self.layerName = layerName
            
        if self.layerName == None:
            self.layerName = self.find_target_layer()
    
    def find_target_layer(self):
        for layer in reversed(self.model.layers):
            if len(layer.output_shape) == 4:
                return layer.name
        raise ValueError("Could not find 4D layer. Cannot apply GradCAM")
            
    def compute_heatmap(self, image, classIdx, upsample_size, eps=1e-5):
        gradModel = tf.keras.Model(
            inputs = [self.model.inputs],
            outputs = [self.model.get_layer(self.layerName).output, self.model.output]
        )
        # record operations for automatic differentiation
        
        with tf.GradientTape() as tape:
            inputs = tf.cast(image, tf.float32)
            (convOuts, preds) = gradModel(inputs) # preds after softmax
            loss = preds[:,classIdx]
        
        # compute gradients with automatic differentiation
        grads = tape.gradient(loss, convOuts)
        # discard batch
        convOuts = convOuts[0]
        grads = grads[0]
        norm_grads = tf.divide(grads, tf.reduce_mean(tf.square(grads)) + tf.constant(eps))
        
        # compute weights
        weights = tf.reduce_mean(norm_grads) # 出錯！ 解決：axis=(0, 1)-->axis(0)
        cam = tf.reduce_sum(tf.multiply(weights, convOuts), axis=-1)
        
        # Apply reLU
        cam = np.maximum(cam, 0)
        cam = cam/np.max(cam)
        # cam = cv2.resize(cam, upsample_size, interpolation=cv2.INTER_LINEAR)
        
        # convert to 3D
        cam3 = np.expand_dims(cam, axis=0)
        cam3 = np.tile(cam3, [1,1,3])
        
        return cam3
    
def preprocess_input(ori_x):
    x = ori_x / 255.0
    return x
    
def overlay_gradCAM(img, cam3, img_size):
    cam3 = np.uint8(255*cam3)
    cam3 = cv2.applyColorMap(cam3, cv2.COLORMAP_JET)

    # img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    new_img = 0.3*cam3 + 0.5*img
    
    return new_img.astype("uint8")
    # return (new_img*255.0/new_img.max()).astype("uint8")

def show_gradCAMs(model, gradCAM, img, decode={}, cls_idx='default', upsample_size='default'):
    """
    model: softmax layer
    """
    plt.subplots(figsize=(30, 10))
    
    if upsample_size=='default':
      upsample_size = img.shape

    # Show original image
    plt.subplot(1,4,1)
    plt.imshow(img)
    plt.title("sample", fontsize=20)
    plt.axis("off")

    # Show overlayed grad
    plt.subplot(1,4,2)
    x = np.expand_dims(img, axis=0)
    x = preprocess_input(x)
    preds = model.predict(x)

    if cls_idx=='default':
      cls_idx = preds.argmax()
    
    res = [cls_idx, preds[0][cls_idx]]
    cam3 = gradCAM.compute_heatmap(image=x, classIdx=cls_idx, upsample_size=upsample_size)
    new_img = overlay_gradCAM(img, cam3, upsample_size)
    new_img = cv2.cvtColor(new_img, cv2.COLOR_BGR2RGB)
    plt.imshow(new_img)
    # plt.imshow(cam3)
    plt.title("GradCAM - Pred: {}. Prob: {}".format(res[0],res[1]), fontsize=20)
    plt.axis("off")

    plt.show()

# path
train_img_path = "./Train/"
val_img_path = "./Dev/"

# image相關設定
num_class = 6 # classes of img
img_color_chnl = 3 # 3:RGB　1:GrayScale
resize_wh = (224, 224)
fill_opt = 'nearest'
resize_opt = 'bilinear' # 'nearest', 'bilinear', 'bicubic'

# Augement設定
#augment_ratio = 1 #用原data產生幾倍的training data ,1:數量不變
augment_ratio = 3

# 設定超參數HyperParameters 
batch_size = 8
hp_Epochs = 70
hp_LearningRate = 1e-4

loss_func = 'categorical_crossentropy'

dropout_en = 1
dropout_rate = 0.5

freeze_layers_en = 0
freeze_layers_num = 2

# 輸出權重檔設定
outp_path = './model/'

def train():
    # ImageDataGenerator: 原data不會用來training，所有training data都用原data依照參數隨機生成
    train_datagen = ImageDataGenerator(rescale = 1./255.,
                                        rotation_range = 90,
                                        width_shift_range = 0.1,
                                        height_shift_range = 0.1,
                                        zoom_range = 0.1,
                                        horizontal_flip = True,
                                        vertical_flip = True
                                        )
            
    valid_datagen = ImageDataGenerator(rescale = 1./255.)

    # train, val 的 batch 到 directory 依 class 提取 img
    train_set = train_datagen.flow_from_directory(directory = train_img_path,
                                                      target_size = resize_wh,
                                                      interpolation = resize_opt,
                                                      color_mode="rgb",
                                                      batch_size=batch_size,
                                                      class_mode='categorical',
                                                      shuffle=True,
                                                      seed = 42
                                                      )

    valid_set = valid_datagen.flow_from_directory(directory = val_img_path,
                                                      target_size = resize_wh,
                                                      interpolation = resize_opt,
                                                      color_mode="rgb",
                                                      class_mode='categorical',
                                                      )

    # 以訓練好的 Xception 為基礎來建立模型，捨棄 Xception 頂層的 fully connected layers
    base_model = Xception(include_top=False, weights='imagenet', input_tensor=None,
                   input_shape=( resize_wh[0], resize_wh[1], img_color_chnl) )
                   
    model = Sequential()
    model.add(base_model)
    model.add(Flatten())
    # 增加 DropOut layer
    if(dropout_en):
        model.add(Dropout(dropout_rate))
    # 增加 Dense layer，以 softmax 產生個類別的機率值
    model.add(Dense(num_class, activation="softmax", name='predictions'))

    # 設定凍結與要進行訓練的網路層
    if(freeze_layers_en):
        for layer in base_model.layers[:freeze_layers_num]:
            layer.trainable = False
        for layer in base_model.layers[freeze_layers_num:]:
            layer.trainable = True

    # 使用 Adam optimizer，以較低的 learning rate 進行 fine-tuning
    model.compile(optimizer=Adam(lr=hp_LearningRate),
                  loss=loss_func,
                  metrics=['accuracy'])

    # 輸出整個網路結構
    print(model.summary())

    # 保存最好的權重
    if not os.path.isdir(outp_path):
        os.mkdir(outp_path)
    filepath = outp_path + "weights.best.hdf5"
    checkpoint = ModelCheckpoint(filepath, verbose=1, monitor='val_accuracy', save_best_only=True, mode='auto')
    callbacks_list = [checkpoint]

    # 訓練模型
    print(train_set.samples)
    history = model.fit_generator(train_set,
                        steps_per_epoch = int(train_set.samples / batch_size),
                        epochs = hp_Epochs,
                        validation_data = valid_set,
                        callbacks=callbacks_list)

    # 儲存模型
    if not os.path.isdir(outp_path):
        os.mkdir(outp_path)
    filepath = outp_path + "model.json"
    model_json = model.to_json()
    with open(filepath, "w") as json_file:
        json_file.write(model_json)
    filepath = outp_path + "weights.finalEpoch.h5"
    model.save_weights(filepath)
    print("Saved model to disk")
    
    plot_learning_curves(history)

# 繪製Model學習成效
def plot_learning_curves(history):
	pd.DataFrame(history.history).plot(figsize=(8,5))
	plt.grid(True)
	pd.DataFrame(history.history).plot(figsize=(8,5))
	plt.grid(True)
	plt.gca().set_ylim(0,1)
	plt.show()

if __name__ == '__main__':
    
    # print(device_lib.list_local_devices())
    # train()
    
    model = model_from_json(open('./model/model.json').read())
    
    sample = plt.imread('./Dev/3/IMG-34094514-S00030.jpg')
    org_len = 1024
    new_len = 224
    new_sample = np.zeros((new_len, new_len, 3))
    new_sample = cv2.resize(sample, (224, 224))
    
    gradCAM = GradCAM(model=model, layerName="predictions")
    show_gradCAMs(model, gradCAM, new_sample, cls_idx=3)
    
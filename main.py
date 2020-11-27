## -*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, model_from_json, Model, load_model#, model_from_json
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.applications import Xception
from tensorflow.python.client import device_lib
import tensorflow as tf
import numpy as np
from keras import backend as K
import os
import cv2

train_img_path = "./Train/"
val_img_path = "./Dev/"

num_class = 6
img_color_chnl = 3
resize_wh = (224, 224)
fill_opt = 'nearest'
resize_opt = 'bilinear'

augment_ratio = 3

batch_size = 8
hp_Epochs = 50
hp_LearningRate = 1e-4

loss_func = 'categorical_crossentropy'

dropout_en = 1
dropout_rate = 0.5

freeze_layers_en = 0
freeze_layers_num = 2

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
                   
    out = base_model.output
    out = Flatten()(out)
    # 增加 DropOut layer
    if(dropout_en):
        out = Dropout(dropout_rate)(out)
    # 增加 Dense layer，以 softmax 產生個類別的機率值
    out = Dense(num_class, activation="softmax", name='predictions')(out)

    model = Model(inputs=base_model.input, outputs=out)

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
    #model.save_weights(filepath)
    model.save('./tmp/model.h5')
    print("Saved model to disk")
    
    plot_learning_curves(history)

def plot_learning_curves(history):
	pd.DataFrame(history.history).plot(figsize=(8,5))
	plt.grid(True)
	pd.DataFrame(history.history).plot(figsize=(8,5))
	plt.grid(True)
	plt.gca().set_ylim(0,1)
	plt.show()

def get_layer_output(model, x, index=-1):
    layer = K.function([model.input], [model.layers[index].output])
    return layer([x])[0]

if __name__ == '__main__':
    """
    print(device_lib.list_local_devices())
    train()
    """
    
    sample = plt.imread('./Dev/3/IMG-34094514-S00030.jpg')
    org_len = 1024
    new_len = 224
    new_sample = np.zeros((new_len, new_len, 3))
    input_x = cv2.resize(sample, (224, 224))
    
    model = load_model('./tmp/model.h5')
    model.summary()
    input_x = np.expand_dims(input_x, 0)
    layer4 = get_layer_output(model, input_x, index=4)
    filter_num = 0
    print(layer4[0].shape)
    len_ = 109
    test_output = np.zeros((len, len))
    for i in range(len):
        for j in range(len):
            test_output[i][j] = layer4[0][i][j][filter_num]
    # test_output = layer4[0][0]
    print(test_output)
    print(test_output.shape)
    
    plt.plot()
    plt.imshow(test_output, cmap="gray")
    plt.title("test")
    plt.axis("off")
    plt.show()
    
    """
    gradCAM = GradCAM(model=model, layerName="block14_sepconv2")
    show_gradCAMs(model, gradCAM, new_sample, cls_idx=3)
    """
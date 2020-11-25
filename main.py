## -*- coding: utf-8 -*-
###############################################
# 資料處理套件
import pandas as pd
import matplotlib.pyplot as plt

# 深度學習模組套件
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential#, model_from_json
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint

# misc
import os

###############################################
# path
train_img_path = "./Train/"
val_img_path = "./Dev/"

# image相關設定
num_class = 6 # classes of img
img_color_chnl = 3 # 3:RGB　1:GrayScale , 後面 repeat grayscale 成 3 channel img
resize_wh = (224, 224)
fill_opt = 'nearest'
resize_opt = 'bilinear' # 'nearest', 'bilinear', 'bicubic'

# Augement設定
augment_ratio = 1 #用原data產生幾倍的training data ,1:數量不變

# 設定超參數HyperParameters 
hp_BatchSize = 1
hp_Epochs = 50
hp_LearningRate = 1e-4

loss_func = 'categorical_crossentropy'

dropout_en = 1
dropout_rate = 0.5


# 輸出權重檔設定
outp_path = './model/'

###############################################
# Model

# 透過 data augmentation 產生訓練與驗證用的影像資料
# ImageDataGenerator: 原data不會用來training，所有training data都用原data依照參數隨機生成
#train_datagen = ImageDataGenerator(rescale=1./255.) # for fastest run
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
train_batches = train_datagen.flow_from_directory(directory = train_img_path,
                                                  target_size = resize_wh,
                                                  interpolation = resize_opt,
                                                  batch_size=hp_BatchSize,
                                                  class_mode='categorical',
                                                  shuffle=True,
                                                  seed = 42
                                                  )

valid_batches = valid_datagen.flow_from_directory(directory = val_img_path,
                                                  target_size = resize_wh,
                                                  interpolation = resize_opt,
                                                  class_mode='categorical'
                                                  )

# 以訓練好的 Xception 為基礎來建立模型，捨棄 Xception 頂層的 fully connected layers
from tensorflow.keras.applications import Xception
base_model = Xception(include_top=False, weights='imagenet', input_tensor=None,
               input_shape=( resize_wh[0], resize_wh[1], img_color_chnl) )

out = base_model.output
out = Flatten()(out)
# 增加 DropOut layer
if(dropout_en):
  out = Dropout(dropout_rate)(out)
# 增加 Dense layer，以 softmax 產生個類別的機率值
out = Dense(num_class, activation="softmax")(out)

model = Model(inputs=base_model.input, outputs=out)

# 使用 Adam optimizer，以較低的 learning rate 進行 fine-tuning
model.compile(optimizer=Adam(lr=hp_LearningRate),
              loss=loss_func,
              metrics=['accuracy'])

# 輸出整個網路結構
print(model.summary())

# 保存最好的權重
if not os.path.isdir(outp_path):
    os.mkdir(outp_path)
filepath = outp_path + "model.best.hdf5"
checkpoint = ModelCheckpoint(filepath, verbose=1, monitor='val_accuracy', save_best_only=True, mode='auto')
callbacks_list = [checkpoint]

# 訓練模型
print(train_batches.samples)
history = model.fit_generator(train_batches,
							steps_per_epoch = train_batches.samples * augment_ratio // hp_BatchSize,
							validation_data = valid_batches,
							epochs = hp_Epochs,
							callbacks=callbacks_list,
                            verbose=1)

# 儲存模型
"""
if not os.path.isdir(outp_path):
    os.mkdir(outp_path)
filepath = outp_path + "model.json"
model_json = model.to_json()
with open(filepath, "w") as json_file:
    json_file.write(model_json)
filepath = outp_path + "weights.finalEpoch.h5"
model.save_weights(filepath)
print("Saved model to disk")
"""

###############################################
# 繪製Model學習成效
def plot_learning_curves(history):
	pd.DataFrame(history.history).plot(figsize=(8,5))
	plt.grid(True)
	pd.DataFrame(history.history).plot(figsize=(8,5))
	plt.grid(True)
	plt.gca().set_ylim(0,1)
	plt.show()

plot_learning_curves(history)

###############################################

import os
import csv
import cv2
import numpy as np
import sklearn
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from keras.layers import Cropping2D, Dense, Dropout, Flatten, Lambda
from keras.layers.convolutional import Convolution2D
from keras.models import Sequential
from sklearn.model_selection import train_test_split

# 读取 csv 日志数据
samples = []
with open('./driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

# 把日志数据划分为训练和验证两部分，比例为 80%,20%
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

# 获得图片和转角
def image_and_angle_from_data(batch_sample, camera_position):
    name = './data/IMG/'+batch_sample[camera_position].split('/')[-1]
    image = cv2.imread(name)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.GaussianBlur(image, (3,3),0)
    angle = float(batch_sample[3])
    return image, angle

# 制造训练数据和验证数据
def generator(samples, batch_size=32):
    num_samples = len(samples)
    # 永远循环，这样生成器就不会终止
    while 1:
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                # 从中心，左和右的相机中读取图像和转向测量
                center_image, center_angle = image_and_angle_from_data(batch_sample, 0)
                images.append(center_image)
                angles.append(center_angle)

                left_image, left_angle = image_and_angle_from_data(batch_sample, 1)
                left_angle = center_angle + 0.2
                images.append(left_image)
                angles.append(left_angle)

                right_image, right_angle = image_and_angle_from_data(batch_sample, 2)
                right_angle = center_angle - 0.2
                images.append(right_image)
                angles.append(right_angle)

            # 包含翻转图像
            augmented_images, augmented_angles = [], []
            for image, angle in zip(images, angles):
                augmented_images.append(image)
                augmented_angles.append(angle)
                augmented_images.append(cv2.flip(image, 1))
                augmented_angles.append(angle * -1.0)

            # 得到训练数据和验证数据
            X_train = np.array(augmented_images)
            y_train = np.array(augmented_angles)
            yield sklearn.utils.shuffle(X_train, y_train)

# 用 generator 方法获得训练模型和验证模型
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

# 使用 NVIDIA 模型
model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160, 320, 3)))
model.add(Cropping2D(cropping=((70, 25), (0, 0))))
model.add(Convolution2D(24, 5, 5, subsample=(2, 2), activation="relu"))
model.add(Convolution2D(36, 5, 5, subsample=(2, 2), activation="relu"))
model.add(Convolution2D(48, 5, 5, subsample=(2, 2), activation="relu"))
model.add(Convolution2D(64, 3, 3, activation="relu"))
model.add(Convolution2D(64, 3, 3))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
history_object = model.fit_generator(train_generator, samples_per_epoch=len(train_samples), validation_data=validation_generator,\
    nb_val_samples=len(validation_samples), nb_epoch=5, verbose=1)

model.save('model.h5')
model.summary()

# 打印历史对象中包含的键
print(history_object.history.keys())

# 为每个层绘制训练和验证损失的图表
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()
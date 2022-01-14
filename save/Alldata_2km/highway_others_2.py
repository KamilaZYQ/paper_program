# 时间：2022年1月9日
# 用途：将1月4的全部数据仅分为高速工况和其他工况（一般+城市）
# %%
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D, Softmax, Dropout
from tensorflow.keras import Model
import numpy as np
from tensorflow.python.keras.engine import training
from tensorflow.python.keras.metrics import accuracy
mnist = tf.keras.datasets.mnist
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
import matplotlib.pyplot as plt
from random import sample

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
# %%
# 处理训练数据
traindata = np.load('/home/yuqi.zhao1/paper_program/save/Alldata_2km/AllTrain_2.npy')
x1_data = traindata[:,:5]
y1_data = traindata[:,5:].astype(np.int)
x1_data[:,0] = x1_data[:,0]/np.max(x1_data[:,0])
labels_count = np.bincount(y1_data[:,0])
x_train = [[],[]]
y_train = [[],[]]
for i in range(x1_data.shape[0]):
    if (y1_data[i, 0] == 0):
        x_train[0].append(x1_data[i,:])
        y_train[0].append(y1_data[i,:])
    else: 
        x_train[1].append(x1_data[i,:])
        y_train[1].append(np.array([1]))
print('highway_train:',len(x_train[0]),'mixed_train:',len(x_train[1]))

# %%
# 处理测试数据
testdata = np.load('/home/yuqi.zhao1/paper_program/save/Alldata_2km/AllTest_2.npy')
x2_data = testdata[:,:5]
y2_data = testdata[:,5:].astype(np.int)
x2_data[:,0] = x2_data[:,0]/np.max(x2_data[:,0])
labels_count = np.bincount(y2_data[:,0])
x_test = [[],[]]
y_test = [[],[]]
for i in range(x2_data.shape[0]):
    if (y2_data[i, 0] == 0):
        x_test[0].append(x2_data[i,:])
        y_test[0].append(y2_data[i,:])
    else: 
        x_test[1].append(x2_data[i,:])
        y_test[1].append(np.array([1]))
print('highway_test:',len(x_test[0]),'mixed_test:',len(x_test[1]))
x_test = np.array(x_test[0] + x_test[1])
y_test = np.array(y_test[0] + y_test[1])

# %%
class MyModel(Model):
  def __init__(self):
    super(MyModel, self).__init__()

    self.d1 = Dense(32, activation='relu', use_bias= True)
    self.d2 = Dense(64, activation='relu', use_bias= True)
    self.d3 = Dense(128, activation='relu', use_bias= True)
    self.d4 = Dense(256, activation='relu', use_bias= True)
    self.d5 = Dense(512, activation='relu', use_bias= True)
    self.d6 = Dense(256, activation='relu', use_bias= True)
    self.d7 = Dense(64, activation='relu', use_bias= True)
    self.d8 = Dense(2)
    self.dropout = Dropout(rate = 0.2)
    self.softmax = Softmax(axis=-1)

  def call(self, x, training=False):

    x = self.d1(x)
    x = self.d2(x)
    x = self.d3(x)
    x = self.d4(x)
    x = self.d5(x)
    x = self.d6(x)
    x = self.dropout(x, training)
    x = self.d7(x)
    x = self.dropout(x, training)
    x = self.d8(x)
    return self.softmax(x)

# %%
model = MyModel()
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)

optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

# %%
@tf.function
def train_step(images, labels):
  with tf.GradientTape() as tape:
    predictions = model(images, training=True)
    loss = loss_object(labels, predictions)
  gradients = tape.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(gradients, model.trainable_variables))

  train_loss(loss)
  train_accuracy(labels, predictions)
# %%
# @tf.function
def test_step(images, labels):
  predictions = model(images, training=False)
  if tf.argmax(predictions,axis=1)[0] == labels:
     return True
#   t_loss = loss_object(labels, predictions)

#   test_loss(t_loss)
#   test_accuracy(labels, predictions)
# %%
EPOCHS = 20
epochs = []
acc1s = []
acc2s = []
for epoch in range(EPOCHS):
  epochs.append(epoch)
  train_loss.reset_states()
  train_accuracy.reset_states()
  test_loss.reset_states()
  test_accuracy.reset_states()
  x_aug = [[],[]]
  y_aug = [[],[]]
  sample_count = 7000
  for i in range(len(x_train[0])):
    if np.random.rand(1) < sample_count/len(x_train[0]):
      for j in range(len(x_train[0][i])):
        x_aug[0].append(x_train[0][i]* (1 + (np.random.rand(1) - 0.5) / 5))
        y_aug[0].append(y_train[0][i])
  # for i in range(len(x_aug[0])):
  #     for j in range(len(x_aug[0][i])):
  #         x_aug[0][i][j] += x_aug[0][i][j] * (np.random.rand(1) - 0.5) / 5
  x_aug[1] = x_train[1]
  for i in range(len(x_aug[1])):
      for j in range(len(x_aug[1][i])):
          x_aug[1][i][j] += x_aug[1][i][j] * (np.random.rand(1) - 0.5) / 5


  x_aug = np.array(x_aug[0] + x_aug[1])
  y_aug = np.array(y_aug[0] + y_train[1] )

  train_ds = tf.data.Dataset.from_tensor_slices(
      (x_aug, y_aug)).shuffle(14000).batch(32)

  test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(1)
  for images, labels in train_ds:
    train_step(images, labels)
    correct1,correct2= 0,0
    count1,count2=0,0

  for test_images, test_labels in test_ds:
    test_labels = test_labels[0,0]
    result = test_step(test_images, test_labels)
    if test_labels == 0:
        count1 +=1
        if result:
            correct1 +=1
    else:
      count2 +=1
      if result:
        correct2 +=1
  acc1 = correct1/count1
  acc2 = correct2/count2
  acc1s.append(acc1*100)
  acc2s.append(acc2*100)
  print(
    f'Epoch {epoch + 1}, '
    f'Loss: {train_loss.result()}, '
    f'Accuracy: {train_accuracy.result() * 100}, '
    f'Test Accuracy: {acc1 * 100} {acc2 * 100} '
  )
# %%
plt.plot(epochs, acc1s, 'b', label='highway accuracy')
plt.plot(epochs, acc2s, 'g', label='mixedway accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
# %%
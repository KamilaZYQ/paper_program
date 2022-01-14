# 时间：2021年12月16日
# 用途：将所有数据（new）分为三种工况进行TF分类
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

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
# %%
# 处理训练数据
traindata = np.load('/home/yuqi.zhao1/paper_program/save/Alldata_3km_first/AllTrain.npy')
x1_data = traindata[:,:5]
y1_data = traindata[:,5:].astype(np.int)
x1_data[:,0] = x1_data[:,0]/np.max(x1_data[:,0])
labels_count = np.bincount(y1_data[:,0])
x_train = [[],[],[]]
y_train = [[],[],[]]
for i in range(x1_data.shape[0]):
    if (y1_data[i, 0] == 0):
        x_train[0].append(x1_data[i,:])
        y_train[0].append(y1_data[i,:])
    elif (y1_data[i, 0] == 1): 
        x_train[1].append(x1_data[i,:])
        y_train[1].append(y1_data[i,:])
    else: 
        x_train[2].append(x1_data[i,:])
        y_train[2].append(y1_data[i,:])
print('highway_train:',len(x_train[0]),'suburb_train:',len(x_train[1]),'city_train:',len(x_train[2]))

# %%
# 处理测试数据
testdata = np.load('/home/yuqi.zhao1/paper_program/save/Alldata_3km_first/AllTest.npy')
x2_data = testdata[:,:5]
y2_data = testdata[:,5:].astype(np.int)
x2_data[:,0] = x2_data[:,0]/np.max(x2_data[:,0])
labels_count = np.bincount(y2_data[:,0])
x_test = [[],[],[]]
y_test = [[],[],[]]
for i in range(x2_data.shape[0]):
    if (y2_data[i, 0] == 0):
        x_test[0].append(x2_data[i,:])
        y_test[0].append(y2_data[i,:])
    elif (y2_data[i, 0] == 1): 
        x_test[1].append(x2_data[i,:])
        y_test[1].append(y2_data[i,:])
    else: 
        x_test[2].append(x2_data[i,:])
        y_test[2].append(y2_data[i,:])
print('highway_test:',len(x_test[0]),'suburb_test:',len(x_test[1]),'city_test:',len(x_test[2]))
x_test = np.array(x_test[0] + x_test[1] + x_test[2])
y_test = np.array(y_test[0] + y_test[1] + y_test[2])

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
    self.d8 = Dense(3)
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

optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
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
acc3s = []
for epoch in range(EPOCHS):
  epochs.append(epoch)
  train_loss.reset_states()
  train_accuracy.reset_states()
  test_loss.reset_states()
  test_accuracy.reset_states()
  x_aug = [[],[],[]]
  y_aug = [[],[],[]]
  x_aug[0] = x_train[0]
  x_aug[1] = x_train[1] * 4
  for i in range(len(x_aug[1])//4, len(x_aug[1])):
      for j in range(len(x_aug[1][i])):
          x_aug[1][i][j] += x_aug[1][i][j] * (np.random.rand(1) - 0.5) / 5
  x_aug[2] = x_train[2] * 20
  for i in range(len(x_aug[2])//20, len(x_aug[2])):
      for j in range(len(x_aug[2][i])):
          x_aug[2][i][j] += x_aug[2][i][j] * (np.random.rand(1) - 0.5) / 5
  x_aug = np.array(x_aug[0] + x_aug[1] + x_aug[2])
  y_aug = np.array(y_train[0] + y_train[1] * 4 + y_train[2] * 20)

  train_ds = tf.data.Dataset.from_tensor_slices(
      (x_aug, y_aug)).shuffle(60000).batch(32)

  test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(1)
  for images, labels in train_ds:
    train_step(images, labels)
    correct1,correct2,correct3 = 0,0,0
    count1,count2,count3=0,0,0

  for test_images, test_labels in test_ds:
    test_labels = test_labels[0,0]
    result = test_step(test_images, test_labels)
    if test_labels == 0:
        count1 +=1
        if result:
            correct1+=1
    elif test_labels == 1:
        count2+=1
        if result:
            correct2+=1
    else:
        count3+=1
        if result:
            correct3+=1
  acc1 = correct1/count1
  acc2 = correct2/count2
  acc3 = correct3/count3
  acc1s.append(acc1*100)
  acc2s.append(acc2*100)
  acc3s.append(acc3*100)
  print(
    f'Epoch {epoch + 1}, '
    f'Loss: {train_loss.result()}, '
    f'Accuracy: {train_accuracy.result() * 100}, '
    f'Test Accuracy: {acc1 * 100} {acc2 * 100} {acc3 * 100}'
  )
# %%
plt.plot(epochs, acc1s, 'b', label='highway accuracy')
plt.plot(epochs, acc2s, 'g', label='suburban accuracy')
plt.plot(epochs, acc3s, 'r', label='city accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
# %%
model.save()
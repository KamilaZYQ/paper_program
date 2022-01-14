# 时间：2022年1月6日
# 用途：从服务器读取数据，加上0，1，2标签，把训练数据和测试数据从总数据中筛出来；
# 区别：不能和3km一样基于老师的训练和测试数据来划分，需要随机抽取10%的数据用于测试，baseline为统计学可视化结果
import os
import os.path as opt
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool
import multiprocessing
import scipy.io as sio
from scipy import stats
import matplotlib as mlb
import matplotlib.pyplot as plt
import random 
# %%
# 城市公路工况数据提取 2
city_all_speed = np.load("/home/yuqi.zhao1/Data/speed/city_speed_2.npy")
print(len(city_all_speed))
#档位
city_all_gear = np.load("/home/yuqi.zhao1/Data/gear/city_gear_2.npy").tolist()
print(len(city_all_gear))

city_all_data = []
for i in range(len(city_all_speed)):
    city_all_data.append([city_all_speed[i]]+city_all_gear[i]+[2])
print("city_all_data:",len(city_all_data))
# %%
# 一般公路工况数据提取 1
#速度
suburban_all_speed = np.load("/home/yuqi.zhao1/Data/speed/suburb_speed_2.npy")
print(len(suburban_all_speed))
#档位
suburban_all_gear = np.load("/home/yuqi.zhao1/Data/gear/suburb_gear_2.npy").tolist()
print(len(suburban_all_gear))

suburban_all_data = []
for i in range(len(suburban_all_speed)):
    suburban_all_data.append([suburban_all_speed[i]]+suburban_all_gear[i]+[1])
print("suburban_all_data:",len(suburban_all_data))
# %%
# 高速公路工况数据提取 0
#速度
highway_all_speed = np.load("/home/yuqi.zhao1/Data/speed/highway_speed_2.npy")
print(len(highway_all_speed))
#档位
highway_all_gear = np.load("/home/yuqi.zhao1/Data/gear/highway_gear_2.npy").tolist()
print(len(highway_all_gear))

highway_all_data = []
for i in range(len(highway_all_speed)):
    highway_all_data.append([highway_all_speed[i]]+highway_all_gear[i]+[0])
print("highway_all_data:",len(highway_all_data))
# %%
data_all = np.concatenate([highway_all_data,suburban_all_data,city_all_data],axis=0)
print('data_all:',len(data_all))
# %%

# 提取随机90%作为训练数据，其他为测试数据
# Train_Data = []
# Test_Data = []
def check_data(data):
    data = np.array(data,dtype=np.float64).tolist()
    # for j in range(len(data_all)):
        # if (abs(data[0] - data_train[j][0]) < 0.00005) and \
        #     (abs(data[1] - data_train[j][1]) < 1e-4) and \
        #     (abs(data[2] - data_train[j][2]) < 1e-4) and \
        #     (abs(data[3] - data_train[j][3]) < 1e-4) and \
        #     (abs(data[4] - data_train[j][4]) < 1e-4):
            # with open('./TrainAll_2.txt','a') as f:
            #     f.write(str(data)) 
            #     f.write('\n')
            # Train_Data.append(data)
            # print(f'training add {len(Train_Data)}')
            # return
    with open('./DataAll_2.txt','a') as f:
        f.write(str(data))
        f.write('\n')
    # Test_Data.append([data])
    # print(f'testing add {len(Test_Data)}')

task_list = []
for i,value in tqdm(enumerate(data_all)):
    task_list.append(([value]))
# 用多线程跑
n_proc = max(1, min(len(task_list), int(multiprocessing.cpu_count())))
pool = Pool(n_proc)
pbar = tqdm(total=len(task_list))
res_list = []
for task in task_list:
    res = pool.apply_async(check_data, args=task)
    res_list.append(res)
for res in res_list:
    res.get()
    pbar.update()
print('multiprocess finished!')
pool.close()
pool.join()

# print(f'training_data: {len(Train_Data)}')
# print(f'testing_data: {len(Test_Data)}')
# np.save('./test.npy', np.array(Test_Data))
# np.save('./train.npy', np.array(Train_Data))
# %%
with open('/home/yuqi.zhao1/paper_program/DataAll_2.txt','r') as f:
    content = f.readlines()
lines = [line.replace('\n','') for line in content]
lines = [line.replace('[','') for line in lines]
lines = [line.replace(']','') for line in lines]
lines = [line.split(',') for line in lines]
all_training_data = []
all_training_data = random.sample(lines,48613)
print('train_all:',len(all_training_data))
all_testing_data = []
for i in lines:
    if i not in all_training_data:
        all_testing_data.append(i)
print('test_all:',len(all_testing_data))
all_training_data=np.array(all_training_data,dtype=np.float64)
np.save('./AllTrain_2.npy', all_training_data)
all_testing_data=np.array(all_testing_data,dtype=np.float64)
np.save('./AllTest_2.npy', all_testing_data)  
# %%
highway_test = []
suburb_test = []
city_test = []
for i in range(len(all_testing_data)):
    if all_testing_data[i][5] == 0:
        highway_test.append(all_testing_data[i][5])
    if all_testing_data[i][5] == 1:
        suburb_test.append(all_testing_data[i][5])
    if all_testing_data[i][5] == 2:
        city_test.append(all_testing_data[i][5])
print('highway_test:',len(highway_test))
print('suburb_test:',len(suburb_test))
print('city_test:',len(city_test))
# %%
highway_train = []  
suburb_train = []
city_train = []
for i in range(len(all_training_data)):
    if all_training_data[i][5] == 0:
        highway_train.append(all_training_data[i][5])
    if all_training_data[i][5] == 1:
        suburb_train.append(all_training_data[i][5])
    if all_training_data[i][5] == 2:
        city_train.append(all_training_data[i][5])
print('highway_train:',len(highway_train))
print('suburb_train:',len(suburb_train))
print('city_train:',len(city_train))
# %%

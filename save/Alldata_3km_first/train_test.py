# 时间：2021年12月16日
# 用途：从服务器读取数据，加上0，1，2标签，把训练数据和测试数据从总数据中筛出来（与老师类似数据量）；
# .txt数据来源:VScode转换（立冬知道）
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
# %%
#所有训练数据提取（共35230段）
data_train = np.load('/home/yuqi.zhao1/data/4gear/TrainingSet.npy')
print(len(data_train))
# %%
# 一般公路工况数据提取 1
#速度
suburban_all_speed = np.load("/home/yuqi.zhao1/data/MeanSpeed/suburban_speed.npy")
print(len(suburban_all_speed))
#档位
suburban_all_gear = np.load("/home/yuqi.zhao1/data/Gear/suburban_gear.npy").tolist()
print(len(suburban_all_gear))

suburban_all_data = []
for i in range(len(suburban_all_speed)):
    suburban_all_data.append([suburban_all_speed[i]]+suburban_all_gear[i]+[1])
print("suburban_all_data:",len(suburban_all_data))
# %%
# 城市公路工况数据提取 2
city_all_speed = np.load("/home/yuqi.zhao1/data/MeanSpeed/city_speed.npy")
print(len(city_all_speed))
#档位
city_all_gear = np.load("/home/yuqi.zhao1/data/Gear/city_gear.npy").tolist()
print(len(city_all_gear))

city_all_data = []
for i in range(len(city_all_speed)):
    city_all_data.append([city_all_speed[i]]+city_all_gear[i]+[2])
print("city_all_data:",len(city_all_data))
# %%
# 高速公路工况数据提取 0
#速度
highway_all_speed = np.load("/home/yuqi.zhao1/data/MeanSpeed/highway_speed.npy")
print(len(highway_all_speed))
#档位
highway_all_gear = np.load("/home/yuqi.zhao1/data/Gear/highway_gear.npy").tolist()
print(len(highway_all_gear))

highway_all_data = []
for i in range(len(highway_all_speed)):
    highway_all_data.append([highway_all_speed[i]]+highway_all_gear[i]+[0])
print("highway_all_data:",len(highway_all_data))
# %%
data_all = np.concatenate([highway_all_data,suburban_all_data,city_all_data],axis=0)
print(len(data_all))
# %%

#洗数据：把训练数据从全部数据中抽取出来，找到测试数据 
def check_data(data):
    data = np.array(data,dtype=np.float64).tolist()
    for j in range(len(data_train)):
        if (abs(data[0] - data_train[j][0]) < 0.00005) and \
            (abs(data[1] - data_train[j][1]) < 1e-4) and \
            (abs(data[2] - data_train[j][2]) < 1e-4) and \
            (abs(data[3] - data_train[j][3]) < 1e-4) and \
            (abs(data[4] - data_train[j][4]) < 1e-4):
            with open('/home/yuqi.zhao1/paper_program/save/Alldata_3km_first/TrainAll.txt','a') as f:
                f.write(str(data)) 
                f.write('\n')
            return
    with open('/home/yuqi.zhao1/paper_program/save/Alldata_3km_first/TestAll.txt','a') as f:
        f.write(str(data))
        f.write('\n')

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
with open('/home/yuqi.zhao1/paper_program/save/Alldata_3km_first/TrainAll.txt','r') as f:
    content = f.readlines()
lines = [line.replace('\n','') for line in content]
lines = [line.replace('[','') for line in lines]
lines = [line.replace(']','') for line in lines]
lines = [line.split(',') for line in lines]
all_training_data = []
for line in lines:
    t_data = []
    for num in line:
        t_data.append(float(num))
    all_training_data.append(t_data)
print(len(all_training_data))
np.save('/home/yuqi.zhao1/paper_program/save/Alldata_3km_first/AllTrain.npy', np.array(all_training_data))
# %%
with open('/home/yuqi.zhao1/paper_program/save/Alldata_3km_first/TestAll.txt','r') as f:
    content = f.readlines()
lines = [line.replace('\n','') for line in content]
lines = [line.replace('[','') for line in lines]
lines = [line.replace(']','') for line in lines]
lines = [line.split(',') for line in lines]
all_testing_data = []
for line in lines:
    t_data = []
    for num in line:
        t_data.append(float(num))
    all_testing_data.append(t_data)
print('all_testing_data:',len(all_testing_data))
np.save('/home/yuqi.zhao1/paper_program/save/Alldata_3km_first/AllTest.npy', np.array(all_testing_data))
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

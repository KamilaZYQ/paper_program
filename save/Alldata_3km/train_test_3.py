# 时间：2022年1月6日
# 用途：从服务器读取数据，加上0，1，2标签，把训练数据和测试数据从总数据中筛出来；
# 目的：和train_test.py的结果比较，若无太大差别则为相同数据
# 修改时间：2022年1月11日，加上加速踏板、刹车开关、离合开关的数据
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
# 城市公路工况数据提取 2
#速度
city_all_speed = np.load("/home/yuqi.zhao1/Data/speed/city_speed_3.npy")
print(len(city_all_speed))
#档位
city_all_gear = np.load("/home/yuqi.zhao1/Data/gear/city_gear_3.npy").tolist()
print(len(city_all_gear))
#加速踏板位置
city_all_accel = np.load("/home/yuqi.zhao1/Data/accel/city_accel_3.npy").tolist()
print(len(city_all_accel))
#刹车开关
city_all_brake = np.load("/home/yuqi.zhao1/Data/brake/city_brake_3.npy").tolist()
print(len(city_all_brake))
#离合器开关
city_all_clutch = np.load("/home/yuqi.zhao1/Data/clutch/city_clutch_3.npy").tolist()
print(len(city_all_clutch))

city_all_data = []
for i in range(len(city_all_speed)):
    city_all_data.append([city_all_speed[i]]+city_all_gear[i]+[city_all_accel[i]]+[city_all_brake[i]]+[city_all_clutch[i]]+[2])
print("city_all_data:",len(city_all_data))
# %%
# 一般公路工况数据提取 1
#速度
suburb_all_speed = np.load("/home/yuqi.zhao1/Data/speed/suburb_speed_3.npy")
print(len(suburb_all_speed))
#档位
suburb_all_gear = np.load("/home/yuqi.zhao1/Data/gear/suburb_gear_3.npy").tolist()
print(len(suburb_all_gear))
#加速踏板位置
suburb_all_accel = np.load("/home/yuqi.zhao1/Data/accel/sub_accel_3.npy").tolist()
print(len(suburb_all_accel))
#刹车开关
suburb_all_brake = np.load("/home/yuqi.zhao1/Data/brake/sub_brake_3.npy").tolist()
print(len(suburb_all_brake))
#离合器开关
suburb_all_clutch = np.load("/home/yuqi.zhao1/Data/clutch/sub_clutch_3.npy").tolist()
print(len(suburb_all_clutch))

suburb_all_data = []
for i in range(len(suburb_all_speed)):
    suburb_all_data.append([suburb_all_speed[i]]+suburb_all_gear[i]+[suburb_all_accel[i]]+[suburb_all_brake[i]]+[suburb_all_clutch[i]]+[1])
print("suburb_all_data:",len(suburb_all_data))
# %%
# 高速公路工况数据提取 0
#速度
highway_all_speed = np.load("/home/yuqi.zhao1/Data/speed/highway_speed_3.npy")
print(len(highway_all_speed))
#档位
highway_all_gear = np.load("/home/yuqi.zhao1/Data/gear/highway_gear_3.npy").tolist()
print(len(highway_all_gear))
#加速踏板位置
highway_all_accel = np.load("/home/yuqi.zhao1/Data/accel/high_accel_3.npy").tolist()
print(len(highway_all_accel))
#刹车开关
highway_all_brake = np.load("/home/yuqi.zhao1/Data/brake/high_brake_3.npy").tolist()
print(len(highway_all_brake))
#离合器开关
highway_all_clutch = np.load("/home/yuqi.zhao1/Data/clutch/high_clutch_3.npy").tolist()
print(len(highway_all_clutch))

highway_all_data = []
for i in range(len(highway_all_speed)):
    highway_all_data.append([highway_all_speed[i]]+highway_all_gear[i]+[highway_all_accel[i]]+[highway_all_brake[i]]+[highway_all_clutch[i]]+[0])
print("highway_all_data:",len(highway_all_data))
# %%
data_all = np.concatenate([highway_all_data,suburb_all_data,city_all_data],axis=0)
print('data_all:',len(data_all))
# %%

#洗数据：把训练数据从全部数据中抽取出来，找到测试数据 
# Train_Data = []
# Test_Data = []
def check_data(data):
    data = np.array(data,dtype=np.float64).tolist()
    for j in range(len(data_train)):
        if (abs(data[0] - data_train[j][0]) < 0.00005) and \
            (abs(data[1] - data_train[j][1]) < 1e-4) and \
            (abs(data[2] - data_train[j][2]) < 1e-4) and \
            (abs(data[3] - data_train[j][3]) < 1e-4) and \
            (abs(data[4] - data_train[j][4]) < 1e-4):
            with open('/home/yuqi.zhao1/paper_program/save/Alldata_3km/TrainAll.txt','a') as f:
                f.write(str(data)) 
                f.write('\n')
            # Train_Data.append(data)
            # print(f'training add {len(Train_Data)}')
            return
    with open('/home/yuqi.zhao1/paper_program/save/Alldata_3km/TestAll.txt','a') as f:
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
with open('/home/yuqi.zhao1/paper_program/save/Alldata_3km/TrainAll.txt','r') as f:
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
np.save('/home/yuqi.zhao1/paper_program/save/Alldata_3km/AllTrain.npy', np.array(all_training_data))
# %%
with open('/home/yuqi.zhao1/paper_program/save/Alldata_3km/TestAll.txt','r') as f:
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
np.save('/home/yuqi.zhao1/paper_program/save/Alldata_3km/AllTest.npy', np.array(all_testing_data))
# %%
highway_test = []
suburb_test = []
city_test = []
for i in range(len(all_testing_data)):
    if all_testing_data[i][8] == 0:
        highway_test.append(all_testing_data[i][5])
    if all_testing_data[i][8] == 1:
        suburb_test.append(all_testing_data[i][5])
    if all_testing_data[i][8] == 2:
        city_test.append(all_testing_data[i][5])
print('highway_test:',len(highway_test))
print('suburb_test:',len(suburb_test))
print('city_test:',len(city_test))
# %%
highway_train = []  
suburb_train = []
city_train = []
for i in range(len(all_training_data)):
    if all_training_data[i][8] == 0:
        highway_train.append(all_training_data[i][5])
    if all_training_data[i][8] == 1:
        suburb_train.append(all_training_data[i][5])
    if all_training_data[i][8] == 2:
        city_train.append(all_training_data[i][5])
print('highway_train:',len(highway_train))
print('suburb_train:',len(suburb_train))
print('city_train:',len(city_train))
# %%

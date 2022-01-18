# %%
import numpy as np
import pandas as pd
from pandas import DataFrame as df
import matplotlib as mlb
import matplotlib.pyplot as plt
# %%
data = np.load('/home/yuqi.zhao1/paper_program/save/Alldata_3km/test_data_check.npy',allow_pickle=True)
data_correct = np.array(data[-1][0])
data_wrong = np.array(data[-1][1])
data_corrct_df = pd.DataFrame(data_correct)
data_wrong_df = pd.DataFrame(data_wrong)
writer = pd.ExcelWriter('/home/yuqi.zhao1/paper_program/save/Alldata_3km/epoch_100_data.xlsx')  #关键2，创建名称为epoch_100_data的excel表格
data_corrct_df.to_excel(writer,'page_1',float_format='%.2f')  #关键3，float_format 控制精度，将data_df写到epoch_100_data表格的第一页中。
data_wrong_df.to_excel(writer,'page_2',float_format='%.2f')  #关键3，float_format 控制精度，将data_df写到epoch_100_data表格的第二页中。
# writer.save()
# %%
# %% 
# 平均速度箱线图（归一化）
y0 = pd.Series(np.array(data_correct[:,0]))
y1 = pd.Series(np.array(data_wrong[:,0]))
Data = pd.DataFrame({"Correct":y0,"Wrong":y1})
Data.boxplot()
plt.ylabel("Suburb Speed")
plt.grid(linestyle="--",alpha=0.3)
plt.show
# %%
# 怠速档位比例
y0 = pd.Series(np.array(data_correct[:,1]))
y1 = pd.Series(np.array(data_wrong[:,1]))
Data = pd.DataFrame({"Correct":y0,"Wrong":y1})
Data.boxplot()
plt.ylabel("Suburb Gear 1")
plt.grid(linestyle="--",alpha=0.3)
plt.show
# %%
# 低速档位比例
y0 = pd.Series(np.array(data_correct[:,2]))
y1 = pd.Series(np.array(data_wrong[:,2]))
Data = pd.DataFrame({"Correct":y0,"Wrong":y1})
Data.boxplot()
plt.ylabel("Suburb Gear 2")
plt.grid(linestyle="--",alpha=0.3)
plt.show
# %%
# 中速档位比例
y0 = pd.Series(np.array(data_correct[:,3]))
y1 = pd.Series(np.array(data_wrong[:,3]))
Data = pd.DataFrame({"Correct":y0,"Wrong":y1})
Data.boxplot()
plt.ylabel("Suburb Gear 3")
plt.grid(linestyle="--",alpha=0.3)
plt.show
# %%
# 高速档位比例
y0 = pd.Series(np.array(data_correct[:,4]))
y1 = pd.Series(np.array(data_wrong[:,4]))
Data = pd.DataFrame({"Correct":y0,"Wrong":y1})
Data.boxplot()
plt.ylabel("Suburb Gear 4")
plt.grid(linestyle="--",alpha=0.3)
plt.show
# %%
# 加速踏板位置（未归一化）
y0 = pd.Series(np.array(data_correct[:,5]))
y1 = pd.Series(np.array(data_wrong[:,5]))
Data = pd.DataFrame({"Correct":y0,"Wrong":y1})
Data.boxplot()
plt.ylabel("Suburb accel pos")
plt.grid(linestyle="--",alpha=0.3)
plt.show
# %%
# 刹车开关
y0 = pd.Series(np.array(data_correct[:,6]))
y1 = pd.Series(np.array(data_wrong[:,6]))
Data = pd.DataFrame({"Correct":y0,"Wrong":y1})
Data.boxplot()
plt.ylabel("Suburb Brake Switch")
plt.grid(linestyle="--",alpha=0.3)
plt.show
# %%
# 离合器开关
y0 = pd.Series(np.array(data_correct[:,7]))
y1 = pd.Series(np.array(data_wrong[:,7]))
Data = pd.DataFrame({"Correct":y0,"Wrong":y1})
Data.boxplot()
plt.ylabel("Suburb Clutch Switch")
plt.grid(linestyle="--",alpha=0.3)
plt.show
# %%

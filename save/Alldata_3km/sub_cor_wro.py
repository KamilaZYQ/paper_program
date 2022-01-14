# %%
import numpy as np
import pandas as pd
data = np.load('/home/yuqi.zhao1/paper_program/save/Alldata_3km/test_data_check.npy',allow_pickle=True)
data_correct = np.array(data[-1][0])
data_wrong = np.array(data[-1][1])
data_corrct_df = pd.DataFrame(data_correct)
data_wrong_df = pd.DataFrame(data_wrong)
writer = pd.ExcelWriter('/home/yuqi.zhao1/paper_program/save/Alldata_3km/epoch_100_data.xlsx')  #关键2，创建名称为hhh的excel表格
data_corrct_df.to_excel(writer,'page_1',float_format='%.2f')  #关键3，float_format 控制精度，将data_df写到hhh表格的第一页中。若多个文件，可以在page_2中写入
data_wrong_df.to_excel(writer,'page_2',float_format='%.2f')  #关键3，float_format 控制精度，将data_df写到hhh表格的第一页中。若多个文件，可以在page_2中写入
writer.save()
# %%

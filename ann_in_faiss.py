# -*- coding: utf-8 -*-
import time 
import faiss 
import numpy as np
import pandas as pd
import sys
import os
sys.path.append(os.getenv('HIVE_TASK'))
from HiveTask import HiveTask
ht = HiveTask()

ngpus = faiss.get_num_gpus()
reader = pd.read_table('content_1562726596.txt', sep='\t',names=columns_name, iterator=True)
chunk =  reader.get_chunk(10) 

loop = True
chunkSize = 100000
chunks = []
while loop:
    try:
        chunk = reader.get_chunk(chunkSize)
        chunks.append(chunk)
    except StopIteration:
        loop = False
        print("Iteration is stopped.")
data = pd.concat(chunks, ignore_index=True)
print(len(chunks))

user_log_acct = data['user_log_acct']
user_id2i = {pin: i for i, pin in enumerate(user_log_acct)}
i2user_id = {i: pin for i, pin in enumerate(user_log_acct)}
#i2user_id.get(62903)
#pin_data_dropid = data.drop('user_log_acct',axis=1)
#pin_data_dropid = data.iloc[:,1:]
pin_data_dropid = data.copy()
del pin_data_dropid['user_log_acct']
d = pin_data_dropid.shape[1]

from sklearn.decomposition import PCA 
estimator = PCA(n_components=60)   # 使用PCA将原64维度图像压缩为20个维度
pca_pin_data_drop = estimator.fit_transform(pin_data_dropid)   # 利用训练特征决定20个正交维度的方向，并转化原训练特征

pin_data_drop = np.array(pin_data_dropid).astype('float32')  #这里量会限制
#pin_data_drop[:10]
d = len(pca_pin_data_drop[0])

pin_data_drop_new = np.ascontiguousarray(pca_pin_data_drop)


nlist = 256
m = 3   # PQ才有 列方向划分个数，必须能被d整除,  特征向量分组,这个很影响速率
k = 10
res = [faiss.StandardGpuResources() for i in range(ngpus)]
# first we get StandardGpuResources of each GPU
# ngpu is the num of GPUs
flat_config = []
for i in range(ngpus): 
    cfg = faiss.GpuIndexIVFFlatConfig() #faiss.GpuIndexFlatConfig()  faiss.GpuIndexIVFPQConfig()
    cfg.useFloat16 = False
    cfg.device = i
    flat_config.append(cfg)

#indexes = [faiss.GpuIndexFlatL2(res[i],d,flat_config[i]) for i in range(ngpus)]    #可行，速度快，不需要train，直接计算L2距离
#indexes = [faiss.GpuIndexIVFPQ(res[i],d,nlist, m,4,faiss.METRIC_L2,flat_config[i]) for i in range(ngpus)]
indexes = [faiss.GpuIndexIVFFlat(res[i],d,nlist,faiss.METRIC_L2,flat_config[i]) for i in range(ngpus)]
# then we make an Index array
# useFloat16 is a boolean value

index = faiss.IndexProxy()

for sub_index in indexes:
    index.addIndex(sub_index)
    
index.train(pin_data_drop_new) #影响PQ的时间的因素？？？
print(index.is_trained)

index.add(pin_data_drop_new)
index.nprobe = 30  #参数需要调，适当增加nprobe可以得到和brute-force相同的结果，nprobe控制了速度和精度的平衡
print(index.ntotal)  #index中向量的个数

beg = time.time()
query_self = pin_data_drop_new[:200000]  # 查询本身
dis, ind = index.search(query_self, k)     #查询自身
print(time.time()-beg)
print(dis)
print(ind)

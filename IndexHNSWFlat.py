import faiss
import pandas as pd
import time
import numpy as np
import sys
import math

#混合索引，使用了上述多种索引方法以增强查询性能
def train_coarse_quantizer(data, quantizer_path, num_clusters, hnsw=False,PQ=False,niter=10, cuda=False):
    d = data.shape[1]

    index_flat = faiss.IndexFlatL2(d)  #初始化量化器
    # make it into a gpu index
    if cuda:
        res = faiss.StandardGpuResources()
        index_flat = faiss.index_cpu_to_gpu(res, 1, index_flat)
    clus = faiss.Clustering(d, num_clusters)
    clus.verbose = True
    clus.niter = niter
    clus.train(data, index_flat)
    centroids = faiss.vector_float_to_array(clus.centroids)
    centroids = centroids.reshape(num_clusters, d)
    print("centroids",centroids.shape)

    if hnsw:
        quantizer = faiss.IndexHNSWFlat(d, 32) #初始化索引
        quantizer.hnsw.efSearch = 128
        quantizer.train(centroids) #训练，该索引会对向量进行聚类来加快查询速度。聚类搜索需要先进行训练，相当于无监督学习。
        quantizer.add(centroids) #将向量集添加进索引
    else:
        quantizer = faiss.IndexFlatL2(d)
        if PQ:
            nlist = 10 #聚类中心个数《=》 设置向量集中簇的个数，centroids
            m = 8
            quantizer = faiss.IndexIVFPQ(quantizer,d,nlist,m,8) #m:code_size
            print (quantizer.is_trained),"@@"
            quantizer.train(centroids)
        quantizer.add(centroids)

    faiss.write_index(quantizer, quantizer_path)

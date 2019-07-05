# 7.4:动手实现k-均值

```python
#encoding=utf8
import numpy as np

# 计算一个样本与数据集中所有样本的欧氏距离的平方
def euclidean_distance(one_sample, X):
    '''
    input:
        one_sample(ndarray):单个样本
        X(ndarray):所有样本
    output:
        distances(ndarray):单个样本到所有样本的欧氏距离平方
    '''
    one_sample = one_sample.reshape(1, -1)
    distances = np.power(np.tile(one_sample, (X.shape[0], 1)) - X, 2).sum(axis=1)
    return distances

# 从所有样本中随机选取k个样本作为初始的聚类中心
def init_random_centroids(k,X):
    '''
    input:
        k(int):聚类簇的个数
        X(ndarray):所有样本
    output:
        centroids(ndarray):k个簇的聚类中心
    '''
    n_samples, n_features = np.shape(X)
    centroids = np.zeros((k, n_features))
    for i in range(k):
        centroid = X[np.random.choice(range(n_samples))]
        centroids[i] = centroid
    return centroids

# 返回距离该样本最近的一个中心索引[0, k)
def _closest_centroid(sample, centroids):
    '''
    input:
        sample(ndarray):单个样本
        centroids(ndarray):k个簇的聚类中心
    output:
        closest_i(int):最近中心的索引
    '''
    distances = euclidean_distance(sample, centroids)
    closest_i = np.argmin(distances)
    return closest_i

# 将所有样本进行归类，归类规则就是将该样本归类到与其最近的中心
def create_clusters(k,centroids, X):
    '''
    input:
        k(int):聚类簇的个数
        centroids(ndarray):k个簇的聚类中心
        X(ndarray):所有样本
    output:
        clusters(list):列表中有k个元素，每个元素保存相同簇的样本的索引
    '''
    clusters = [[] for _ in range(k)]
    for sample_i, sample in enumerate(X):
        centroid_i = _closest_centroid(sample, centroids)
        clusters[centroid_i].append(sample_i)
    return clusters

# 对中心进行更新
def update_centroids(k,clusters, X):
    '''
    input:
        k(int):聚类簇的个数
        X(ndarray):所有样本
    output:
        centroids(ndarray):k个簇的聚类中心
    '''
    n_features = np.shape(X)[1]
    centroids = np.zeros((k, n_features))
    for i, cluster in enumerate(clusters):
        centroid = np.mean(X[cluster], axis=0)
        centroids[i] = centroid
    return centroids

# 将所有样本进行归类，其所在的类别的索引就是其类别标签
def get_cluster_labels(clusters, X):
    '''
    input:
        clusters(list):列表中有k个元素，每个元素保存相同簇的样本的索引
        X(ndarray):所有样本
    output:
        y_pred(ndarray):所有样本的类别标签
    '''
    y_pred = np.zeros(np.shape(X)[0])
    for cluster_i, cluster in enumerate(clusters):
        for sample_i in cluster:
            y_pred[sample_i] = cluster_i
    return y_pred

# 对整个数据集X进行Kmeans聚类，返回其聚类的标签
def predict(k,X,max_iterations,varepsilon):
    '''
    input:
        k(int):聚类簇的个数
        X(ndarray):所有样本
        max_iterations(int):最大训练轮数
        varepsilon(float):最小误差阈值
    output:
        y_pred(ndarray):所有样本的类别标签
    '''
    # 从所有样本中随机选取k样本作为初始的聚类中心
    centroids = init_random_centroids(k,X)
    # 迭代，直到算法收敛(上一次的聚类中心和这一次的聚类中心几乎重合)或者达到最大迭代次数
    for _ in range(max_iterations):
        # 将所有进行归类，归类规则就是将该样本归类到与其最近的中心
        clusters = create_clusters(k,centroids, X)
        former_centroids = centroids
        # 计算新的聚类中心
        centroids = update_centroids(k,clusters, X)
        # 如果聚类中心几乎没有变化，说明算法已经收敛，退出迭代
        diff = centroids - former_centroids
        if diff.any() < varepsilon:
            break
    y_pred = get_cluster_labels(clusters, X)
    return y_pred
```


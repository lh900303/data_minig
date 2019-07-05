# 4.4:动手实现k-近邻

`knn`算法实现`python`代码如下：

```python
#encoding=utf8
import numpy as np

def knn_clf(k,train_feature,train_label,test_feature):
    '''
    input:
        k(int):最近邻样本个数
        train_feature(ndarray):训练样本特征
        train_label(ndarray):训练样本标签
        test_feature(ndarray):测试样本特征
    output:
        predict(ndarray):测试样本预测标签
    '''
    #初始化预测结果
    predict = np.zeros(test_feature.shape[0],).astype('int')
    #对测试集每一个样本进行遍历
    for i in range(test_feature.shape[0]):
        #测试集第i个样本到训练集每一个样本的距离
        distance = np.sqrt(np.power(np.tile(test_feature[i],(train_feature.shape[0],1))-train_feature,2).sum(axis=1))
        #最近的k个样本的距离
        distance_k = np.sort(distance)[:k]
        #最近的k个样本的索引
        nearest = np.argsort(distance)[:k]
        #最近的k个样本的标签
        topK = [train_label[i] for i in nearest]
        #初始化进行投票的字典，字典的键为标签，值为投票分数
        votes = {}
        #初始化最大票数
        max_count = 0
        #进行投票
        for j,label in enumerate(topK):
            #如果标签在字典的键中则投票计分
            if label in votes.keys():
                votes[label] += 1/(distance_k[j]+1e-10)#防止分母为0
                #如果评分最高则将预测值更新为对应标签
                if votes[label] > max_count:
                    max_count = votes[label]
                    predict[i] = label
            #如果标签不在字典中则将标签加入字典的键，同时计入相应的分数
            else:
                votes[label] = 1/(distance_k[j]+1e-10)
                if votes[label] > max_count:
                    max_count = votes[label]
                    predict[i] = label
    return predict
```


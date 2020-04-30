# 模型 - KNN

---

[TOC]

## 简介

KNN算法是分类算法的一种，也属于监督学习算法，其基本思想为：

- 当输入一个新的样本时，将新数据的每个特征与样本集中每个样本数据的特征进行比较。

- 从样本集中选取最**相近**的 K 个样本，然后依据某种决策原则（少数服从多数）来判定这个新样本的 label。

在KNN算法中，有三个主要要素：距离度量，K 的取值，分类决策规则。

邻近算法，或者说K最近邻(kNN，k-NearestNeighbor)分类算法是数据挖掘分类技术中最简单的方法之一。所谓K最近邻，就是k个最近的邻居的意思，说的是每个样本都可以用它最接近的k个邻居来代表。Cover和Hart在1968年提出了最初的邻近算法。KNN是一种分类(classification)算法，它输入基于实例的学习（instance-based learning），属于懒惰学习（lazy learning）即KNN没有显式的学习过程，也就是说没有训练阶段，数据集事先已有了分类和特征值，待收到新样本后直接进行处理。与急切学习（eager learning）相对应。

　　KNN是通过测量不同特征值之间的距离进行分类。 

　　思路是：如果一个样本在特征空间中的k个最邻近的样本中的大多数属于某一个类别，则该样本也划分为这个类别。KNN算法中，所选择的邻居都是已经正确分类的对象。该方法在定类决策上只依据最邻近的一个或者几个样本的类别来决定待分样本所属的类别。



## 算法步骤

- 计算新数据与数据集中所有样本之间的距离， 距离度量公式可以选择多种，如欧式距离，曼哈顿距离等。可以参见：[基础理论 - 距离度量方法](./基础理论 - 距离度量方法.md)
- 按照距离，将样本按照距离值进行排序

- 选取与当前数据距离最小的 K 个点。
- 返回这 K 个点的 label， 依据某种决策原则来判定这个新数据的 label 。



## K 的选择

- 如果 K 较小，预测结果会对邻近的点十分敏感，如果邻近点恰好是噪声，则很容易预测错误。 换个角度说说， **K的减小意味着模型变得复杂，容易发生过拟合。**
- 如果K 较大， 此时不相似的样本也会对预测产生作用，使得预测发生错误。换个角度说，**K的增大意味着模型变得简单，容易发生欠拟合。**
  常用的方法是从k=1开始，使用检验集估计分类器的误差率。重复该过程，每次K增值1，允许增加一个近邻。选取产生最小误差率的K。
  一般k的取值不超过20，上限是n的开方，随着数据集的增大，K的值也要增大。

---

python代码实现
```python
#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: JYRoooy
import csv
import random
import math
import operator

# 加载数据集
def loadDataset(filename, split, trainingSet = [], testSet = []):
    with open(filename, 'r') as csvfile:
        lines = csv.reader(csvfile)
        dataset = list(lines)
        for x in range(len(dataset)-1):
            for y in range(4):
                dataset[x][y] = float(dataset[x][y])
            if random.random() < split:  #将数据集随机划分
                trainingSet.append(dataset[x])
            else:
                testSet.append(dataset[x])

# 计算点之间的距离，多维度的
def euclideanDistance(instance1, instance2, length):
    distance = 0
    for x in range(length):
        distance += pow((instance1[x]-instance2[x]), 2)
    return math.sqrt(distance)

# 获取k个邻居
def getNeighbors(trainingSet, testInstance, k):
    distances = []
    length = len(testInstance)-1
    for x in range(len(trainingSet)):
        dist = euclideanDistance(testInstance, trainingSet[x], length)
        distances.append((trainingSet[x], dist))   #获取到测试点到其他点的距离
    distances.sort(key=operator.itemgetter(1))    #对所有的距离进行排序
    neighbors = []
    for x in range(k):   #获取到距离最近的k个点
        neighbors.append(distances[x][0])
        return neighbors

# 得到这k个邻居的分类中最多的那一类
def getResponse(neighbors):
    classVotes = {}
    for x in range(len(neighbors)):
        response = neighbors[x][-1]
        if response in classVotes:
            classVotes[response] += 1
        else:
            classVotes[response] = 1
    sortedVotes = sorted(classVotes.items(), key=operator.itemgetter(1), reverse=True)
    return sortedVotes[0][0]   #获取到票数最多的类别

#计算预测的准确率
def getAccuracy(testSet, predictions):
    correct = 0
    for x in range(len(testSet)):
        if testSet[x][-1] == predictions[x]:
            correct += 1
    return (correct/float(len(testSet)))*100.0


def main():
    #prepare data
    trainingSet = []
    testSet = []
    split = 0.67
    loadDataset(r'irisdata.txt', split, trainingSet, testSet)
    print('Trainset: ' + repr(len(trainingSet)))
    print('Testset: ' + repr(len(testSet)))
    #generate predictions
    predictions = []
    k = 3
    for x in range(len(testSet)):
        # trainingsettrainingSet[x]
        neighbors = getNeighbors(trainingSet, testSet[x], k)
        result = getResponse(neighbors)
        predictions.append(result)
        print ('predicted=' + repr(result) + ', actual=' + repr(testSet[x][-1]))
    print('predictions: ' + repr(predictions))
    accuracy = getAccuracy(testSet, predictions)
    print('Accuracy: ' + repr(accuracy) + '%')

if __name__ == '__main__':
    main()
 ```


## QA

### 1. KNN 中为何采用欧式距离而不采用曼哈顿距离？

我们不用曼哈顿距离，因为它只计算水平或垂直距离，有维度的限制。另一方面，欧式距离可用于任何空间的距离计算问题。因为，数据点可以存在于任何空间，欧氏距离是更可行的选择。例如：想象一下国际象棋棋盘，象或车所做的移动是由曼哈顿距离计算的，因为它们是在各自的水平和垂直方向的运动。

### 2. 


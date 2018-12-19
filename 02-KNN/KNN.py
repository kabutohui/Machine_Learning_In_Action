'''
Created on Sep 16, 2010
kNN: k Nearest Neighbors

Input:      inX: vector to compare to existing dataset (1xN)
            dataSet: size m data set of known vectors (NxM)
            labels: data set labels (1xM vector)
            k: number of neighbors to use for comparison (should be an odd number)

Output:     the most popular class label

@author: pbharrin

---------------------------
@modified: Kabuto_hui
@date: 2018/12/19
---------------------------
'''
from numpy import *
import operator
from os import listdir


def classify0(inX, dataSet, labels, k):
    '''
    knn分类器
    :param inX:         待分类向量
    :param dataSet:     数据集
    :param labels:      数据集对应的标签
    :param k:           邻居个数
    :return:            返回分类的类别
    '''
    # 获取训练集的样本数量
    dataSetSize = dataSet.shape[0]
    # 先在列方向上重复待分类向量dataSetSize次，再减去训练集；其实就是待分类向量与训练集中的每个向量相减
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet
    # 差值的平方
    sqDiffMat = diffMat ** 2
    # 差值的平方和
    sqDistances = sqDiffMat.sum(axis=1)
    # 差值平方和再开方
    distances = sqDistances ** 0.5
    # 将距离从小到大排序并返回index
    sortedDistIndicies = distances.argsort()
    classCount = {}
    # 遍历与待测样本距离最近的k个训练集样本，选择数量最多的label作为返回值
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1  # 统计最近的k个训练集样本中各个label的数量
    # sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True) # 对dict按value值由大到小排序
    return sortedClassCount[0][0]   # 返回value值最大key作为返回，即为label


def file2matrix(filename):
    '''
    读取文件中的数据，并返回数据集及其对应的label
    :param filename:    文件名称
    :return:            返回样本集合及其label
    '''
    fr = open(filename)
    # 获取行数【样本个数】
    numberOfLines = len(fr.readlines())  # get the number of lines in the file
    returnMat = zeros((numberOfLines, 3))  # prepare matrix to return
    classLabelVector = []  # prepare labels return
    index = 0
    # 按行读取
    fr = open(filename)
    for line in fr.readlines():
        line = line.strip()    # 去掉空格
        listFromLine = line.split('\t')         # 以\t作为分割
        returnMat[index, :] = listFromLine[0:3] # 取前三列作为特征
        classLabelVector.append(int(listFromLine[-1]))  # 取最后一列作为label
        index += 1
    return returnMat, classLabelVector


def autoNorm(dataSet):
    '''
    归一化特征值x' = (x - x_min) / (x_max - x_min)
    :param dataSet: 数据集
    :return:        归一化后的数据集， 最大减最小值， 最小值
    '''
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet))
    # 获取样本的个数
    m = dataSet.shape[0]
    # 对于数据集中的每个数据进行处理： x' = (x - x_min) / (x_max - x_min)
    normDataSet = dataSet - tile(minVals, (m, 1))
    normDataSet = normDataSet / tile(ranges, (m, 1))  # element wise divide
    return normDataSet, ranges, minVals


def datingClassTest():
    '''
    针对于约会网站的测试代码
    '''
    # 测试集的比例
    hoRatio = 0.10  # hold out 10%
    # 获取数据
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')  # load data setfrom file
    # 归一化
    normMat, ranges, minVals = autoNorm(datingDataMat)
    # 获取数据集中的样本数量
    m = normMat.shape[0]
    # 获取测试集数量
    numTestVecs = int(m * hoRatio)
    # 初始化误差
    errorCount = 0.0
    for i in range(numTestVecs):
        # 对测试集中的每一个样本进行分类，k=3
        classifierResult = classify0(normMat[i, :], normMat[numTestVecs:m, :], datingLabels[numTestVecs:m], 3)
        print("the classifier came back with: %d, the real answer is: %d" % (classifierResult, datingLabels[i]))
        # 如果分类错误，错误的个数+1
        if (classifierResult != datingLabels[i]): errorCount += 1.0
    print("the total error rate is: %f" % (errorCount / float(numTestVecs)))
    print(errorCount)


def img2vector(filename):
    '''
    图片转向量
    Note： 书中所给的图片都转化成只有01的32*32矩阵，这个函数将矩阵转化为向量
    :param filename:    文件名
    :return:            返回向量
    '''
    returnVect = zeros((1, 1024))
    fr = open(filename)
    for i in range(32):
        # 读取一行
        lineStr = fr.readline()
        for j in range(32):
            # 对一行中的每个数据转化为int型再存入向量中
            returnVect[0, 32 * i + j] = int(lineStr[j])
    return returnVect


def handwritingClassTest():
    '''
    针对于手写识别实验的测试代码
    '''
    hwLabels = []
    # 读取文件夹中的文件列表
    trainingFileList = listdir('trainingDigits')  # load the training set
    # 获取文件的个数
    m = len(trainingFileList)
    # 初始化训练集
    trainingMat = zeros((m, 1024))
    # 对每个文件进行处理
    for i in range(m):
        # 获取文件名称
        fileNameStr = trainingFileList[i]
        # 将文件名切割以获取label： 如0_0.txt
        fileStr = fileNameStr.split('.')[0]  # take off .txt
        classNumStr = int(fileStr.split('_')[0])
        # 存储该图片的label
        hwLabels.append(classNumStr)
        # 图片矩阵转化为向量存入训练集中
        trainingMat[i, :] = img2vector('trainingDigits/%s' % fileNameStr)

    # 获取测试集的文件列表
    testFileList = listdir('testDigits')  # iterate through the test set
    # 初始化错分的个数
    errorCount = 0.0
    # 获取测试集的数量
    mTest = len(testFileList)
    for i in range(mTest):
        # 获取该测试样本的label
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]  # take off .txt
        classNumStr = int(fileStr.split('_')[0])
        # 将该测试样本转化为向量
        vectorUnderTest = img2vector('testDigits/%s' % fileNameStr)
        # 调用knn进行分类
        classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)
        print("the classifier came back with: %d, the real answer is: %d" % (classifierResult, classNumStr))
        # 统计错分的个数
        if (classifierResult != classNumStr): errorCount += 1.0
    print("\nthe total number of errors is: %d" % errorCount)
    print("\nthe total error rate is: %f" % (errorCount / float(mTest)))

if __name__ == '__main__':
    print('-'*15, '约会网站实验的测试代码', '-'*15)
    datingClassTest()
    print('-' * 15, '手写识别实验的测试代码', '-' * 15)
    handwritingClassTest()
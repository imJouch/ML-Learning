from numpy import *
import operator
import loadmnist


def knn(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat ** 2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances ** 0.5
    sortedDistIndicies = distances.argsort()
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    # print sortedClassCount
    return sortedClassCount[0][0]


def handwritingClassTest():
    trainingMat, hwLabels, size = loadmnist.load('/Users/jouch/Downloads/train-images-idx3-ubyte',
                                                 '/Users/jouch/Downloads/train-labels-idx1-ubyte')
    dataUnderTest, classNumStr, size = loadmnist.load('/Users/jouch/Downloads/t10k-images-idx3-ubyte',
                                                      '/Users/jouch/Downloads/t10k-labels-idx1-ubyte')
    errorCount = 0.0
    for i in range(size):
        classifierResult = knn(dataUnderTest[i, :], trainingMat, hwLabels, 3)
        print("第%d个对象分类为: %d, 正确答案: %d, 目前错误数: %d" % (
        i, classifierResult, classNumStr[i], errorCount))
        if (classifierResult != classNumStr[i]): errorCount += + 1.0
    print("\n总错误数: %d" % errorCount)
    print("\n总错误率: %f" % (errorCount / float(size)))
handwritingClassTest()
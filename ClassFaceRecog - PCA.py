# -*- coding:utf8 -*-

from PIL import Image
import numpy as np
from sklearn import svm
from pylab import *
from sklearn import neighbors  
import os
# 调用scikit的朴素贝叶斯算法包,GaussianNB和MultinomialNB
from sklearn.naive_bayes import GaussianNB      # nb for 高斯分布的数据   
from sklearn.naive_bayes import MultinomialNB   # nb for 多项式分布的数据    
    
def loadImage(path, tab):
    '''from path load Image and return ndarray'''
    images = []
    for root, dirs, files in os.walk( path ):
        for x in files:
            impath = path + x 
            im = Image.open(impath)
            data = im.getdata()
            data = np.matrix(data)
            tab.append(x.split("_")[0]) # 添加对应标签
            if images == []:
                images = data
                continue
            images = np.concatenate((images, data))
    print u"原图像矩阵：", images.shape
    return images

def percentage2n(eigVals, percentage):
    '''acquire n by percentage'''
    sortArray = np.sort(eigVals)  # 升序
    sortArray = sortArray[-1::-1] # 降序
    arraySum = sum(sortArray)
    tmpSum = 0
    num = 0
    for i in sortArray:
        tmpSum += i
        num += 1
        if tmpSum >= arraySum * percentage:
            return num

def pca(dataMat, percentage):
    '''Principal Component Analysis'''
    meanVal = np.mean(dataMat, axis = 0)    # 按列求真值，即求各个特征的均值
    newData = dataMat - meanVal             # newData是零均值化后的数据
    covMat = np.cov(newData, rowvar = 0)    # 以行代表样本求协方差矩阵

    U, eigenvalue, vector = np.linalg.svd(covMat , full_matrices=True)
    n = percentage2n(eigenvalue, percentage)# 要达到percent的方差百分比，需要前n个特征向量  
    n_eigVect = vector[: , :n]
    lowDDataMat = newData * n_eigVect       # 低维特征空间的数据
    print u"保留的特征数：", n
    return lowDDataMat, n_eigVect,meanVal
    
def testImg(test_path, n_eigVect, clf):
    true_num = 0    # 匹配成功数
    all_num = 0     # 总图片数
    for root, dirs, files in os.walk(test_path):
        for x in files:
            print u"测试：%16s "%x,
            all_num += 1
            test_im = Image.open(test_path + "\\" + x)
            # 获得测试图片矩阵，降维匹配
            test_data = test_im.getdata()
            test_data = np.matrix(test_data)
            test_data = test_data - meanVal
            test_data = test_data * n_eigVect
            print u"匹配：",clf.predict(test_data),
            if clf.predict(test_data) == x.split('_')[0]:
                true_num += 1
                print u"  ！成功！"
            else:
                print u"  ！失败！"
    print u"共匹配：%3d张"%all_num,
    print u"成功率:%.2f"%(float(true_num)/all_num)
    
    
train_path = r"f:\ORL_face_train\\"   # 训练集图片路径
test_path = r"f:\ORL_face_test"       # 测试集集图片路径
percentage=0.95                         # 方差百分比
tab = []                                # 标签

train_images = loadImage(train_path, tab)   # 获取训练集图片矩阵
lowDDataMat, n_eigVect, meanVal = pca(train_images, percentage)# 主成分分析获得特征向量，低维数据

print "svm："
clf = svm.SVC(kernel='linear')              # 线性核函数
clf.fit(lowDDataMat, tab)                   # 训练，分类
testImg(test_path, n_eigVect, clf)          # 识别测试

print "knn"
clf2 = neighbors.KNeighborsClassifier(algorithm='kd_tree',n_neighbors=1) #knn 
clf2.fit(lowDDataMat, tab)                  # 训练，分类
testImg(test_path, n_eigVect, clf2)          # 识别测试

print 'GaussianNB'
clf3 = GaussianNB()          
clf3.fit(lowDDataMat, tab)
testImg(test_path, n_eigVect, clf3)          # 识别测试

'''print 'MultinomialNB'
clf4 = MultinomialNB(alpha=0.1)#default alpha=1.0,Setting alpha = 1 is called Laplace smoothing, while alpha < 1 is called Lidstone smoothing.       
clf4.fit(lowDDataMat, tab)
testImg(test_path, n_eigVect, clf4)          # 识别测试'''

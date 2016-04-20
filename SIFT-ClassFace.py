
import cv2
import numpy as np
from sklearn.cluster import KMeans
from sklearn import svm
from sklearn import neighbors
from sklearn.naive_bayes import GaussianNB      #nb for 高斯分布的数据 
import os

def acquir_all_descs(path, tab):
    '''from path load Image and acquire all descs'''
    descs = []
    for root, dirs, files in os.walk( path ):
        for x in files:
            impath = path + x 
            im = cv2.imread(impath)
            tab.append(x.split("_")[0]) # 添加对应标签    
            # find the keypoints and descriptors with SIFT
            kp, des = sift.detectAndCompute(im,None)
            if descs == []:
                descs = des
                continue
            descs = np.concatenate((descs, des))
    return descs

def new_features(descriptors, clust):
    '''get new features'''
    features = np.zeros(clust.n_clusters)
    pre = clust.predict(descriptors)
    for i in pre:
        features[i] += 1
    return features

def new_img_data(path, clust):
    '''get new features'''
    new_mat = []
    for root, dirs, files in os.walk( path ):
        for x in files:
            impath = path + x 
            im = cv2.imread(impath)           
            # find the keypoints and descriptors with SIFT
            kp, des = sift.detectAndCompute(im,None)
            feature = new_features(des, clust)
            feature = np.matrix(feature)
            if new_mat == []:
                new_mat = feature
                continue
            new_mat = np.concatenate((new_mat, feature))
    return new_mat

def testImg(test_path, cluster, clf):
    true_num = 0    #匹配成功数
    all_num = 0     #总图片数
    for root, dirs, files in os.walk(test_path):
        for x in files:
            print("测试：%16s "%x,end='')
            all_num += 1
            test_im = cv2.imread(test_path + "\\" + x)
            #获得测试图片描述子，生成新矩阵

            kp, des = sift.detectAndCompute(test_im,None)
            feature = new_features(des, cluster)
            test_data = np.matrix(feature)
            
            print (u"匹配：",clf.predict(test_data),end='')
            if clf.predict(test_data) == x.split('_')[0]:
                true_num += 1
                print("  ！成功！")
            else:
                print("  ！失败！")
    print ("共匹配：%3d张"%all_num,end='')
    print ("成功率:%.2f"%(float(true_num)/all_num))

train_path = r"f:\ORL_face_train\\"   # 训练集图片路径
test_path = r"f:\ORL_face_test"       # 测试集集图片路径
tab = []                              # 标签

# Initiate SIFT detector
sift = cv2.xfeatures2d.SIFT_create()
# 获取训练集图片全部描述子
all_descs = acquir_all_descs(train_path, tab)
# 对描述子聚类
n_clusters = 250
cluster = KMeans(n_clusters = n_clusters).fit(all_descs)
#获得新的特征向量
new_img_mat = new_img_data(train_path, cluster)
#测试
print ("svm：")
clf = svm.SVC(kernel='linear')              #线性核函数
clf.fit(new_img_mat, tab)                   #训练，分类
testImg(test_path, cluster, clf)          #识别测试

print ("knn")
clf2 = neighbors.KNeighborsClassifier(algorithm='kd_tree',n_neighbors=1) #knn 
clf2.fit(new_img_mat, tab)                  #训练，分类
testImg(test_path, cluster, clf2)          #识别测试

print ('GaussianNB')
clf3 = GaussianNB()          
clf3.fit(new_img_mat, tab)
testImg(test_path, cluster, clf3)          #识别测试

import numpy as np
from numpy import linalg as LA
from sklearn.decomposition import PCA
import cv2
import os
import time

def loadImageVector(name):
    img = cv2.imread('./images/'+name, cv2.IMREAD_GRAYSCALE).astype(np.float)
    img = cv2.normalize(img, dst = None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    return img.flatten()

def showDiff(pca, testImage):
    testIm = loadImageVector(testImage)
    Fit = pca.inverse_transform(pca.transform([testIm])).astype(np.float)
    diff = abs(testIm - Fit).reshape((225,160))
    
    diff = 255-(diff/diff.max())*255
    #_,diff = cv2.threshold(diff.astype(np.uint8),25,255,0)
    cv2.imshow('Fit', diff.astype(np.uint8))
    cv2.waitKey(0)

x = np.loadtxt('./images/marking',  dtype=str, delimiter=' ')
print('loading images...')
data = np.array([loadImageVector(name) for name in x[:]])

pca = PCA(n_components=1500)
pca.fit(data)
#cv2.imshow('Fit', (100*pca.components_[6].reshape((225,160))).astype(np.uint8))
cv2.imshow('Fit', data[0].reshape(225,160).astype(np.uint8))
cv2.waitKey(0)

#showDiff(pca,'test.jpg')
showDiff(pca,'test2.jpg')
#showDiff(pca,'Gray1520242891.93.jpg')
#showDiff(pca,'Gray1520242968.85.jpg')
#showDiff(pca,'Gray1520243053.39.jpg')
#showDiff(pca,'Gray1520242922.41.jpg')
#showDiff(pca,'Gray1520243103.83.jpg')
#showDiff(pca,'Gray1520243031.33.jpg')
#cv2.imshow('median',abs(data[-1].astype(np.int64) - m ).astype(np.uint8))
#m = np.median(data,0)
#m = m.astype(np.int64)
#print(type(data[0][0,0]))
#print(type(m[0,0]))
#cv2.imshow('median',abs(data[-1].astype(np.int64) - m ).astype(np.uint8))
#cv2.waitKey(0)
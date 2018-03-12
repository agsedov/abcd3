import numpy as np
from numpy import linalg as LA
import cv2
import os
import time
cap = cv2.VideoCapture(0)

#sudo modprobe pcspkr

# Получить список квадратов из списка контуров
def getSquares(contours):
    squares = []
    # Проверка каждого контура
    for contour in contours:
        # упрощение контура с точностью зависимой от его периметра
        e = cv2.arcLength(contour, True)*0.04
        approx = cv2.approxPolyDP(contour, e , True)
        if len(approx) == 4: # Если получился четырёхугольник
            v1 = (approx[1]-approx[0]) # Векторы рёбер
            v2 = (approx[2]-approx[1])
            v3 = (approx[3]-approx[2])
            v4 = (approx[0]-approx[3])
            l1 = LA.norm(v1)
            l2 = LA.norm(v2)
            v1 = v1/l1          # нормирование      
            v2 = v2/l2          
            v3 = v3/LA.norm(v3) 
            v4 = v4/LA.norm(v4) 
            s1 = abs((v1).dot(v2.T)[0,0]) # скалярные произведения
            s2 = abs((v2).dot(v3.T)[0,0])
            s3 = abs((v3).dot(v4.T)[0,0])
            s4 = abs((v4).dot(v1.T)[0,0])
            if(l1 > 10): # игнорировать квадраты со стороной меньше 10
                if(max(s1,s2,s3,s4) < 0.2 and abs(l1/l2-1) < 0.2 ): # приблизительно это квадраты
                    squares.append(approx);
    return squares

def sortCorners(rect):
    #Уже известно, что левой верхней должна быть либо точка 0, либо точка 1.
    v1 = (rect[1]-rect[0]) #верняя сторона
    v2 = (rect[3]-rect[2]) #нижняя сторона
    dotproduct1 = (v1/LA.norm(v1)).dot((v2/LA.norm(v2)).T) #должно быть в районе -1
    if(abs(dotproduct1-(-1))>0.01): #иначе поменяем местами 1 и 2
        rect[0], rect[1] = rect[1].copy(), rect[0].copy()

    det = LA.det([rect[1]-rect[0], rect[2] - rect [1]])   #Посчитаем ориентацию чтобы исправить
    if(det<0):                                            #случай зеркального отражения (det < 0)
        rect[0], rect[1] = rect[1].copy(), rect[0].copy() #поменяв вершины
        rect[2], rect[3] = rect[3].copy(), rect[2].copy()
    return det, rect

def fourPointTransform(image, rect):    
    Width = 160
    Height = 225

    dst = np.array([
        [0, 0],
        [Width - 1, 0],
        [Width - 1, Height - 1],
        [0, Height - 1]], dtype = "float32")
 
    # Подсчёт матрицы перспективного преобразования
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (Width, Height))

    return warped

kernel = np.ones((5,5),np.float32)#/25
flush_buffer = False

while(True):
    if flush_buffer:            #сброс буфера видеокамеры
        for i in range(1,15):   #TODO: найти более лучший путь
            cap.grab()
        flush_buffer = False

    ret, frame = cap.read()

    #Преобразование к чёрно-белому
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #Перебор всех пороговых значений пока не найдется ровно 6 квадратов
    for thresholdVal in range(16,255,16):
        #Бинаризация с пороговым значением thresholdVal
        ret,thresh = cv2.threshold(gray,thresholdVal,255,0)
        thresh = cv2.dilate(thresh,kernel)
            
        threshim = thresh.copy() #копия картинки для показа
        
        # поиск контуров
        im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE);
        
        squares = getSquares(contours)

        if(len(squares) == 6):
            break;
    if(len(squares) == 6):
        img = cv2.drawContours(frame, squares, -1, (255,255,255), 3)
        centers = np.transpose(np.array(squares),(0,2,3,1)).dot(np.matrix([[1,1,1,1]]).T)/4
        center = np.array(centers.T.dot(np.matrix([[1,1,1,1,1,1]]).T)/6).T

        #сортировка найденных квадратов по расстоянию до центра
        centers = sorted(centers, key=lambda x: LA.norm(center - x), reverse=False)

        cv2.circle(img, (int(center[0,0]),int(center[0,1])), 5, (0,0,255))
        cv2.circle(img, (int(centers[0][0,0]),int(centers[0][0,1])), 5, (0,255,0))
        cv2.circle(img, (int(centers[1][0,0]),int(centers[1][0,1])), 5, (0,255,0))
        cv2.circle(img, (int(centers[2][0,0]),int(centers[2][0,1])), 5, (255,255,255))
        cv2.circle(img, (int(centers[3][0,0]),int(centers[3][0,1])), 5, (255,255,255))
        cv2.circle(img, (int(centers[4][0,0]),int(centers[4][0,1])), 5, (255,0,0))
        cv2.circle(img, (int(centers[5][0,0]),int(centers[5][0,1])), 5, (255,0,0))

        cv2.imshow('img',img)
        
        sq = np.transpose(np.array(centers[2:6], dtype = "float32"),(1,0,2))[0]
        det, rect  = sortCorners(sq)
        p4 = fourPointTransform(gray, rect)
        
        laplacian = cv2.Laplacian(p4, cv2.CV_64F).var()
        if(laplacian > 90 ): #отсекаем размытые изображения
            print('laplacian:',laplacian)
            print('determinant:',det)
            cv2.imshow('last',img)
            cv2.imshow('capture',p4)
            cv2.imwrite( "./images/Gray"+str(time.time())+".jpg", p4 );
            os.system("beep -f 555 -l 150") #Beep
            cv2.waitKey(2000)   #ждем две секунды чтобы не детектировать снова 
            flush_buffer = True #флаг что надо почистить буфер после ожидания

        #cv2.waitKey(2000)
        #time.sleep(2)
    else:
        cv2.imshow('img',frame)

    cv2.imshow('thresh',threshim)
    key = cv2.waitKey(1)
    if  (key & 0xFF == ord('q')): #выход по нажатию q
        os.system("beep -f 111 -l 150") 
        break
    if key & 0xFF == ord('c'):
        os.system("beep -f 777 -l 150")

cap.release()
cv2.destroyAllWindows()
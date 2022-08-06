import cv2
import numpy as np
from keras.datasets import mnist
from matplotlib import pyplot 
from skimage.transform import resize
import math
from sklearn import svm
from sklearn.svm import SVC
from sklearn.metrics import classification_report,accuracy_score
import pickle
#loading
(train_X, train_y), (test_X, test_y) = mnist.load_data()

train_X = np.array(~train_X[:50000] , dtype=float)
train_y = np.array(train_y[:50000])
test_img = np.array(~test_X[:10000] , dtype=float)
test_label = np.array(test_y[:10000])

training_img = np.pad(train_X, ((0,0), (2,2), (2, 2)), 'constant')
test_img = np.pad(test_img, ((0,0), (2,2), (2, 2)), 'constant')

training_gradiant_x = np.pad(training_img, ((0,0), (1,1), (1, 1)), 'constant')
training_gradiant_y = np.pad(training_img, ((0,0), (1,1), (1, 1)), 'constant')

test_gradiant_x = np.pad(test_img, ((0,0), (1,1), (1, 1)), 'constant')
test_gradiant_y = np.pad(test_img, ((0,0), (1,1), (1, 1)), 'constant')

training_Magnitude = np.zeros((50000,32,32))
training_Angle = np.zeros((50000,32,32))

test_Magnitude = np.zeros((50000,32,32))
test_Angle = np.zeros((50000,32,32))

training_feature_vector = np.zeros((50000,9,4,9))
test_feature_vector = np.zeros((10000,9,4,9))

def calc_Gradiant_x(x,images_array):
    
    for i in range(x):
        for j in range(1,34):
            for k in range(1,33):
                images_array[i][j][k] = images_array[i][j][k+1] - images_array[i][j][k-1] / 2
    
    return images_array

def calc_Gradiant_y(x,images_array):
    
    for i in range(x):
        for j in range(1,33):
            for k in range(1,34):
                images_array[i][j][k] = images_array[i][j+1][k] - images_array[i][j-1][k]  / 2
    
    return images_array

def calc_Magnitude(x,gradiant_x,gradiant_y):
    images_array = np.zeros((x,32,32))
    for i in range(x):
        for j in range(0,32):
            for k in range(0,32):
                images_array[i][j][k] = math.sqrt( pow(gradiant_x[i][j+1][k+1],2) + pow(gradiant_y[i][j+1][k+1],2) )
    return images_array   

def calc_Angle(x,gradiant_x,gradiant_y):
    images_array = np.zeros((x,32,32))
    for i in range(x):
        for j in range(0,32):
            for k in range(0,32):
                images_array[i][j][k] = np.rad2deg(np.arctan(gradiant_y[i][j+1][k+1] / (gradiant_x[i][j+1][k+1]+0.0000001)))
                images_array[i][j][k] = math.ceil(images_array[i][j][k]%180)
    return images_array            

def calc_feature_vector(x,training_Magnitude,training_Angle):
    images_array =np.zeros((x,9,4,9))
    
    
    for pic in range(x):
        c1 = 0  
        c2 = 0
        for block in range(9):
            if block == 1 :
                c1=0
                c2=8
            elif block == 2:
                c1=0
                c2=16
            elif block == 3:
                c1 =8
                c2 = 0
            elif block == 4: 
                c1=8
                c2=8
            elif block == 5:
                c1=8
                c2=16
            elif block == 6:
                c1=16 
                c2=0 
            elif block == 7:
                c1=16
                c2=8
            elif block == 8:
                c1=16
                c2=16
            for cell in range(4):
                
                if cell == 1 :
                    c2+=8
                elif cell == 2:
                    c1+=8
                    c2-=8
                elif cell == 3:
                    c2+=8
                for i in range(c1,c1+8):
                    for j in range(c2,c2+8):
                        print(str(pic))
                        x = training_Angle[pic][i][j]
                        if x==180:
                            images_array[pic][block][cell][8] = training_Magnitude[pic][i][j]
                            continue
                        p1 = (x - math.ceil(x/20)*20 - 10) / ((math.floor(x/20)+1)*20)
                        p2 = (x - math.ceil(x/20)*20 + 10) / ((math.floor(x/20)+1)*20)
                        
                        if p1 <= 0 :
                            images_array[pic][block][cell][math.floor(x/20)] += training_Magnitude[pic][i][j]
                            
                        else :
                            images_array[pic][block][cell][math.floor(x/20)] += p2 * training_Magnitude[pic][i][j]
                            images_array[pic][block][cell][math.floor(x/20) + 1] += p1 * training_Magnitude[pic][i][j]
                  
    return images_array 

def normalization(x,images_array):
    temp = []
    for pic in range(x):
        for block in range(9):
            Sum = np.sum(images_array[pic][block]) 
            if Sum == 0:
                continue
            for i in range(4):
                for j in range(9):
                    images_array[pic][block][i][j] = images_array[pic][block][i][j] / Sum
        temp.append(images_array[pic].flatten().tolist())         
    
    return temp     

def training(training_img):      
    calc_Gradiant_x(10000,training_gradiant_x)
    calc_Gradiant_y(10000,training_gradiant_y)
    training_Magnitude = calc_Magnitude(10000,training_gradiant_x,training_gradiant_y)
    training_Angle = calc_Angle(10000,training_gradiant_x,training_gradiant_y)
    training_feature_vector = calc_feature_vector(10000,training_Magnitude,training_Angle)
    training_fv = normalization(10000,training_feature_vector)

    np.save('training_fv.txt',training_fv)
    return training_fv

def test(test_img):      
    calc_Gradiant_x(3000,test_gradiant_x)
    calc_Gradiant_y(3000,test_gradiant_y)
    test_Magnitude = calc_Magnitude(3000,test_gradiant_x,test_gradiant_y)
    test_Angle = calc_Angle(3000,test_gradiant_x,test_gradiant_y)
    test_feature_vector = calc_feature_vector(3000,test_Magnitude,test_Angle)
    test_fv = normalization(3000,test_feature_vector)
    np.save('test_fv.txt',test_fv)
    return test_fv

def fo(training_fv,train_y,test_fv,test_label):
    training_fv = np.asarray(training_fv , dtype='float')
    test_fv = np.asarray(test_fv ,dtype = 'float')
    

    clf = svm.SVC()
    clf.fit(training_fv,train_y)
    
    pred = clf.predict(test_fv)

    filename = 'SvMClass.sav'
    pickle.dump(clf, open(filename, 'wb'))

    print("Accurate percentage: " + str(accuracy_score(test_label, pred)))
    print('\n')
    print(classification_report(test_label, pred))
    

def main():
    training_fv = training(training_img)
    print("train")
    test_fv = test(test_img)
    print("test")
    fo(training_fv,train_y[:10000],test_fv,test_label[:3000])
    
main()
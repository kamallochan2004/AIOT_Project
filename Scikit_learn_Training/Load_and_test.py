import cv2 as cv
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

pretrained=cv.CascadeClassifier('D:/Work/AIOT_Project/Scikit_learn_Training/haarcascade_frontalface_default.xml') #Replace your own path

mask=np.load('Scikit_learn_Training/mask.npy')
no_mask=np.load('Scikit_learn_Training/no_mask.npy')

#print(mask.shape)
#print(no_mask.shape)

mask=mask.reshape(200,50*50*3)
no_mask=no_mask.reshape(200,50*50*3)

#print(mask.shape)
#print(no_mask.shape)

#Concatenate all rows only
X=np.r_[no_mask,mask]
X.shape

Y=np.zeros(X.shape[0])
Y.shape

Y[200:]=1.0

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.20)
print(len(X_train))
print(len(X_test))
print(len(Y_train))
print(len(Y_test))

print(X_train.shape)
print(X_test.shape)

pca=PCA(n_components=3)
X_train=pca.fit_transform(X_train)
X_test=pca.fit_transform(X_test)
print(X_train.shape)
print(X_test.shape)

print(X_train[0])

model=SVC()
model.fit(X_train,Y_train)
model.score(X_test,Y_test)

Y_pred=model.predict(X_test)
Y_pred

Y_test

accuracy_score(Y_test,Y_pred)

print(classification_report(Y_test,Y_pred))

cm=confusion_matrix(Y_test,Y_pred)
plt.figure(figsize=(4,3))
g=sns.heatmap(cm,cmap="Blues",annot=True, fmt='g')
g.set_xticklabels(labels=['No Mask (0)','Mask(1)'],rotation=30)
g.set_yticklabels(labels=['No Mask (0)','Mask(1)'],rotation=30)
plt.ylabel('True Label',fontsize=14)
plt.xlabel('Prediction Label',fontsize=14)
plt.title('Confusion Matrix',fontsize=16)
plt.show()

#Code to capture video
capture=cv.VideoCapture(0)

name={0:'No Mask', 1:'Mask'}
data=[]
font=cv.FONT_HERSHEY_COMPLEX

while True:
    ret,img=capture.read()
    if ret:
        faces=pretrained.detectMultiScale(img)
        for x,y,w,h in faces:
            cv.rectangle(img,(x,y),(x+w,y+h),(255,0,255),4)
            face=img[y:y+h,x:x+w,:]
            face=cv.resize(face,(50,50))
            face=face.reshape(1,-1)
            face=pca.transform(face)
            pred=model.predict(face)
            n=name[int(pred)]
            cv.putText(img,n,(x,y),font,1,(244,250,250),2)  
            print(n)    
        cv.imshow('Testing',img) 
        if cv.waitKey(1)==ord('q'):
            break
capture.release()
cv.destroyAllWindows()
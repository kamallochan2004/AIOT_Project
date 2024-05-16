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

pretrained=cv.CascadeClassifier('haarcascade_frontalface_default.xml')

capture=cv.VideoCapture(0)

data=[]

while True:
    ret,img=capture.read()
    if ret:
      faces=pretrained.detectMultiScale(img)
      for x,y,w,h in faces:
        cv.rectangle(img,(x,y),(x+w,y+h),(0,255,0),4)
        face=img[y:y+h,x:x+w,:]
        face=cv.resize(face,(50,50))
        print(len(data))
        if len(data)<400:
            data.append(face)
      cv.imshow("With_mask",img)
      if cv.waitKey(1)==ord('q') or len(data)>=200:
        break
capture.release()
cv.destroyAllWindows()
np.save('mask.npy',data)
plt.figure(figsize=(2,2))
plt.imshow(data[0])
plt.axis(False)
plt.show()

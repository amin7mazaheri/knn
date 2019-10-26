import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA


faces = datasets.fetch_olivetti_faces()
fig = plt.figure(figsize=(8, 6))
for i in range(15):
    ax = fig.add_subplot(3, 5, i + 1, xticks=[], yticks=[])
    ax.imshow(faces.images[i], cmap='gray')


X_train, X_test, y_train, y_test = train_test_split(faces.data,faces.target, test_size=0.2)
pca = PCA(whiten=True)
pca.fit(X_train)
W=pca.components_.T
fig = plt.figure(figsize=(16, 6))
for i in range(30):
    ax = fig.add_subplot(3, 10, i + 1, xticks=[], yticks=[])
    ax.imshow(W[:,i].reshape(faces.images[0].shape),
              cmap='gray')
X_train_pca = pca.transform(X_train)[:,:30]
X_test_pca = pca.transform(X_test)[:,:30]

Knno=KNeighborsClassifier(n_neighbors=3,metric='minkowski',p=2)
Knnpca=KNeighborsClassifier(n_neighbors=3,metric='minkowski',p=2)
Knno.fit(X_train,y_train)
Knnpca.fit(X_train_pca,y_train)
#confusion_matrix(y_test,Knn.predict(X_test))
#Knn.predict(X_test)
#Knn.predict(X_test_pca)
print('Knn-PCA Prediction Accuracy:',Knnpca.score(X_test_pca,y_test))
print('Oeigial Knn Prediction Accuracy:',Knno.score(X_test,y_test))



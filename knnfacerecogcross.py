from sklearn.model_selection import cross_val_score
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
comp=np.arange(10,50,10)
nneighbors=np.arange(1,5)
l=0
cv=5
cvscores=[]
cvscorespca=[]
for c in comp:
    for k in nneighbors:
     X_train_pca = pca.transform(X_train)[:,:c]
     #Knno=KNeighborsClassifier(n_neighbors=k,metric='minkowski',p=2)
     Knnpca=KNeighborsClassifier(n_neighbors=k,metric='minkowski',p=2)
     #cvscores[l]= np.mean(cross_val_score(Knno, X_train, y_train, cv=cv))
     cvscorespca.append([c,k,np.mean(cross_val_score(Knnpca, X_train_pca, y_train, cv=cv))])
     l=l+1
l=0
for k in nneighbors:    
     Knno=KNeighborsClassifier(n_neighbors=k,metric='minkowski',p=2)
     cvscores.append([k,np.mean(cross_val_score(Knno, X_train, y_train, cv=cv))])
     l=l+1
for i in range(len(cvscorespca)):
	print(cvscorespca[i])
for i in range(len(cvscores)):
	print(cvscores[i])


Knno=KNeighborsClassifier(n_neighbors=1,metric='minkowski',p=2)
Knnpca=KNeighborsClassifier(n_neighbors=1,metric='minkowski',p=2)
X_train_pca = pca.transform(X_train)[:,:20]
X_test_pca = pca.transform(X_test)[:,:20]
Knno.fit(X_train,y_train)
Knnpca.fit(X_train_pca,y_train)

print('Knn-PCA Prediction Accuracy:',Knnpca.score(X_test_pca,y_test))
print('Oeigial Knn Prediction Accuracy:',Knno.score(X_test,y_test))



import matplotlib.pyplot as plt
import sklearn.datasets as skdata
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

numeros = skdata.load_digits()
target = numeros['target']
imagenes = numeros['images']
n_imagenes = len(target)
#print(np.shape(imagenes), n_imagenes) # Hay 1797 digitos representados en imagenes 8x8
data = imagenes.reshape((n_imagenes, -1)) # para volver a tener los datos como 
#imagen basta hacer data.reshape((n_imagenes, 8, 8))
#print(np.shape(data))

scaler = StandardScaler()
x_train, x_test, y_train, y_test = train_test_split(data, target, train_size=0.5)
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
clf = LogisticRegression(C=1, penalty='l1', solver='saga', tol=0.01)
clf.fit(x_train, y_train)
score = clf.score(x_test, y_test)
#print(score)

coef = clf.coef_.copy()
plt.figure(figsize=(10, 6))
scale = np.abs(coef).max()
for i in range(10):
    l1_plot = plt.subplot(2, 5, i + 1)
    l1_plot.imshow(coef[i].reshape(8,8), interpolation='nearest',cmap=plt.cm.RdBu, vmin=-scale, vmax=scale)
    l1_plot.set_xticks(())
    l1_plot.set_yticks(())
    l1_plot.set_xlabel('Class %i' % i,fontsize=15)
plt.suptitle('Clasificación',fontsize=15)
plt.savefig('coeficientes.png')
#plt.show()

cm = confusion_matrix(y_test, clf.predict(x_test))
plt.figure(figsize=(10, 10))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
classNames = ['0','1','2','3','4','5','6','7','8','9']
plt.title('Matriz de confusión - Score = %0.5f' %score, fontsize=15)
plt.ylabel('True',fontsize=15)
plt.xlabel('Predict',fontsize=15)
tick_marks = np.arange(len(classNames))
plt.xticks(tick_marks, classNames, rotation=45,fontsize=15)
plt.yticks(tick_marks, classNames, fontsize=15)
for i in range(10):
    for j in range(10):
        plt.text(j-0.2, i+0.1, str(cm[i][j]),fontsize=15)
plt.savefig('confusion.png')
#plt.show()

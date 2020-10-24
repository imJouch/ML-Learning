#coding:utf-8
"""
python 3
sklearn 0.18
"""
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
from sklearn.linear_model import LogisticRegression
import input_data
import numpy as np
import pickle
import datetime

start_time = datetime.datetime.now()
mnist = input_data.read_data_sets('MNIST_data/',one_hot=False)
x = mnist.train.images
y = mnist.train.labels

#采用交叉验证
train_data,validation_data,train_labels,validation_labels = train_test_split(x,y,test_size=0.2)
#训练一个LogisticRegression分类器

clf = LogisticRegression(penalty='l2',tol=0.001)
clf.fit(train_data,train_labels)
predictions = []
for i in range(1000):
    if i % 100 == 0:
        print('======>>>>>>','epoch:',int(i/100))
    #将预测的结果存入prediction
    output = clf.predict([mnist.test.images[i]])
    predictions.append(output)
#混淆矩阵
print(confusion_matrix(mnist.test.labels[0:1000],predictions))
#classification_report
print(classification_report(mnist.test.labels[0:1000],np.array(predictions)))
print('test accuracy is:',accuracy_score(mnist.test.labels[0:1000],predictions))
with open('logistic.pickle','wb') as f:
    pickle.dump(clf,f)
end_time = datetime.datetime.now()
print('total time is :',(end_time - start_time).seconds)
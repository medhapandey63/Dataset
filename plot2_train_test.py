# -*- coding: utf-8 -*-
"""
Created on Tue Jan 22 13:54:15 2019

@author: medha
"""
import plotly
import plotly.plotly as py
plotly.tools.set_credentials_file(username='medha123', api_key='yDo44ODA0VxyBYTE5hSf')
import numpy as np
import keras
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import plot_model
from sklearn import model_selection
from sklearn.preprocessing import StandardScaler
#from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.metrics import classification_report,confusion_matrix,precision_score, recall_score, f1_score, cohen_kappa_score
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
import pandas
from sklearn.metrics import roc_curve
from sklearn.utils import shuffle

	
# Visualize training history
from keras.models import Sequential
from keras.layers import Dense
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy
# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)
# load dataset

df1 = pandas.read_csv("balanced_window_5.csv",header = None)
dataset = df1.values
dataset = shuffle(dataset)
#dataset.ndim
# Fit the model on 33%
X = dataset[:,0:-1].astype(float)
#dtype(X)
Y = dataset[:,-1].astype(int)
en_Y = label_binarize(Y, classes=[0, 1])


############ define the dataset and labels ################ 
test_size = 0.30
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, en_Y, test_size=test_size, random_state=seed)
scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
#dtype(en_Y)
#b = np.unique(en_Y)
#print(b)
#dataset = numpy.loadtxt("pssm_domain_location1_without_header.csv", delimiter=",").head(100)
#print(dataset)
# split into input (X) and output (Y) variables

####create model
skf = StratifiedKFold(n_splits=10)
cvscores = []
model = Sequential()
model.add(Dense(400, input_dim = 49, kernel_initializer='uniform', activation='relu'))
model.add(Dense(300, kernel_initializer='uniform', activation = 'relu'))
model.add(Dense(200, kernel_initializer='uniform', activation = 'relu'))
model.add(Dense(100, kernel_initializer='uniform', activation = 'relu'))
#model.add(Dense(200, kernel_initializer='uniform', activation = 'relu'))
model.add(Dense(1, kernel_initializer='uniform', activation='sigmoid'))
#model = Sequential()
#model.add(Dense(10000, input_dim = 49, kernel_initializer='uniform', activation='relu'))
#model.add(Dense(8000, kernel_initializer='uniform', activation = 'relu'))
#model.add(Dense(6000, kernel_initializer='uniform', activation = 'relu'))
#model.add(Dense(4000, kernel_initializer='uniform', activation = 'relu'))
#model.add(Dense(1, kernel_initializer='uniform', activation='sigmoid'))
## Compile model
model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])
#model.compile(loss='mean_squared_error', optimizer='rmsprop', metrics=['accuracy'])
## Fit the model
history = model.fit(X_train, Y_train, epochs=1, batch_size=500, verbose=1)
## list all data in history
print(history.history.keys())
scores = model.evaluate(X_test, Y_test, verbose = 1)
print(scores)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
#cvscores.append(scores[1] * 100)
#print(cvsscores)
val_y_pred = model.predict_classes(X_test)
#print(confusion_matrix(Y_test, val_y_pred))
target_names = ['class 0', 'class 1']
print (np.unique(val_y_pred))
print(classification_report(Y_test, val_y_pred))
#fpr = dict()
#tpr = dict()
#roc_auc = dict()
##for i in range(n_classes):
#fpr, tpr, _ = roc_curve(Y_test, history)
#roc_auc = auc(fpr, tpr)
#
## Compute micro-average ROC curve and ROC area
##fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
##roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
#
##lw = 2
#
#trace1 = go.Scatter(x=fpr, y=tpr, 
#                    mode='lines', 
#                    line=dict(color='darkorange'),
#                    name='ROC curve (area = %0.2f)' % roc_auc)
#
#trace2 = go.Scatter(x=[0, 1], y=[0, 1], 
#                    mode='lines', 
#                    line=dict(color='navy', dash='dash'),
#                    showlegend=False)
#
#layout = go.Layout(title='Receiver operating characteristic example',
#                   xaxis=dict(title='False Positive Rate'),
#                   yaxis=dict(title='True Positive Rate'))
#
#fig = go.Figure(data=[trace1, trace2], layout=layout)
#
#py.iplot(fig)
#print(roc_curve(Y_test, y_score, pos_label=None, sample_weight=None, drop_intermediate=True))
#print(precision_score(Y_test, val_y_pred))
#print(recall_score(Y_test, val_y_pred))
#print(f1_score(Y_test, val_y_pred, average = 'weighted'))
#print(cohen_kappa_score(Y_test, val_y_pred))
#print("%.2f%% (+/- %.2f%%)" % (numpy.mean(cvscores), numpy.std(cvscores)))
#set(Y_test) - set(val_y_pred)
## summarize history for accuracy
#plt.plot(history.history['acc'])
#plt.plot(history.history['val_acc'])
#plt.title('model accuracy')
#plt.ylabel('accuracy')
#plt.xlabel('epoch')
#plt.legend(['train', 'test'], loc='upper left')
#plt.savefig('acc_pssm.png')
##plt.show()
## summarize history for loss
#plt.plot(history.history['loss'])
#plt.plot(history.history['val_loss'])
#plt.title('model loss')
#plt.ylabel('loss')
#plt.xlabel('epoch')
#plt.legend(['train', 'test'], loc='upper left')
#plt.savefig('loss_pssm.png')
##plt.show()
#plot_model(model, to_file='model.png')
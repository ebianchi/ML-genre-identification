



#import csv
import numpy as np
import pandas as pd
#import scikit as sk
import re

#reader = csv.reader(open("ZeroCross_SpectralCentroids.csv", "rb"), delimiter=",");
#x = list(reader);
#DATA = np.array(x).astype("float");

#Data Import & Cleansing
df = pd.read_csv('ALLDataFeatures_trimmed.csv')

Y = df['Original Audio File']

X = df.loc[:, df.columns != 'Original Audio File']
#X = X.iloc[:,0:1292] #truncate incomplete columns



from sklearn import model_selection, metrics
from sklearn import linear_model
from sklearn.linear_model import SGDClassifier



#train, validate, test = np.split(df.sample(frac=1), [int(.6*len(df)), int(.8*len(df))])
X_train, X_cv, Y_train, Y_cv = model_selection.train_test_split(X, Y, test_size=0.2, random_state=123, stratify=Y)

clf = linear_model.SGDClassifier(max_iter=300)
clf.fit(X_train, Y_train)
 
Y_train_pred = clf.predict(X_train)
Y_cv_pred = clf.predict(X_cv)
  
print(300)
print(metrics.accuracy_score(np.asarray(Y_train), Y_train_pred))
print(metrics.accuracy_score(np.asarray(Y_cv), Y_cv_pred))


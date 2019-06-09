#import csv
import numpy as np
import pandas as pd
#import scikit as sk
import collections


from sklearn import model_selection, metrics
from sklearn.svm import LinearSVC
from sklearn import linear_model
from sklearn.linear_model import SGDClassifier



#reader = csv.reader(open("ZeroCross_SpectralCentroids.csv", "rb"), delimiter=",");
#x = list(reader);
#DATA = np.array(x).astype("float");


#Data Import & Cleansing
#colnames = ['Genre']
df = np.transpose(pd.read_csv('AllDataFeatures_trimmed_11kHz.csv', header=None))

print("Finished reading csv...")

#Y = df['Genre']
X = np.transpose(df[1:])
Y = np.repeat(np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]), 100)

print("Finished building X and Y matrices...")
print(df.shape)
print(Y.shape)
print(X.shape)



#train, validate, test = np.split(df.sample(frac=1), [int(.6*len(df)), int(.8*len(df))])
X_train, X_cv, Y_train, Y_cv = model_selection.train_test_split(X, Y, test_size=0.2, random_state=123, stratify=Y)

print("Finished splitting into training/cv sets...")

iterations = [10, 30, 100, 300, 1000, 3000]

for it in iterations:
	print("--------------------", it, "ITERATIONS --------------------")

	#clf = LinearSVC(random_state=0, C = 1000, dual=True, multi_class="ovr", max_iter=it)
	clf = linear_model.SGDClassifier(max_iter=it, alpha=0.000001, tol=None)
	clf.fit(X_train, Y_train)
	
	Y_train_pred = clf.predict(X_train)
	Y_cv_pred = clf.predict(X_cv)

	print("Training accuracy and cross-validation accuracy")
	print(metrics.accuracy_score(np.asarray(Y_train), Y_train_pred))
	print(metrics.accuracy_score(np.asarray(Y_cv), Y_cv_pred))
	print()


	print("True values vs. predicted values counted:")
	trueValueVsPredicted = np.transpose(np.array([Y_cv, Y_cv_pred]))
	#print(trueValueVsPredicted)
	# u, indices = np.unique(trueValueVsPredicted, axis=0, return_index=True)
	unq, cnt = np.unique(trueValueVsPredicted, return_counts=True, axis=0)
	# print("Unique rows:")
	# print(unq)
	# print()
	# print("Counts:")
	# print(cnt)

	print(np.concatenate((np.vstack(cnt), unq), axis=1))






#import csv
import numpy as np
import pandas as pd
#import scikit as sk


from sklearn import model_selection, metrics
from sklearn.svm import LinearSVC



#Data Import & Cleansing
colnames = ['Genre', 'features ->']
df = pd.read_csv('ALLDataFeatures_trimmed.csv', names=colnames)

print("Finished reading csv...")

#Y = df['Genre']
X = df.loc[:, df.columns != 'Genre']
Y = np.transpose(np.repeat(np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]), 100))

print("Finished building X and Y matrices...")



#train, validate, test = np.split(df.sample(frac=1), [int(.6*len(df)), int(.8*len(df))])
X_train, X_cv, Y_train, Y_cv = model_selection.train_test_split(X, Y, test_size=0.2, random_state=123, stratify=Y)

print("Finished splitting into training/cv sets...")

iterations = [10, 30, 100, 300, 1000, 3000, 10000, 30000, 100000]

for it in iterations:
	print("--------------------", it, " ITERATIONS --------------------")

	clf = LinearSVC(dual=True, tol=0.0000001, C=100000, max_iter=it)
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
	print("Unique rows:")
	print(unq)
	print()
	print("Counts:")
	print(cnt)



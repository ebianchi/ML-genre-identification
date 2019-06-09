import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np




iterations = [10, 30, 100, 300, 1000, 3000]

sgd_22khz_train_accuracy = [0.3975, 0.36875, 0.9875, 1, 1, 1]
sgd_22khz_cv_accuracy = [0.195, 0.18, 0.235, 0.25, 0.23, 0.24]

svm_22khz_train_accuracy = [0.46875, 0.83125, 0.92875, 0.99625, 0.99875, 0.99875]
svm_22khz_cv_accuracy = [0.2, 0.23, 0.25, 0.235, 0.235, 0.245]

sgd_11khz_train_accuracy = [0.29, 0.62375, 0.76375, 0.975, 1, 1]
sgd_11khz_cv_accuracy = [0.135, 0.22, 0.18, 0.23, 0.22, 0.21]

svm_11khz_train_accuracy = [0.2675, 0.62625, 0.725, 0.9525, 0.99625, 0.99625]
svm_11khz_cv_accuracy = [0.135, 0.185, 0.13, 0.205, 0.2, 0.19]

plt.figure(1)
plt.title("22kHz Sampling: SGD Accuracy vs. Number of Iterations")
plt.ylabel("Accuracy")
plt.xlabel("Learning Iterations")
plt.semilogx(iterations, sgd_22khz_train_accuracy)
plt.semilogx(iterations, sgd_22khz_cv_accuracy)
plt.legend(["Training Set", "Cross-Validation Set"])

plt.figure(2)
plt.title("22kHz Sampling: SVM \"One vs. All\" Accuracy vs. Number of Iterations")
plt.ylabel("Accuracy")
plt.xlabel("Learning Iterations")
plt.semilogx(iterations, svm_22khz_train_accuracy)
plt.semilogx(iterations, svm_22khz_cv_accuracy)
plt.legend(["Training Set", "Cross-Validation Set"])

plt.figure(3)
plt.title("11kHz Sampling: SGD Accuracy vs. Number of Iterations")
plt.ylabel("Accuracy")
plt.xlabel("Learning Iterations")
plt.semilogx(iterations, sgd_11khz_train_accuracy)
plt.semilogx(iterations, sgd_11khz_cv_accuracy)
plt.legend(["Training Set", "Cross-Validation Set"])

plt.figure(4)
plt.title("11kHz Sampling: SVM \"One vs. All\" Accuracy vs. Number of Iterations")
plt.ylabel("Accuracy")
plt.xlabel("Learning Iterations")
plt.semilogx(iterations, svm_11khz_train_accuracy)
plt.semilogx(iterations, svm_11khz_cv_accuracy)
plt.legend(["Training Set", "Cross-Validation Set"])




# genres = ['blues', 'classical', 'country', 'disco', 'hiphop', \
#           'jazz', 'metal', 'pop', 'reggae', 'rock']

# sgd_22khz_genre_accuracy = [0.15, 0.7, 0.05, 0.2, 0.25, 0.4, 0.45, 0.15, 0.0, 0.15]
# svm_22khz_genre_accuracy = [0.15, 0.7, 0.05, 0.1, 0.2, 0.35, 0.5, 0.25, 0.0, 0.15]
# sgd_11khz_genre_accuracy = [0.15, 0.8, 0.05, 0.05, 0.05, 0.2, 0.5, 0.15, 0.05, 0.3]
# svm_11khz_genre_accuracy = [0.15, 0.55, 0.2, 0.1, 0.05, 0.3, 0.35, 0.15, 0.1, 0.15]


# ind = np.arange(len(genres))  # the x locations for the genres
# width = 0.2  # the width of the bars

# fig, ax = plt.subplots()
# rects1 = ax.bar(ind - (3/2) * width, sgd_22khz_genre_accuracy, width,
# 	              label='SGD 22kHz')
# rects2 = ax.bar(ind - width/2, svm_22khz_genre_accuracy, width,
# 	              label='SVM 22kHz')
# rects3 = ax.bar(ind + width/2, sgd_11khz_genre_accuracy, width,
# 	              label='SGD 11kHz')
# rects4 = ax.bar(ind + (3/2) * width, svm_11khz_genre_accuracy, width,
# 	              label='SVM 11kHz')

# # Add some text for labels, title and custom x-axis tick labels, etc.
# ax.set_ylabel('Accuracy')
# ax.set_title('Accuracy by Genre and Algorithm')
# ax.set_xticks(ind)
# ax.set_xticklabels(genres)
# ax.legend()





# men_means, men_std = (20, 35, 30, 35, 27), (2, 3, 4, 1, 2)
# women_means, women_std = (25, 32, 34, 20, 25), (3, 5, 2, 3, 3)

# ind = np.arange(len(men_means))  # the x locations for the groups
# width = 0.35  # the width of the bars

# fig, ax = plt.subplots()
# rects1 = ax.bar(ind - width/2, men_means, width, yerr=men_std,
#                 label='Men')
# rects2 = ax.bar(ind + width/2, women_means, width, yerr=women_std,
#                 label='Women')

# # Add some text for labels, title and custom x-axis tick labels, etc.
# ax.set_ylabel('Scores')
# ax.set_title('Scores by group and gender')
# ax.set_xticks(ind)
# ax.set_xticklabels(('G1', 'G2', 'G3', 'G4', 'G5'))
# ax.legend()




plt.show()

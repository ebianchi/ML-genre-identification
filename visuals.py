import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt




svm_iterations = [10, 30, 100, 300, 1000, 3000]
svm_train_accuracy = [0.57375, 0.60375, 0.9975, 1, 1, 1]
svm_cv_accuracy = [0.22, 0.225, 0.245, 0.25, 0.245, 0.235]


plt.figure()

plt.title("SVM Accuracy vs. Number of Iterations")
plt.ylabel("Accuracy")
plt.xlabel("Learning Iterations")
plt.semilogx(svm_iterations, svm_train_accuracy)
plt.semilogx(svm_iterations, svm_cv_accuracy)

plt.legend(["Training Set", "Cross-Validation Set"])

plt.show()

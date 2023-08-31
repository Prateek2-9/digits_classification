"""
================================
Recognizing hand-written digits
================================

This example shows how scikit-learn can be used to recognize images of
hand-written digits, from 0-9.

"""

# Author: Gael Varoquaux <gael dot varoquaux at normalesup dot org>
# License: BSD 3 clause

# Standard scientific Python imports
import matplotlib.pyplot as plt

# Import datasets, classifiers and performance metrics
from sklearn import metrics, svm
from utils import preprocess_data, split_train_dev_test, train_model, read_digits, predict_and_eval

# 1. Get the data
X, y = read_digits()

# 2. Split data into train, dev, and test sets
X_train, X_test, y_train, y_test, X_dev, y_dev = split_train_dev_test(X, y, test_size=0.2, dev_size=0.1)

# 3. Data preprocessing
X_train = preprocess_data(X_train)
X_dev = preprocess_data(X_dev)
X_test = preprocess_data(X_test)

# Create a classifier: a support vector classifier
clf = svm.SVC(gamma=0.001)

# 4. Model training
model = train_model(X_train, y_train, {'gamma': 0.001}, model_type='svm')

# 5. Predict and evaluate using the dev set
predicted_dev, classification_report_dev = predict_and_eval(model, X_dev, y_dev)
print("Evaluation on Dev Set:")
print(classification_report_dev)

# 6. Getting model predictions on the test set and evaluating
predicted_test, classification_report_test = predict_and_eval(model, X_test, y_test)
print("Evaluation on Test Set:")
print(classification_report_test)

predicted = model.predict(X_test)

# 7. Visualization and printing confusion matrix
# Below we visualize the first 4 test samples and show their predicted digit value in the title.
_, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
for ax, image, prediction in zip(axes, X_test, predicted_test):
    ax.set_axis_off()
    image = image.reshape(8, 8)
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
    ax.set_title(f"Prediction: {prediction}")

# Display the confusion matrix
disp = metrics.ConfusionMatrixDisplay.from_predictions(y_test, predicted_test)
disp.figure_.suptitle("Confusion Matrix")
print(f"Confusion matrix:\n{disp.confusion_matrix}")

# Show the plots
plt.show()

# 8. Evaluation
print(
    f"Classification report for classifier {model}:\n"
    f"{metrics.classification_report(y_test, predicted)}\n"
)

###############################################################################
# We can also plot a :ref:`confusion matrix <confusion_matrix>` of the
# true digit values and the predicted digit values.

disp = metrics.ConfusionMatrixDisplay.from_predictions(y_test, predicted)
disp.figure_.suptitle("Confusion Matrix")
print(f"Confusion matrix:\n{disp.confusion_matrix}")


###############################################################################
# If the results from evaluating a classifier are stored in the form of a
# :ref:`confusion matrix <confusion_matrix>` and not in terms of `y_true` and
# `y_pred`, one can still build a :func:`~sklearn.metrics.classification_report`
# as follows:


# The ground truth and predicted lists
y_true = []
y_pred = []
cm = disp.confusion_matrix

# For each cell in the confusion matrix, add the corresponding ground truths
# and predictions to the lists
for gt in range(len(cm)):
    for pred in range(len(cm)):
        y_true += [gt] * cm[gt][pred]
        y_pred += [pred] * cm[gt][pred]

print(
    "Classification report rebuilt from confusion matrix:\n"
    f"{metrics.classification_report(y_true, y_pred)}\n"
)

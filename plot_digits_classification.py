"""
================================
Recognizing hand-written digits
================================

This example shows how scikit-learn can be used to recognize images of
hand-written digits, from 0-9.

"""

# Author: Gael Varoquaux <gael dot varoquaux at normalesup dot org>
# License: BSD 3 clause
import matplotlib.pyplot as plt
from sklearn import metrics
from utils import preprocess_data, split_train_dev_test, read_digits, tune_hparams, predict_and_eval
from itertools import product

gamma_ranges = [0.001, 0.01, 0.1, 1, 10, 100]
C_ranges = [0.1, 1, 2, 5, 10]
test_dev_sizes = [(0.1, 0.1), (0.1, 0.2), (0.1, 0.3), (0.2, 0.1), (0.2, 0.2), (0.2, 0.3), (0.3, 0.1), (0.3, 0.2), (0.3, 0.3)]

for test_size, dev_size in test_dev_sizes:
    train_size = 1.0 - test_size - dev_size

    # 1. Get the data
    X, y = read_digits()

    # 2. Split data into train, dev, and test sets
    X_train, X_test, X_dev, y_train, y_test, y_dev = split_train_dev_test(X, y, test_size=test_size, dev_size=dev_size)

    # 3. Data preprocessing
    X_train = preprocess_data(X_train)
    X_dev = preprocess_data(X_dev)
    X_test = preprocess_data(X_test)

    # HYPERPARAMETER TUNING
    # Create a list of dictionaries for hyperparameter combinations
    param_combinations = [{'gamma': gamma, 'C': C} for gamma, C in product(gamma_ranges, C_ranges)]

    # Perform hyperparameter tuning
    best_hparams, best_model, best_acc = tune_hparams(X_train, y_train, X_dev, y_dev, param_combinations, model_type='svm')

    # Print the results
    train_acc = predict_and_eval(best_model, X_train, y_train)
    dev_acc = predict_and_eval(best_model, X_dev, y_dev)
    test_acc = predict_and_eval(best_model, X_test, y_test)

    print(f'test_size={test_size} dev_size={dev_size} train_size={train_size:.1f} train_acc={train_acc:.4f} dev_acc={dev_acc:.4f} test_acc={test_acc:.4f}')
    print(f'Best Hyperparameters for this combination: {best_hparams}\n')

    # 2.1: Print the number of total samples
    total_samples = len(X_train) + len(X_dev) + len(X_test)
    print(f"Total number of samples in the dataset: {total_samples}")

    # Task 2.2: Print the size (height and width) of the images in the dataset
    image_height, image_width = 8, 8  # Since these are flattened 64-pixel images
    print(f"Size of the images in the dataset (height x width): {image_height} x {image_width}")







# # 5. Predict and evaluate using the dev set
# predicted_dev, classification_report_dev = predict_and_eval(model, X_dev, y_dev)
# print("Evaluation on Dev Set:")
# print(classification_report_dev)

# # 6. Getting model predictions on the test set and evaluating
# predicted_test, classification_report_test = predict_and_eval(model, X_test, y_test)
# print("Evaluation on Test Set:")
# print(classification_report_test)

# predicted = model.predict(X_test)

# # 7. Visualization and printing confusion matrix
# # Below we visualize the first 4 test samples and show their predicted digit value in the title.
# _, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
# for ax, image, prediction in zip(axes, X_test, predicted_test):
#     ax.set_axis_off()
#     image = image.reshape(8, 8)
#     ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
#     ax.set_title(f"Prediction: {prediction}")

# # Display the confusion matrix
# disp = metrics.ConfusionMatrixDisplay.from_predictions(y_test, predicted_test)
# disp.figure_.suptitle("Confusion Matrix")
# print(f"Confusion matrix:\n{disp.confusion_matrix}")

# # Show the plots
# plt.show()

# # 8. Evaluation
# print(
#     f"Classification report for classifier {model}:\n"
#     f"{metrics.classification_report(y_test, predicted)}\n"
# )

# ###############################################################################
# # We can also plot a :ref:`confusion matrix <confusion_matrix>` of the
# # true digit values and the predicted digit values.

# disp = metrics.ConfusionMatrixDisplay.from_predictions(y_test, predicted)
# disp.figure_.suptitle("Confusion Matrix")
# print(f"Confusion matrix:\n{disp.confusion_matrix}")


# ###############################################################################
# # If the results from evaluating a classifier are stored in the form of a
# # :ref:`confusion matrix <confusion_matrix>` and not in terms of `y_true` and
# # `y_pred`, one can still build a :func:`~sklearn.metrics.classification_report`
# # as follows:


# # The ground truth and predicted lists
# y_true = []
# y_pred = []
# cm = disp.confusion_matrix

# # For each cell in the confusion matrix, add the corresponding ground truths
# # and predictions to the lists
# for gt in range(len(cm)):
#     for pred in range(len(cm)):
#         y_true += [gt] * cm[gt][pred]
#         y_pred += [pred] * cm[gt][pred]

# print(
#     "Classification report rebuilt from confusion matrix:\n"
#     f"{metrics.classification_report(y_true, y_pred)}\n"
# )

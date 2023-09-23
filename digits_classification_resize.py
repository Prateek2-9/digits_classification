import matplotlib.pyplot as plt
from sklearn import metrics
from PIL import Image
from utils import preprocess_data, split_train_dev_test, read_digits, tune_hparams, predict_and_eval
from itertools import product
import numpy as np

image_sizes = [(4, 4), (6, 6), (8, 8)]
train_size = 0.7
dev_size = 0.1
test_size = 0.2

for image_size in image_sizes:
    X, y = read_digits()
    resized_images = [Image.fromarray((image * 255).astype('uint8')).resize(image_size) for image in X]

    flattened_images = [np.array(image).reshape(-1) for image in resized_images]

    X_train, X_test, X_dev, y_train, y_test, y_dev = split_train_dev_test(flattened_images, y, test_size=test_size, dev_size=dev_size)

    X_train = np.array(X_train)
    X_dev = np.array(X_dev)
    X_test = np.array(X_test)

    gamma_ranges = [0.001, 0.01, 0.1, 1, 10, 100]
    C_ranges = [0.1, 1, 2, 5, 10]
    param_combinations = [{'gamma': gamma, 'C': C} for gamma, C in product(gamma_ranges, C_ranges)]

    best_hparams, best_model, best_acc = tune_hparams(X_train, y_train, X_dev, y_dev, param_combinations, model_type='svm')

    train_acc = predict_and_eval(best_model, X_train, y_train)
    dev_acc = predict_and_eval(best_model, X_dev, y_dev)
    test_acc = predict_and_eval(best_model, X_test, y_test)

    print(f"image size: {image_size[0]}x{image_size[1]} train_size: {train_size} dev_size: {dev_size} test_size: {test_size} train_acc: {train_acc:.4f} dev_acc: {dev_acc:.4f} test_acc: {test_acc:.4f}")



from utils import get_list_of_param_comination, load_dataset,split_train_dev_test
import utils
from joblib import load
import os
import json
from api.app import app
from sklearn import datasets
import numpy as np

def test_get_list_of_param_comination():
    gamma_values = [0.001, 0.002, 0.005, 0.01, 0.02]
    C_values = [0.1, 0.2, 0.5, 0.75, 1]
    list_of_param = [gamma_values, C_values]
    param_names =  ['gamma', 'C']
    assert len(get_list_of_param_comination(list_of_param, param_names)) == len(gamma_values) * len(C_values)


def test_get_list_of_param_comination_values():
    gamma_values = [0.001, 0.01] 
    C_values = [0.1]
    list_of_param = [gamma_values, C_values]
    param_names =  ['gamma', 'C']
    hparams_combs = get_list_of_param_comination(list_of_param, param_names)
    assert ({'gamma':0.001, 'C':0.1} in hparams_combs) and  ({'gamma':0.01, 'C':0.1} in hparams_combs)


def test_data_splitting():
    X,y = load_dataset()
    X = X[:100, :, :]
    y = y[:100]
    test_size = 0.1
    dev_size = 0.6
    train_size = 1 - test_size - dev_size
    X_train, y_train, X_test, y_test, X_dev, y_dev = split_train_dev_test(X, y, test_size, dev_size)
    assert (X_train.shape[0] == int(train_size*X.shape[0]))  and ( X_test.shape[0] == int(test_size*X.shape[0])) and ( X_dev.shape[0] == int(dev_size*X.shape[0]))

def test_prediction():
    digits = datasets.load_digits()

    image_digits = {i: [] for i in range(10)}

    for image, label in zip(digits.images, digits.target):
        image_digits[label].append(image)
        assert len(image_digits) == 10
    for key in image_digits.keys():
        image_array = utils.preprocess_data(np.array([(image_digits[key][1])]))
        image_dict = {"image": image_array[0].tolist()}
        response = app.test_client().post("/prediction", json=json.dumps(image_dict))
        # this assert is running for 10 times with different images
        assert int(json.loads(response.data)["prediction"]) == key
#imports
from sklearn.model_selection import train_test_split
from sklearn import datasets, svm, metrics, tree
import matplotlib.pyplot as plt
from itertools import product
from joblib import dump, load


#load dataset from sklearn
def load_dataset():
    digit_data = datasets.load_digits()
    X = digit_data.images
    y = digit_data.target
    return X,y


#data preprocessing
def data_preprocessing(data):
    n_samples = len(data)
    data = data.reshape((n_samples, -1))
    return data

#spliting data 
# def train_test_spliting(X,y):
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=10)
#     return  X_train, X_test, y_train, y_test

def split_train_dev_test(X, y, test_size, dev_size):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size, random_state=10)
    X_train, X_dev, y_train, y_dev = train_test_split(X_train, y_train, test_size = dev_size/(1-test_size), random_state=10)
    return X_train, y_train, X_test, y_test, X_dev, y_dev



#model training
def train_model(X_train, y_train, model_params, model_type):
    if model_type == 'svm':
        clf = svm.SVC
    if model_type == 'tree':
        clf =tree.DecisionTreeClassifier
    model = clf(**model_params)
    model.fit(X_train, y_train)
    tmp = '_'.join([f"{k}:{v}"for k,v in model_params.items()])
    model_path = f"models/{model_type}_{tmp}.joblib"
    try:
        dump(model, model_path)
    except Exception as e:
        print(str(e))
    return model

#prediction and accuracy evaluation
def predict_and_eval(model, X_test, y_test):
    predicted = model.predict(X_test)
    accuracy = metrics.accuracy_score(y_test, predicted) * 100
    return accuracy, predicted



#this is done in two for loops, irespective of number of params
#for exmple if there are total 4 params list then it will return all combination in 2 loops
def get_list_of_param_comination(list_of_param, param_names):
    list_of_param_comination = []
    for each in list(product(*list_of_param)):
        comb = {}
        for i in range(len(list_of_param)):
            comb[param_names[i]] = each[i]
        list_of_param_comination.append(comb)
    return list_of_param_comination


## hparams tuning function as per assignment 3
def tune_hparams(X_train, y_train, X_dev, y_dev, list_of_all_param_combination, model_type):
    best_accuracy = -1
    for hparams in list_of_all_param_combination:
        model = train_model(X_train=X_train, y_train=y_train, model_params=hparams, model_type=model_type)
        val_accuracy, _ = predict_and_eval(model, X_dev, y_dev)
        if val_accuracy > best_accuracy:
            best_hparams = hparams
            best_model = model
            best_accuracy = val_accuracy
    return best_hparams, best_model, best_accuracy
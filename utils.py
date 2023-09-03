from sklearn.model_selection import train_test_split
from sklearn import svm, datasets, metrics

def read_digits():
    digits = datasets.load_digits()
    X = digits.images
    y = digits.target
    return X, y

def preprocess_data(data):
    n_samples = len(data)
    data = data.reshape((n_samples, -1))
    return data

def split_train_dev_test(X, y, test_size, dev_size):
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=test_size + dev_size, random_state=1)
    X_dev, X_test, y_dev, y_test = train_test_split(X_temp, y_temp, test_size=dev_size / (test_size + dev_size), random_state=1)
    return X_train, X_test, X_dev, y_train, y_test, y_dev

def train_model(x, y, model_params, model_type="svm"):
    if model_type == "svm":
        clf = svm.SVC
    model = clf(**model_params)
    model.fit(x, y)
    return model

def evaluate_model(model, X_data, y_data):
    predicted = model.predict(X_data)
    classification_report = metrics.classification_report(y_data, predicted)
    confusion_matrix = metrics.confusion_matrix(y_data, predicted)
    return predicted, classification_report, confusion_matrix

def predict_and_eval(model, X_data, y_data):
    predicted, classification_report, _ = evaluate_model(model, X_data, y_data)
    return predicted, classification_report

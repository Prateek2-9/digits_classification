from sklearn.model_selection import train_test_split
from sklearn import svm, datasets

def read_digits():
    digits = datasets.load_digits()
    X = digits.images
    y = digits.target
    return X,y

# we will put all utils here
def preprocess_data(data):
    # flatten the images
    n_samples = len(data)
    data = data.reshape((n_samples, -1))
    return data

# Split data into 50% train and 50% test subsets
def split_data(x,y, test_size, random_state=1):
    X_train, X_test, y_train, y_test = train_test_split(
    x, y, test_size=0.5, shuffle=False, random_state=random_state
    )
    return X_train, X_test, y_train, y_test

# train the model of choice with model parameter
def train_model(x, y, model_params, model_type="svm"):
    if model_type=="svm":
        # Create a classifier: a support vector classifier
        clf = svm.SVC
    model = clf(**model_params)
    # train the model 
    model.fit(x,y)
    return model


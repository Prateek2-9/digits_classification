from flask import Flask,request
from joblib import  load
import numpy as np
app = Flask(__name__)

@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"

@app.route('/user/<username>')
def profile(username):
    return f'{username}\'s profile'


@app.route("/sum/<x>/<y>")
def sum_two_numbers(x,y):
    res = int(x) + int(y)
    return f"sum of {x} and {y} is {res}"

# model = load("models/production_model.joblib")

# @app.route("/predict", methods = ['POST'])
# def predict_fn():
#     input_data = request.get_json()
#     x = input_data['x']
#     y = input_data['y']
#     # res = int(x) + int(y)
#     return f"sum of {x} and {y} is {x + y}"

@app.route("/predict", methods = ['POST'])
def predict_fn():
    input_data = request.get_json()
    img1 = input_data['img1']
    img2 = input_data['img2']

    img1 = list(map(float, img1))
    img2 = list(map(float, img2))

    img1 = np.array(img1).reshape(1,-1)
    img2 = np.array(img2).reshape(1,-1)

    model = load("production_model.joblib")
    predicted1 = model.predict(img1)
    predicted2 = model.predict(img2)
    if predicted1[0] == predicted2[0]:
        print("Both images are of same digit")
        return "True"
    else:
        print("Both images are not of same digit")
        return "False"

if __name__ == '__main__':
    app.run(host = '0.0.0.0', port = 80)

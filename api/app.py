from flask import Flask,request

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


@app.route("/predict", methods = ['POST'])
def predict_fn():
    input_data = request.get_json()
    x = input_data['x']
    y = input_data['y']
    # res = int(x) + int(y)
    return f"sum of {x} and {y} is {x + y}"
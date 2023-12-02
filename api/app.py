from flask import Flask,request
#from joblib import  load
import numpy as np
from PIL import Image
import numpy as np
from utils import preprocess_data
import joblib
import json

from flask import Flask, request, jsonify
from joblib import load

app = Flask(__name__)

def load_model(model_type):
    roll_no = "m22aie214"  

    if model_type == 'svm':
        model_path = f"models/{roll_no}_svm.joblib"
    elif model_type == 'lr':
        model_path = f"models/{roll_no}_lr_liblinear.joblib"
    elif model_type == 'tree':
        model_path = f"models/{roll_no}_tree.joblib"
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    loaded_model = load(model_path)
    return loaded_model

@app.route("/predict/<string:model_type>", methods=["POST"])
def predict_model(model_type):
    try:
        
        model = load_model(model_type)

        
        input_data = request.json.get("image")
        
        
        processed_data = preprocess_data(input_data)

        
        prediction = model.predict(processed_data)

       
        return jsonify({"prediction": int(prediction[0])})

    except Exception as e:
        # Handle any exceptions
        return jsonify({"error": str(e)}), 500

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

# @app.route("/predict", methods = ['POST'])
# def predict_fn():
#     input_data = request.get_json()
#     img1 = input_data['img1']
#     img2 = input_data['img2']

#     img1 = list(map(float, img1))
#     img2 = list(map(float, img2))

#     img1 = np.array(img1).reshape(1,-1)
#     img2 = np.array(img2).reshape(1,-1)

#     model = load("production_model.joblib")
#     predicted1 = model.predict(img1)
#     predicted2 = model.predict(img2)
#     if predicted1[0] == predicted2[0]:
#         print("Both images are of same digit")
#         return "True"
#     else:
#         print("Both images are not of same digit")
#         return "False"

# if __name__ == '__main__':
#     app.run(host = '0.0.0.0', port = 80)


# Define a route to handle image upload and prediction
@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file part"})

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"error": "No selected file"})

    if file:
        image = _read_image(Image.open(file))
        model_path = "models/best_model_C-1_gamma-0.001.joblib"
        model = joblib.load(model_path)
        prediction = model.predict(image)
        return jsonify({"prediction": str(prediction[0])})
    else:
        return jsonify({"error": "Invalid file format"})


@app.route("/prediction", methods=["POST"])
def prediction():
    data_json = request.json
    if data_json:
        data_dict = json.loads(data_json)
        image = np.array([data_dict["image"]])
        model_path = "models/best_model_C-1_gamma-0.001.joblib"
        model = joblib.load(model_path)
        try:
            prediction = model.predict(image)
            return jsonify({"prediction": str(prediction[0])})
        except Exception as e:
            return jsonify({"error": str(e)})
    else:
        return jsonify({"error": "Invalid data format"})

if __name__ == "__main__":
    print("server is running")
    app.run(host="0.0.0.0", port=8000)
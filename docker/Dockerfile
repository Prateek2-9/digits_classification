# FROM ubuntu:22.04
FROM python:3.9.17

# Copy code folder
COPY . /digits/
# RUN apt-get update
# RUN apt-get install -y python3 python3-pip
RUN pip3 install -r /digits/requirements.txt

# Declare a VOLUME for the model directory
VOLUME /digits/models

WORKDIR /digits

# CMD ["pytest"]

# Entry point: Run the training script
CMD ["python", "plot_digits_classification.py", "5", "0.2", "0.2", "/digits/models/production_model.joblib", "svm"]
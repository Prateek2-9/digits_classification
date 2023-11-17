# FROM ubuntu:22.04
FROM python:3.9.17
WORKDIR /digits
# copy code folder
COPY . /digits/
# RUN apt-get update
# RUN apt-get install -y python3 python3-pip
#RUN pip3 install -r /digits/requirements.txt
RUN pip3 install --no-cache-dir -r /digits/requirements.txt
# setting work dir
#WORKDIR /digits
#creating volume for mounting on host
#VOLUME /digits/models
# to run pytest 
# CMD ["pytest"]
# FOR flask api run 
#ENV FLASK_APP=/digits/api/app.py
# FOR flask api run
#CMD ["flask","run","--host=0.0.0.0"]
ENV FLASK_APP=api/app
CMD ["python", "api/app.py"]
# running py script to train model
#ENTRYPOINT ["python","plot_digits_classification.py"]
# run exp.py to train svm model
#CMD ["python", "exp.py", "--total_run", "1", "--dev_size", "0.2", "--test_size", "0.2", "--model_type", "svm"]
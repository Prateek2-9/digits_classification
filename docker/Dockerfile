# FROM ubuntu:22.04
FROM python:3.9.17
# copy code folder
COPY . /digits/
# RUN apt-get update
# RUN apt-get install -y python3 python3-pip
RUN pip3 install -r /digits/requirements.txt
#set work dir
WORKDIR /digits
#create volume to mount it on host
VOLUME /digits/models
#run pytest 
CMD ["pytest"]
#run python script to train model
ENTRYPOINT ["python","plot_digits_classification.py"]
# run plot_digits_classification.py to train svm model
# CMD ["python", "exp.py", "--total_run", "1", "--dev_size", "0.2", "--test_size", "0.2", "--model_type", "svm"]


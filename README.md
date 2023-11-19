how to SETUP: follow below steps

conda create -n plot_digits_classification python=3.9   // creating env
conda activate plot_digits_classification // activating env

installing project requirements:
pip install -r requirements.txt 

how to RUN:

python plot_digits_classification.py

how to run from dockersh:

./docker_run.sh     

how to build and run docker :

docker build -t digits:v1 -f docker/Dockerfile .  
docker run -it digits:v1   
docker run -it -p 5001:5000 digits:v1           

how to run flask : goto api folder

flask run
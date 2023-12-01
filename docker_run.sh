#docker build -t digits:v1 -f docker/Dockerfile .
#docker run -v /Users/prateeksingh/Desktop/digits_trained_model:/digits/models digits:v1 --total_run 1 --dev_size 0.2 --test_size 0.2 --model_type 'svm' 
#!/bin/bash

# Build the Docker image (if not already built)
docker build -t digits:v2 -f docker/Dockerfile .

# Run the Docker container with specific parameters
docker run digits:v2 \
  python plot_digits_classification.py \
  --total_run 5 \
  --dev_size 0.2 \
  --test_size 0.3 \
  --prod_model_path production_model.joblib \
  --model_type svm


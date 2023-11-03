docker build -t digits:v1 -f docker/Dockerfile .
docker run -v /Users/prateeksingh/Desktop/digits_trained_model:/digits/models digits:v1
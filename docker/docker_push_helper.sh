#!/bin/bash

# Log in to your Azure Container Registry
az acr login --name prateekprateekmlops23

# Build and push the Dependency Image
docker build -t prateekprateekmlops23/dependency-image:latest -f DependencyDockerfile .
docker push prateekprateekmlops23/dependency-image:latest
# Docker login
docker login prateekprateekmlops23.azurecr.io
# Build and push the Final Image
docker build -t prateekprateekmlops23/final-image:latest -f FinalDockerfile .
docker push prateekprateekmlops23/final-image:latestdocker_push_helper.sh
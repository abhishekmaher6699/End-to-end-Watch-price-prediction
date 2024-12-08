#!/bin/bash
# Log everything to start_docker.log
exec > /home/ubuntu/start_docker.log 2>&1

echo "Logging in to ECR..."
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 529088280615.dkr.ecr.us-east-1.amazonaws.com

echo "Pulling Docker image..."
docker pull 529088280615.dkr.ecr.us-east-1.amazonaws.com/watch-price-predictor:latest

echo "Checking for existing container..."
if [ "$(docker ps -q -f name=watch-price-predictor)" ]; then
    echo "Stopping existing container..."
    docker stop watch-price-predictor
fi

if [ "$(docker ps -aq -f name=watch-price-predictor)" ]; then
    echo "Removing existing container..."
    docker rm watch-price-predictor
fi

echo "Starting new container..."
docker run -d -p 8501:8501 --name watch-price-predictor 529088280615.dkr.ecr.us-east-1.amazonaws.com/watch-price-predictor:latest

echo "Container started successfully."
# Use the official Python base image
FROM python:3.11

# Clone the repository
RUN git clone https://github.com/mistralai/mistral-finetune.git

# Set the working directory
WORKDIR /mistral-finetune

# Update pip, install torch and other dependencies
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy the 7B.yaml file
COPY 7B.yaml /mistral-finetune/example/7B.yaml

# Script to prepare the training data
COPY dataset_processing.py /mistral-finetune/dataset_processing.py
FROM python:latest

# Python library dependencies
RUN apt-get update && apt-get install -y swig vim awscli rsyslog cron jq

# Pip packages
ADD requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Add repository scripts
WORKDIR /home/python
ADD ./ ./

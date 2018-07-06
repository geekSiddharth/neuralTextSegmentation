FROM tensorflow/tensorflow:latest-gpu-py3

RUN apt-get update && apt-get install -y wget unzip nano

RUN pip install keras nltk

WORKDIR  /src/
CMD bash
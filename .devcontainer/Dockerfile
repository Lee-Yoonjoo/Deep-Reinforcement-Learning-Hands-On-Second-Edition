FROM ubuntu:20.04

RUN apt-get upgrade && apt-get update && apt install software-properties-common -y

RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt-get update
RUN apt-get install python3.7 -y && apt install wget -y && apt install python3.7-distutils -y
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.7 1

RUN python3 --version

RUN wget https://bootstrap.pypa.io/get-pip.py

RUN python3 get-pip.py

WORKDIR /app
COPY requirements.txt requirements.txt

RUN pip install --default-timeout=300 -r requirements.txt


RUN pip install swig
RUN pip install box2d-py

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

ENV PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

FROM ubuntu:20.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get upgrade -y && apt-get install -y software-properties-common

RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt-get update

RUN apt-get install python3.7 -y && apt-get install python3.7-dev -y && apt install wget -y && apt install python3.7-distutils -y
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.7 1

RUN python3 --version

RUN wget https://bootstrap.pypa.io/get-pip.py

RUN python3 get-pip.py

WORKDIR /app
COPY requirements.txt requirements.txt

RUN apt-get install python-dev gcc mono-mcs g++ -y
RUN apt-get install build-essential -y

RUN pip install gym
RUN apt-get install swig build-essential python-dev python3-dev -y
RUN pip install opencv-python==4.5.1.48
RUN pip install box2d-py

RUN pip install --default-timeout=300 -r requirements.txt



RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

ENV PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
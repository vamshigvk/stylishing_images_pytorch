FROM alpine

FROM python:3.7

COPY . /usr/app/
EXPOSE 5000
WORKDIR /usr/app/
RUN pip install --upgrade pip

RUN apt install libtinfo5


RUN pip3 install --upgrade pip
RUN python -m pip3 install torch==1.7.1 -f https://download.pytorch.org/whl/torch_stable.html
RUN pip3 install  torchvision===0.8.2 -f  https://download.pytorch.org/whl/torch_stable.html
RUN pip3 install opencv-python==4.1.0.25
RUN pip3 install numpy==1.16.3
RUN pip3 install  matplotlib==3.0.3
CMD python stylize.py

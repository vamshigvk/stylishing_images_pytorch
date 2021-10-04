FROM alpine
FROM python:3.7
RUN apt update
RUN apt-get install -y libncurses5-dev libncursesw5-dev libtinfo5
RUN apt-get install -y unzip

COPY . /usr/app/
EXPOSE 5000
WORKDIR /usr/app/
RUN pip install --upgrade pip
RUN wget https://web.eecs.umich.edu/~justincj/models/vgg16-00b39a1b.pth -O models/vgg16-00b39a1b.pth
RUN wget http://images.cocodataset.org/zips/val2017.zip   -O dataset/coco.zip
RUN unzip dataset/coco.zip -d dataset/

RUN python -m pip install torch==1.7.1 -f https://download.pytorch.org/whl/torch_stable.html
RUN pip install  torchvision===0.8.2 -f  https://download.pytorch.org/whl/torch_stable.html
RUN pip install opencv-python==4.1.0.25
RUN pip install numpy==1.16.3
RUN pip install  matplotlib==3.0.3
RUN pip install boto3
#RUN pip install pathlib
#RUN pip install glob2
CMD python train.py

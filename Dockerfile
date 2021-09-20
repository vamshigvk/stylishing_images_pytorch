FROM continuumio/anaconda3:4.4.0
COPY . /usr/app/
EXPOSE 5000
WORKDIR /usr/app/
RUN python -m pip install torch==1.7.1 -f https://download.pytorch.org/whl/torch_stable.html
RUN pip3 install  torchvision===0.8.2 -f  https://download.pytorch.org/whl/torch_stable.html
RUN pip install opencv-python==4.1.0
RUN pip install numpy==1.16.3
RUN pip3 install  matplotlib==3.0.3


CMD python stylize.py
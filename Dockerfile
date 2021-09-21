FROM continuumio/anaconda3:4.4.0
COPY . /usr/app/
EXPOSE 5000
WORKDIR /usr/app/
RUN apt-get update
RUN apt-get install -y libgl1-mesa-dev

RUN conda update conda -y
RUN conda create -y -n faststyle python=3.7


ENV PATH=~/anaconda3/bin:$PATH


RUN echo "source activate faststyle" >> ~/.bashrc
SHELL ["/bin/bash", "--login", "-c"]


RUN pip install --upgrade pip
RUN python -m pip install torch==1.7.1 -f https://download.pytorch.org/whl/torch_stable.html
RUN pip install  torchvision===0.8.2 -f  https://download.pytorch.org/whl/torch_stable.html
RUN pip install opencv-python==4.1.0
RUN pip install numpy==1.16.3
RUN pip install  matplotlib==3.0.3
CMD python stylize.py

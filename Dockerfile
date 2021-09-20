FROM continuumio/anaconda3:4.4.0
COPY . /usr/app/
EXPOSE 5000
WORKDIR /usr/app/

RUN conda update conda -y
RUN conda create -y -n faststyle python=3.7

RUN ls /opt/conda/envs/
RUN export PATH=/opt/conda/envs/faststyle/bin:$PATH


RUN source activate faststyle
RUN pip3 install --upgrade pip
RUN python -m pip3 install torch==1.7.1 -f https://download.pytorch.org/whl/torch_stable.html
RUN pip3 install  torchvision===0.8.2 -f  https://download.pytorch.org/whl/torch_stable.html
RUN pip3 install opencv-python==4.1.0
RUN pip3 install numpy==1.16.3
RUN pip3 install  matplotlib==3.0.3
CMD python stylize.py

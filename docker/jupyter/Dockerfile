FROM tensorflow/tensorflow:latest-py3-jupyter


COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt && rm requirements.txt

RUN useradd -ms /bin/bash jovyan

USER jovyan

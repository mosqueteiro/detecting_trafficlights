FROM tensorflow/tensorflow:latest-gpu-py3-jupyter

ADD .. /app
WORKDIR /app

RUN pip install -r requirements.txt

CMD ['jupyter', 'lab']

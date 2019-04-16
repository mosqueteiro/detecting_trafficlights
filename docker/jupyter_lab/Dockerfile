FROM tensorflow/tensorflow:latest-gpu-py3-jupyter

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

# Start the jupyter notebook
ENTRYPOINT ["jupyter", "lab", "--ip=*"]

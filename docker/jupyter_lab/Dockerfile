FROM tensorflow/tensorflow:latest-py3-jupyter

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt && rm requirements.txt

# Create a new system user
RUN useradd -ms /bin/bash jovyan

# Change to this new user
USER jovyan

# Set the container working directory to the user home folder
WORKDIR /home/jovyan

# Start the jupyter notebook
ENTRYPOINT ["jupyter", "lab", "--ip=*"]

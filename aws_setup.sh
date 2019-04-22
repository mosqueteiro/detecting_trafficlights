#!/bin/bash

PROJECT=`basename "$PWD"`
PROJECT="$(echo `basename $PWD` | awk '{print tolower($0)}')"

# make directory to mount EBS datasets to
mkdir -p data/coco/train2017
mkdir -p data/coco/val2017

# mount and own new directories
sudo mount /dev/xvdf data/coco/train2017
sudo mount /dev/xvdg data/coco/val2017
sudo chown -R $USER:$USER data/coco/train2017
sudo chown -R $USER:$USER data/coco/val2017

# installing docker-compose
sudo curl -L "https://github.com/docker/compose/releases/download/1.24.0/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose
sudo ln -s /usr/local/bin/docker-compose /usr/bin/docker-compose

# starting up docker services and installing requirements
bash start_services.sh
docker exec ${PROJECT}_jupyter_flow_1 bash -c \
"apt update && apt install -y libpq-dev"
docker exec ${PROJECT}_jupyter_flow_1 \
pip install -r docker/jupyter/requirements.txt

docker-compose stop
docker-compose start
docker exec ${PROJECT})_jupyter_flow_1 -c \
"cd src && python data_pipeline.py"

docker exec ${PROJECT}_jupyter_flow_1 jupyter notebook list

# tensorboard setup
mkdir tb_logs
docker exec -d ${PROJECT}_jupyter_flow_1 tensorboard --logdir=tb_logs

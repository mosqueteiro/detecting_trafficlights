#!/bin/bash

sudo curl -L "https://github.com/docker/compose/releases/download/1.24.0/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose
sudo ln -s /usr/local/bin/docker-compose /usr/bin/docker-compose
# docker-compose up -d
# docker exec detecting_trafficlights_jupyter_flow_1 \
# pip install -r docker/jupyter/requirements.txt
# docker exec detecting_trafficlights_jupyter_flow_1 jupyter notebook list
bash start_services.sh

# download train2017 images
# sudo apt install unzip
# mkdir -p data/coco/train2017
# curl -O http://images.cocodataset.org/zips/train2017.zip
# unzip train2017.zip -d data/coco/
# rm train2017.zip

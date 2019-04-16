#!/bin/bash

# install docker
curl -sSL https://get.docker.com/ | bash

# add ubuntu to docker user group
sudo usermod -aG docker ubuntu
# relog user to apply group add w/o logging out
exec sudo su -l $USER

# add docker-compose
sudo curl -L "https://github.com/docker/compose/releases/download/1.24.0/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose
sudo ln -s /usr/local/bin/docker-compose /usr/bin/docker-compose

docker-compose up -d
docker exec detecting_trafficlights_jupyter_flow_1 \
pip install -r docker/jupyter/requirements.txt
docker exec detecting_trafficlights_jupyter_flow_1 jupyter notebook list

# download train2017 images
# sudo apt install unzip
# mkdir -p data/coco/train2017
# curl -O http://images.cocodataset.org/zips/train2017.zip
# unzip train2017.zip -d data/coco/train2017/
# rm train2017.zip

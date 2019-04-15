#!/bin/bash

# install docker
curl -sSL https://get.docker.com/ | bash

# add ubuntu to docker user group
sudo usermod -aG docker ubuntu

# add docker-compose
sudo curl -L "https://github.com/docker/compose/releases/download/1.24.0/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose
sudo ln -s /usr/local/bin/docker-compose /usr/bin/docker-compose

# clone repo
git clone https://github.com/mosqueteiro/detecting_trafficlights.git

# download train2017 images
sudo apt install unzip
mkdir -p detecting_trafficlights/data/coco/train2017
# cd detecting_trafficlights/data/coco/train2017
curl -O http://images.cocodataset.org/zips/train2017.zip
unzip train2017.zip -d detecting_trafficlights/data/coco/train2017/


# exit terminal now

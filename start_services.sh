#!/bin/bash

docker-compose up -d
docker exec detecting_trafficlights_jupyter_flow_1 \
pip install -r docker/jupyter/requirements.txt
docker exec detecting_trafficlights_jupyter_flow_1 jupyter notebook list

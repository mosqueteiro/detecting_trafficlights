#!/bin/bash

PROJECT="$(echo `basename $PWD` | awk '{print tolower($0)}')"

docker-compose up -d

docker exec ${PROJECT}_jupyter_flow_1 jupyter notebook list

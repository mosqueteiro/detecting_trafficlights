#!/bin/bash

# install docker
curl -sSL https://get.docker.com/ | bash

# add ubuntu to docker user group
sudo usermod -aG docker ubuntu

# exit terminal now

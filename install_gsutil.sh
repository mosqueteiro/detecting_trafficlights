#!/bin/bash

# add Cloud SDK repo to package source
export CLOUD_SDK_REPO="cloud-sdk-$(lsb_release -c -s)"
echo "deb http://packages.cloud.google.com/apt $CLOUD_SDK_REPO main" | sudo tee -a /etc/apt/sources.list.d/google-cloud-sdk.list

# import Google Cloud public key
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -

# Update and install Cloud SDK
sudo apt-get update && sudo apt-get install google-cloud-sdk

# Have to initialize
gcloud init

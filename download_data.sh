#!/bin/bash

# run this from the top of the git repo directory
mkdir -p data/coco/annotations


# downloading the annotations file
gsutil -m rsync gs://images.cocodataset.org/annotations data/coco/


# unzip annotations
unzip data/coco/*.zip -d data/coco/

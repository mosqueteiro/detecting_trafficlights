# Detecting Trafficlights

## Table of contents
1. [Introduction](#introduction)
2. [Goal of project](#goal-of-project)
3. [Description of Data](#description-of-data)
4. [EDA](#exploratory-data-analysis)
5. [Modeling Methodology](#modeling-methodology)
6. [Results](#results)
7. [Future Work](#future-work)


Training models to detect traffic lights with grayscale images

## Introduction
As autonomous vehicle technology continues to drive ahead greater computing requirements are levied. Finding ways to reduce needed compute power while maintaining high accuracy is important to making autonomous vehicles a reality. Color images and video frames are generally represented by height, width, and (3) color channels. If the image is reduced to grayscale the color channels are reduced to one potentially reducing the compute power needed for a model. This could allow smaller models to be used as an ensemble or allow a bigger model to fit into a smaller space.

## Goal of project
The goal of this project is to train models on small (100x100), grayscale images of traffic lights with accuracy above 97% and compare them to models trained on color images based on model size, compute speed, accuracy, and AUC.

## Description of data
Images are from the Common Objects in Context (COCO) dataset. COCO is a large-scale object detection, segmentation, and captioning dataset. COCO has 330K images (>200K labeled), 1.5 million object instances, and 80 object categories. They host annual image detection competitions and so datasets are categorized by the year of competition. Further separation is added between train and validation sets. The subset of images used here are filtered on traffic light images and non-traffic light, street-context from the 2017 dataset. Each dataset comes with a json file with tables for categories, images, and annotations.

![raw_gray](images/raw_gray.png)
![small_gray](images/small_gray.png)


## Exploratory Data Analysis


## Modeling Methodology


## Results


## Future Work

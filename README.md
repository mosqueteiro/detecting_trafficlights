# Building Deployable Data Science Environments  

The `master` branch on this repo provides an exploration of the environments created in the project. To deploy one of the environments read this `README.md`, the `setup.md`, and switch to the appropriate branch for the environment you would like to deploy.

## Table of contents
1. [Introduction](#introduction)
2. [Goal of project](#goal-of-project)
3. [Description of Data](#description-of-data)
4. [Docker](#docker)
6. [Tensorflow GPU with Jupyter](#tensorflow-gpu-with-jupyter-notebooks)
7. [PostgreSQL](#postgresql)
  * [Database class](#database-class)
7. [Docker Compose](#docker-compose)
8. [Amazon Web Services](#amazon-web-services)
4. [EDA](#exploratory-data-analysis)
5. [Modeling Methodology](#modeling-methodology)
6. [Results](#results)
7. [Future Work](#future-work)



## Introduction
Through the course of working on data science projects different package requirements are needed for different projects. Having a large catch-all environment may be able to satisfy most requirements but will be bulky and take up unnecessary space. Further, some specialized requirements will not be met by this strategy. Ultimately, with different machines running and testing the project at different times a standard environment will need to be shared between machines.  

A self-contained environment is a good solution to this problem. Requirements can be specified in a file and the environment built out to match between different machines. Another benefit of this strategy allows the project to scale up to bigger, and more powerful machines as needed. Also, additional testing and, later, solution deployment will benefit from a well defined environment keeping an entire team working with the same tools from the beginning.  

[Back to Top](#Table-of-Contents)

## Goal of project
The goal of this project is to develop a data science environment for an image recognition system that can be deployed onto GPU instances on AWS. This allows rapid model prototyping, concurrent model testing, scalability between machines, and future system deployment.  

[Back to Top](#Table-of-Contents)

## Description of data
Images are from the Common Objects in Context (COCO) dataset. COCO is a large-scale object detection, segmentation, and captioning dataset. COCO has 330K images (>200K labeled), 1.5 million object instances, and 80 object categories. They host annual image detection competitions and so datasets are categorized by the year of competition. Further separation is added between train and validation sets. The subset of images used here are filtered on traffic light images and non-traffic light, street-context from the 2017 dataset. Each dataset comes with a json file with tables for categories, images, and annotations.  

[Back to Top](#Table-of-Contents)


## Docker

Docker is a computer program that runs virtualization directly on top of the operating system. This allows applications to be run with the speed of a native install without the length and difficulties of a manual install. Apps run through docker are called containers. Their build and use is the same between machines allowing cross-platform development and deployment.

In this project containers for Tensorflow and PostgreSQL are used to provide needed tools for an image recognition project. The main prerequisite is having docker installed on the machine. For GPU computing, the Nvidia graphics driver and Nvidia-docker app are, additionally, needed. Everything else is built into the docker containers.

Many base docker images for tools like Tensorflow and PostgreSQL are provided by the teams that build the tools. These can be customized with a _Dockerfile_ that begins with the base image and adds layers of functionality or project requirements on top of it. The _Dockerfile_ used for Tensorflow in this project adds all the packages listed in the _requirements.txt_ file and any dependencies those packages may need.

Tensorflow _Dockerfile_:  
```docker
FROM tensorflow/tensorflow:latest-py3-jupyter


COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt && rm requirements.txt

RUN useradd -ms /bin/ jovyan

USER jovyan

CMD ["bash", "-c", "source /etc/bash.bashrc && jupyter notebook --notebook-dir=/tf --ip 0.0.0.0 --no-browser"]
```

[Back to Top](#Table-of-Contents)


## Tensorflow-GPU with Jupyter Notebooks

The Tensorflow team provides docker images for different versions of tensorflow and support for gpu functionality as well as Jupyter Notebooks. The gpu functionality takes what is normally a long and grueling process of installing CUDA, cudnn, and other supporting libraries and distills it into the build of the tensorflow container. This in and of itself is an enormous time saver and allows rapid deployment of models to gpu-enhanced machines.

Adding Jupyter Notebooks to the container allows the machine to run a notebook server. With the right security parameters on a cloud computer, this allows access to the notebook server from any browser with the correct credentials. Jupyter notebooks are especially helpful on cloud computers which generally do not have a way to show data visualizations for exploratory data analysis and model results as well as including human-readable context for the steps taken through the project. Jupyter notebooks are generally accessed using the machine IP address at port `8888` with a token that can be printed using `jupyter notebook list` passed as a command to the docker container with `docker exec ...`.

For training large models that take hours or more, I would suggest using `screen` and `bash` instead of a Jupyter Notebook as these options allow detachment from the program while it trains while the notebook will need to be kept open to keep kernels running. From `ssh` _terminal_ at the current machine use:

```bash
$ screen -S training
$ docker exec -it scalable_ds_envs_jupyter_flow_1 bash
container@~$ python path/file.py
# ipython can also be used
```

Another tool useful with tensorflow is `Tensorboard`. Tensorboard tracks and visualizes the metrics of a tensorflow model, it's weights and gradients, and a model graph while the model is training. Tensorboard runs a local server that is accessed through a web browser. The model training must use the `Tensorboard` callback to utilize this tool. Again, with the right security parameters, the Tensorboard server can be accessed from any web browser outside of the system. This is useful to quickly check on the progress of a larger model training while being detached from the system. The Tensorboard server is started with `tensorboard --logdir=logs/` passed to the docker container using `docker exec ...`. It is accessed using the machine IP address at port `6006`.

[Back to Top](#Table-of-Contents)


## PostgreSQL

The PostgreSQL container provides a quick implementation of the PostgreSQL database application. The PostgreSQL team provides docker hookpoints to build the database at the time the container is run by copying sql and bash scripts into the `docker-entrypoint-initdb.d` directory which runs any files present at this location. This provides a quick and portable way to use postgres and build out a database on multiple machines.

The COCO dataset uses object localization annotations where an image generally has mulitple annotations that belong to different categories. The SQL database provides a powerful way to query and subset the images an annotations as needed.

### Database class
 To manage all the images and annotations a PostgreSQL database is created to hold image metadata and image annotation data. As images are downloaded their local path is recorded into the database for quick access later on. A data pipeline class helps to connect python to the database. The data pipeline is built on top of `psycopg2`.

```python
class DataPipeline(object):
    def __init__(self, dataset, user, host, db_prefix='coco_', data_dir=None):
        ...
        self.connect_sql(dbname=dbname, user=user, host=host)

    def __del__(self):
        self.cursor.close()
        self.connxn.close()

    def __enter__(self):
        return self
    def __exit__(self, exc_type, exc_value, traceback):
        self.__del__()

    def connect_sql(self, dbname, user='postgres', host='/tmp'):
        print('Connecting to PostgreSQL server....')
        self.connxn = connect(dbname=dbname, user=user, host=host)
        print('\tConnected.')
        self.cursor = self.connxn.cursor()
```
A class for building the database and loading and inserting all the metadata helps to keep the database structure the same between datasets such as train and validation as well as datasets from previous or future years. New tables filtered down to the current scope of the project make access to data quick and the ability to create new views allow the project scope to expand as needed.

```python
class BuildDatabase(DataPipeline):
    ...
    def build_sql(self, coco_dir):
      ...
    def load_json(self, coco_dir):
      ...
    def create_tables(self, file):
      ...
    def insert_into_table(self, table_name, dict_lst, pages=100):
      ...
```
The `QueryDatabase` class facilitates running queries that return into a `pandas.DataFrame`. From there a list of image paths can be served to the models' `ImageDataGenerator` using the `get_images` method which checks if the image has been downloaded yet (downloading if necessary). As images are downloaded their absolute path is recorded and updated into the SQL database.

```python
class QueryDatabase(DataPipeline):
    ...
    def query_database(self, query=None):
      ...
    def download(self, image_id, image_name, image_url):
      ...
    def update_sql(self, table, id, field, value):
      ...
    def get_images(self):
      ...
```  

[Back to Top](#Table-of-Contents)


## Docker Compose

The compose tool by Docker allows simple orchestration of multiple docker containers through a single `docker-compose.yml` file. Containers created using Docker Compose are setup on their own docker network simplifying communication between the containers. For instance, the `DataPipeline` class running in the Tensorflow container uses the name of the PostgreSQL container as the host name to connect to it. A single command builds and runs the entire app with all the containers

```bash
.../scalable_DS_envs$ docker-compose up
```

[Back to Top](#Table-of-Contents)


## Amazon Web Services

Amazon Web Services (AWS) provide powerful cloud tools for data science projects, especially when the project scope goes beyond what can be accomplished on a local machine. In the context of image recognition systems, GPU machines are needed to train models in a short enough time to be useful. AWS EC2 has a variety of machine instances available including a number of GPU enabled machines. This allows project scaling depending on current needs. For instance, when first setting up the project data, before deploying the environment and during EDA, a less expensive smaller machine instance can be used.

Elastic Block Storage (EBS) provides an ideal way to store data that can be extended to multiple machines with ease. In this project, during the initial setup, EBS volumes are created and loaded with the project data. A snapshot of the volume is then created after which the volume can be terminated. The snapshot can then be loaded into any subsequent (and multiple) EBS volumes and attached to another machine. Storing the data as a snapshot minimizes storage cost because snapshots are charged at the AWS S3 rate rather than the much higher EBS volume storage rate.

A guide to setting these tools up is provided in the [setup](setup.md) file.

[Back to Top](#Table-of-Contents)


## Exploratory Data Analysis
Within this database there are ~27,000 street-context images of which ~4,000 contain traffic lights. The size of the traffic lights in each image vary and some images have multiple traffic lights in them. The street-context subset is filtered on road vehicle and outdoor supercategories. Image sizes range from (52-640) x (59-640). The images in the COCO dataset can even be challenging for humans to identify.

This image containes four traffic lights, can you spot them all?  
![tl_1](images/trafficlight_1.png)

This image, while part of a roadway context, does not contain any traffic lights.  
![no_tl_1](images/no_trafficlight_1.png)  

[Back to Top](#Table-of-Contents)

## Modeling Methodology
The current model used is modeled after [AlexNet](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf) an 8-layer convolutional neural network. The first (5) layers are convolutional layers that learn filters to apply to the image to make sense of what is in an image. The last (3) layers are fully-connected layers that take the filtered, simplified images and try to learn what makes up a traffic light. AlexNet was originally run in parallel on two GTX 580 graphics cards with 3GB of memory each.  

[Back to Top](#Table-of-Contents)

## Results
The first models do not improve much from the 56% on the validation data. The model is overfitting which could be addressed with more Dropout and more data, however, the validation accuracy will not likely improve much with this strategy. Due to the complexity of the image data it is likely that better results would be obtained training the model first on an easier dataset before training using the COCO dataset. A more complex model may also be helpful for this dataset.

![tb_results](images/tensor_board_results.png)

[Back to Top](#Table-of-Contents)

## Future Work  
* copy and adapt Tensorflow Dockerfile source to use conda and install needed dependencies within image rather than in shell script
* add tensorboard to Tensorflow Dockerfile
* PostgreSQL mount reusable database through EBS and snapshot rather than rebuilding for every new instance
* train on [ImageNet](http://www.image-net.org/) or [Google Open Images](https://storage.googleapis.com/openimages/web/index.html) first.
* experiment with ResNets, Inception models, and newer RCNN models


[Back to Top](#Table-of-Contents)

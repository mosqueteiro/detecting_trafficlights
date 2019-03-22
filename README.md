# Detecting Trafficlights

## Table of contents
1. [Introduction](#introduction)
2. [Goal of project](#goal-of-project)
3. [Description of Data](#description-of-data)
3. [SQL Database](#sql-database)
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


## SQL Database
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


## Exploratory Data Analysis
Within this database there are ~27,000 street-context images of which ~4,000 contain traffic lights. The size of the traffic lights in each image vary and some images have multiple traffic lights in them. The street-context subset is filtered on road vehicle and outdoor supercategories. Image sizes range from (52-640) x (59-640).

![raw_gray](images/raw_gray.png)
![small_gray](images/small_gray.png)

## Modeling Methodology



## Results


## Future Work

from data_pipeline import BuildDatabase


coco_dir = '../data/coco'
dataset = 'train2017'
user = 'mosqueteiro'
host = '/var/run/postgresql'
with BuildDatabase(dataset, user, host, data_dir=coco_dir) as build:
    build.build_sql(coco_dir)

    build.create_tables('street_tables.sql')

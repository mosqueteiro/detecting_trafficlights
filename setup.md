

## Initial Setup: Getting the data ready  
Initial setup of the data science environment should take into consideration the data that will be used. In this project I am using the COCO 2017 dataset. This includes training and validation data and annotations which take up 19 Gb and 1 Gb, respectively. I found it helpful to give each dataset its own storage volume and snapshot to be attached to new machine instances that are built.  

![aws_data_setup](images/aws_data_setup.gif)

On AWS, I instantiated a t2 machine with Amazon Linux on it and two additiional storage volumes with enough space to fit each dataset. The steps to set this up are as follows:  
1. Initialize EC2 instance with attached storage volumes
2. `ssh` into the instance
3. Mount the drives and create a file structure on them, The [AWS docs](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/ebs-using-volumes.html) are helpful for how to get these mounted with file systems.
4. Get ownership of this file system so we can read and write to it
   ```bash
   $ sudo chown storage/volume/path
   ```
4. Download the data and annotations  
   ```bash
   $ cd storage/volume/path
   $ wget http://images.cocodataset.org/zips/train2017.zip
   $ wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
   ```
5. Unzip the data and move it to the root of the storage volume so that when we mount it later we don't have extra directories to sort through
   ```bash
   $ unzip train2017.zip
   $ find train2017/ -name *.jpg exec mv {} ./ ;\
   $ rm -r train2017

   $ mkdir annotations
   $ unzip annotations_trainval2017.zip -d annotations/
   ```
7. Terminate the instance (storage volumes will persist if `Delete on Termination` isn't checked) and save the data as a snapshot.

   ![aws_snapshot](images/aws_snapshot.gif)

8. Terminate the data volumes.




## Starting the Data Science environment  
The first thing to do in an aws instance is to clone the repo:  
```bash
git clone https://github.com/mosqueteiro/detecting_trafficlights.git && \
cd detecting_trafficlights
```

Then the aws_setup.sh script:
```bash
sudo chmod +x aws_setup.sh
bash aws_setup.sh
```

## PostgreSQL
### Starting the postgres server in a container

Let's create & start up a container with the official postgres image. Here I've named the container `pgserv`, but you can call it anything.
```
$ docker run --name pgserv -d -p 5435:5432 -v "$PWD":/home/data postgres
```
- the `-d` flag means "run this container in the background"
- `-p 5435:5432` means "connect port 5435 from this computer (localhost) to the container's port 5432". This will allow us to connect to the Postgres server (which is inside the container) from services running outside of the container (such as python, as we'll see later).

In the future, you can start this container by using the `docker start` command
```bash
$ docker start pgserv
```

### Accessing the postgres terminal, [psql](http://postgresguide.com/utilities/psql.html)

`psql` is the command to open up a postgres terminal, and we need to run this command on inside the container. `docker exec` is the way to execute commands in a running container. See the documentation [here](https://docs.docker.com/engine/reference/commandline/exec/)
```
$ docker exec -it pgserv psql -U postgres
=# CREATE DATABASE yeah;
=# CREATE TABLE whatever ... ;
=# SELECT * FROM whatever
=# \q
```

### Loading data into the server from a local file

Say we have a database dump on our machine called `really_important.sql`, and we'd like to load it into our containerized postgres server and run some queries on it.

First, make sure the data file is in the folder that was mounted as a volume when you created the `pgserv` container. For example, if you ran the `docker run` command from `~`, make sure that `really_important.sql` is in some sub-folder of `~`.

Suppose the data is in `~/path/to/data_dump/really_important.sql`. We can access it from the container as follows:  

```
$ docker exec -it pgserv bash
# cd /home/data/path/to/data_dump/
# psql -U postgres
=# CREATE DATABASE new_database;
=# \q
# psql -U postgres new_database < really_important.sql;
# psql -U postgres new_database
=# \d
=# SELECT * FROM critical_table LIMIT 13;
```

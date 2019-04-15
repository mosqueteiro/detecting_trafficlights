

## Initial Setup
The first thing to do in an aws instance is to clone the repo:  
```bash
git clone https://github.com/mosqueteiro/detecting_trafficlights.git
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

#!/bin/bash

createdb -U postgres train2017
psql -U postgres train2017 < initdb.sql

createdb -U postgres val2017
psql -U postgres val2017 < initdb.sql

#!/bin/bash

createdb -U postgres train2017
psql train2017 < initdb.sql

createdb -U postgres val2017
psql val2017 < initdb.sql

#!/bin/bash

createdb train2017
psql train2017 < initdb.sql

createdb val2017
psql val2017 < initdb.sql

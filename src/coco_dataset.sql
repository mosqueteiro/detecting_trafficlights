
-- Title: coco_dataset.sql
-- Author: Nathan James
-- Date: 03/18/19
--
-- PostgreSQL build SQL database from COCO dataset and annotations
-- ### Prior steps ###
-- First create the database with $ sudo -u postgres -c "createdb <name>"
-- Then $ psql <name> < coco_dataset.sql


BEGIN;

CREATE TABLE categories (
  id INTEGER,
  name TEXT,
  supercategory TEXT
);

CREATE TABLE images (
  id INTEGER,
  file_name TEXT NOT NULL,
  coco_url TEXT NOT NULL,
  flickr_url TEXT NOT NULL,
  local_path TEXT,
  height INTEGER,
  width INTEGER,
  date_captured TIMESTAMP(0),
  license INTEGER
);

CREATE TABLE annotations (
  id BIGINT,
  image_id INTEGER,
  category_id INTEGER,
  bbox FLOAT[],
  area FLOAT,
  segmentation TEXT,
  iscrowd INT
);

CREATE TABLE license (
  id INTEGER,
  name TEXT,
  url TEXT
);


COMMIT;


-- clear out database for testing purposes
-- DROP SCHEMA public CASCADE;
-- CREATE SCHEMA public;
-- GRANT ALL ON SCHEMA public TO postgres;
-- GRANT ALL ON SCHEMA public TO public;

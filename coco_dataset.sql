-- PostgreSQL build SQL database from COCO dataset and annotations
--
-- Author: Nathan James
-- Date: 03/18/19
--
-- Builds the tables for a coco dataset database


BEGIN;

CREATE TABLE categories (
  id INTEGER PRIMARY KEY,
  name TEXT,
  supercategory TEXT
);

CREATE TABLE images (
  id INTEGER PRIMARY KEY,
  filename TEXT NOT NULL,
  coco_url TEXT NOT NULL,
  flickr_url TEXT NOT NULL,
  local_path TEXT,
  height INTEGER,
  width INTEGER,
  date_captured TIMESTAMP(0),
  license_id INTEGER
);

CREATE TABLE annotations (
  id INTEGER PRIMARY KEY,
  image_id INTEGER REFERENCES images(id),
  category_id INTEGER REFERENCES categories(id),
  bbox TEXT,
  area FLOAT,
  segmentation TEXT
);

CREATE TABLE license (
  id INTEGER PRIMARY KEY,
  name TEXT,
  url TEXT
);

-- ALTER TABLE ONLY images
--   ADD CONSTRAINT image_license_fkey
--   FOREIGN KEY (license_id) REFERENCES license(id);

COMMIT;

ANALYZE categories;
ANALYZE images;
ANALYZE annotations;
ANALYZE license;

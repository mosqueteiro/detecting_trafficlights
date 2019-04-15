-- Title: sql_constraints.sql
-- Author: Nathan James
-- Date: 03/19/19


BEGIN;

ALTER TABLE categories
  ADD CONSTRAINT categories_pkey PRIMARY KEY (ID);

ALTER TABLE images
  ADD CONSTRAINT images_pkey PRIMARY KEY (id);
-- ALTER TABLE images
--   ADD CONSTRAINT image_license_fkey
--   FOREIGN KEY (license) REFERENCES license(id);


ALTER TABLE annotations
  ADD CONSTRAINT annotations_pkey PRIMARY KEY (id);
ALTER TABLE annotations
  ADD CONSTRAINT annotations_image_id_fkey
  FOREIGN KEY (image_id) REFERENCES images(id);
ALTER TABLE annotations
  ADD CONSTRAINT annotations_category_id_fkey
  FOREIGN KEY (category_id) REFERENCES categories(id);

ALTER TABLE license
  ADD CONSTRAINT license_pkey PRIMARY KEY (id);

COMMIT;

ANALYZE categories;
ANALYZE images;
ANALYZE annotations;
ANALYZE license;

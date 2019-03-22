-- Create quick view tables for street-context annotations and images


BEGIN;

-- DROP TABLE IF EXISTS street_annotations;
-- DROP TABLE IF EXISTS street_images;


CREATE VIEW street_annotations AS
    (SELECT
      an.image_id as image_id,
      cat.name as category,
      an.id as annotation_id,
      an.bbox as bbox
    FROM
      annotations as an
    JOIN
        categories as cat
      ON
        an.category_id = cat.id
    AND
        cat.supercategory
      IN ('vehicle', 'outdoor')
    AND
        cat.name
      NOT IN ('airplane', 'train', 'boat'))
;


CREATE VIEW street_images AS (
    SELECT
        DISTINCT image_id,
        img.file_name as file_name,
        img.coco_url as coco_url,
        img.local_path as local_path
    FROM
        street_annotations
    JOIN
            images as img
        ON
            image_id = img.id
);


COMMIT;

FROM postgres

COPY initdb.sql /docker-entrypoint-initdb.d/initdb.sql
COPY at_startup.sh /docker-entrypoint-initdb.d/at_startup.sh

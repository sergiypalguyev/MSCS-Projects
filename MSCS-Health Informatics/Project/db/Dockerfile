FROM ubuntu:18.04

RUN apt-get update && apt-get install -y \
    software-properties-common postgresql-10 postgresql-client-10 postgresql-contrib-10

USER postgres

# Create a PostgreSQL role named ``docker`` with ``docker`` as the password and
# then create a database `docker` owned by the ``docker`` role.
# Note: here we use ``&&\`` to run commands one after the other - the ``\``
#       allows the RUN command to span multiple lines.

# SQL Script used to initialize the database.
#COPY ./init.sql /project/init.sql

COPY ./fhirbase fhirbase
COPY ./bundle.ndjson.gzip bundle.ndjson.gzip

RUN    /etc/init.d/postgresql start &&\
    psql --command "CREATE USER postgress WITH SUPERUSER PASSWORD 'postgres';"  &&\
    psql --command "CREATE DATABASE fhirbase;" &&\
    #&&\
    #psql -h localhost -p 5432 -U postgress --password postgres
    #&&\
    ./fhirbase --host localhost -p 5432 -d fhirbase -U postgress -W postgres --fhir=3.3.0 init &&\
    ./fhirbase --host localhost -p 5432 -d fhirbase -U postgress -W postgres --fhir=3.3.0 load -m insert ./bundle.ndjson.gzip



# Adjust PostgreSQL configuration so that remote connections to the
# database are possible.
RUN echo "host all  all    0.0.0.0/0  md5" >> /etc/postgresql/10/main/pg_hba.conf

# And add ``listen_addresses`` to ``/etc/postgresql/10/main/postgresql.conf``
RUN echo "listen_addresses='*'" >> /etc/postgresql/10/main/postgresql.conf

# Expose the PostgreSQL port
EXPOSE 5432

# Add VOLUMEs to allow backup of config, logs and databases
# VOLUME  ["/etc/postgresql", "/var/log/postgresql", "/var/lib/postgresql"]

# Set the default command to run when starting the container
CMD ["/usr/lib/postgresql/10/bin/postgres", "-D", "/var/lib/postgresql/10/main", "-c", "config_file=/etc/postgresql/10/main/postgresql.conf"]
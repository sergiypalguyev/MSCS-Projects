FROM tiangolo/uwsgi-nginx-flask:python3.6-alpine3.7
#RUN apk --update add bash nano vim
#ENV STATIC_URL /static
#ENV STATIC_PATH app/static
# install psycopg2 dependencies
RUN apk update
RUN apk add postgresql-dev gcc python3-dev musl-dev
COPY ./app /app
COPY ./requirements.txt requirements.txt
RUN pip3 install -r requirements.txt
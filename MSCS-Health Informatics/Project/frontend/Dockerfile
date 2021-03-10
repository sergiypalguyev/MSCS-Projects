# base image
FROM node:9.6.1

# set working directory
RUN mkdir /usr/src/app
WORKDIR /usr/src/app

# add `/usr/src/app/node_modules/.bin` to $PATH
ENV PATH /usr/src/app/node_modules/.bin:$PATH

# install and cache app dependencies
COPY public/ src/ package.json /usr/src/app/public/
COPY src/ /usr/src/app/src/
COPY package.json /usr/src/app/
RUN npm install --silent 
RUN npm install create-react-app -g
RUN npm install react-scripts@3.4.0 --silent
RUN npm install @material-ui/core --silent
RUN npm install @material-ui/icons --silent
RUN npm install axios --silent
RUN npm install http-server -D
RUN npm run build


# start app
CMD npm run start-sw



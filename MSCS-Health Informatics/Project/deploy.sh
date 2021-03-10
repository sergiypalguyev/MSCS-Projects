#!/bin/bash

# exit if one of the commands fails
set -e 

backend="gt-build.hdap.gatech.edu/ghivt30backend"
frontend="gt-build.hdap.gatech.edu/ghivt30frontend"
db="gt-build.hdap.gatech.edu/ghivt30db"

echo "\nBuilding backend image...\n"
docker build -t ${backend} ./backend/
echo "\nPushing backend image to gt docker repo...\n"
docker push ${backend}

echo "\nBuilding frontend image...\n"
docker build -t ${frontend} ./frontend/
echo "\nPushing frontend image to gt docker repo...\n"
docker push ${frontend}

echo "\nBuilding db image...\n"
docker build -t ${db} ./db/
echo "\nPushing db image to gt docker repo...\n"
docker push ${db}

echo "\nPushing git repo to gt remote..."
git push origin web

#!/bin/bash

# get templates needed for local deployment
git clone https://github.gatech.edu/es130/HDAP_helm_deploy_templates.git

# get the namespace set by the developer
userns=`awk '/namespace:/{print $2}' values.yaml`

# Delete old container and image
docker rm localapptest
docker rmi localbuild

# build templates, probably using docker image with volume mounted here
docker build -t localbuild -f ./HDAP_helm_deploy_templates/localtest/Dockerfile .
# run script to call kubectl on local machine
docker run --name localapptest -it localbuild
# copy the generated template over
docker cp localapptest:/apps/helm/k8s_install.yaml ./HDAP_helm_deploy_templates

# set the value of the NAMESPACE environment variable to the value you use for global.namespace in your values.yaml
export NAMESPACE=$userns

# deploy to local k8s
if [ "$(kubectl get namespace $NAMESPACE --ignore-not-found --output=custom-columns=NAME:.metadata.name --no-headers)" == "" ]; then kubectl create namespace $NAMESPACE; fi;
if [ "$(kubectl get deployment -n default ambassador --no-headers -o name | grep -c ambassador)" -eq 0 ]; then kubectl apply -f https://getambassador.io/yaml/ambassador/ambassador-rbac.yaml; else  echo "Ambassador already deployed"; fi;
if [ "$(kubectl get svc -n default ambassador --no-headers -o name | grep -c ambassador)" -eq 0 ]; then kubectl apply -n default -f ./HDAP_helm_deploy_templates/localtest/config/ambassador.yaml; else  echo "Ambassador SVC already Running"; fi;
kubectl delete --all deployment --namespace=$NAMESPACE
kubectl delete --all svc --namespace=$NAMESPACE
kubectl delete --all configmap --namespace=$NAMESPACE
kubectl delete --all pvc --namespace=$NAMESPACE
kubectl apply --namespace=$NAMESPACE -f ./HDAP_helm_deploy_templates/k8s_install.yaml

# Clean up
rm -rf ./HDAP_helm_deploy_templates/
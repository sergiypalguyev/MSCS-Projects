# HDAP K8S Deployment
This project contains artifacts to use to deploy applications to the HDAP Kubernetes development cluster. It is intended to
provide a base from which to build your applications.

## Required Software
This project allows you to test locally. In order to test locally the following must be installed on your local computer
- Docker Desktop
    - For Mac https://docs.docker.com/docker-for-mac/
        - Kubernetes must be enabled - https://docs.docker.com/docker-for-mac/kubernetes/
    - For Windows https://docs.docker.com/docker-for-windows/
        - Kubernetes must be enabled - https://docs.docker.com/docker-for-windows/kubernetes/
    - For Linux you will likely want to install
        - Docker CE - https://docs.docker.com/install/linux/docker-ce/ubuntu/
        - Minikube - https://kubernetes.io/docs/tasks/tools/install-minikube/

## Contents
- `values.yaml` - the file with values to configure your application.
- `.drone.yml` - a Drone.io configuration file used to define how your project should be built, packaged, and deployed. It
is used by the Drone.io CI tool to build your application.

## Before Using
- Log in to the drone.io server and activate your git repository.
- Contact an admin of the drone.io server to get your repository marked as a trusted.

## How to use
- Copy the contents of this project into your application.
- Set the `namespace` value for your application. It must be set in the `values.yaml` file.
  This value will be used to create a namespace in Kubernetes in which your application will run. For more info on Kubernetes
  namespaces go here: https://kubernetes.io/docs/concepts/overview/working-with-objects/namespaces/
  - Open the `values.yaml` file and set the value of `global.namespace`.
- Modify the `.drone.yml` to test/build/deploy your application. Add steps to it to build your application and deploy a
  docker image to the HDAP docker registry. Samples for each step are provided in the file.
- Modify the `values.yaml` file to have specific data for your application. The file has the following values:
    - **global.namespace** - the Kubernetes namespace that will contain your application.
    - **frontend.name** - the name of the application. `name` is a DNS-1035 label and must consist of lower case alphanumeric characters or '-', start with an alphabetic character, and end with an alphanumeric character (e.g. 'my-name',  or 'abc-123')
    - **frontend.port** - the port used by your application. This is the port used by the Kubernetes deployment for your application.
    - **frontend.replicaCount** - the number of replicas of your application that should be deployed.
    - **frontend.env** - a YAML dictionary of environment variable key:value pairs.
    - **frontend.volumes** - a YAML dictionary of key:value pairs that define the volumes needed for an application. The `key` is
    the name of the volume (it must contain no space or special characters). The `value` is the path to the directory in
    the container that needs to be mounted as a volume.
    - **frontend.proxy.incomingPath** - the path used by the proxy to route incoming traffic to your application
    - **frontend.proxy.containerPath** - the path used within your container to route traffic to your application. If this path is the
    empty string "" it means to use the same path as the `proxy.incomingPath`. Otherwise, this value will be used to
    replace the value identified by `proxy.incomingPath` on the incoming URL before forwarding the request to the container.
    - **frontend.image.repository** - the image to deploy. This field can be of the format `url.to.repository/my_image` to retrieve an
    image from a specific Docker registry or `my_image` to retrieve the image from DockerHub.
    - **frontend.image.tag** - the tag for the image to retrieve
    - **frontend.image.pullPolicy** - the image pull policy to use when deploying to Kubernetes. I can be `Always`, `IfNotPresent`, or `Never`

    - **backend.name** - the name of the application. `name` is a DNS-1035 label and must consist of lower case alphanumeric characters or '-', start with an alphabetic character, and end with an alphanumeric character (e.g. 'my-name',  or 'abc-123')
    - **backend.port** - the port used by your application. This is the port used by the Kubernetes deployment for your application.
    - **backend.replicaCount** - the number of replicas of your application that should be deployed.
    - **backend.env** - a YAML dictionary of environment variable key:value pairs.
    - **backend.volumes** - a YAML dictionary of key:value pairs that define the volumes needed for an application. The `key` is
    the name of the volume (it must contain no space or special characters). The `value` is the path to the directory in
    the container that needs to be mounted as a volume.
    - **backend.proxy.incomingPath** - the path used by the proxy to route incoming traffic to your application
    - **backend.proxy.containerPath** - the path used within your container to route traffic to your application. If this path is the
    empty string "" it means to use the same path as the `proxy.incomingPath`. Otherwise, this value will be used to
    replace the value identified by `proxy.incomingPath` on the incoming URL before forwarding the request to the container.
    - **backend.image.repository** - the image to deploy. This field can be of the format `url.to.repository/my_image` to retrieve an
    image from a specific Docker registry or `my_image` to retrieve the image from DockerHub.
    - **backend.image.tag** - the tag for the image to retrieve
    - **backend.image.pullPolicy** - the image pull policy to use when deploying to Kubernetes. I can be `Always`, `IfNotPresent`, or `Never`

    - **database.name** - the name of the application. `name` is a DNS-1035 label and must consist of lower case alphanumeric characters or '-', start with an alphabetic character, and end with an alphanumeric character (e.g. 'my-name',  or 'abc-123')
    - **database.port** - the port used by your application. This is the port used by the Kubernetes deployment for your application.
    - **database.replicaCount** - the number of replicas of your application that should be deployed.
    - **database.env** - a YAML dictionary of environment variable key:value pairs.
    - **database.volumes** - a YAML dictionary of key:value pairs that define the volumes needed for an application. The `key` is
    the name of the volume (it must contain no space or special characters). The `value` is the path to the directory in
    the container that needs to be mounted as a volume.
    - **database.image.repository** - the image to deploy. This field can be of the format `url.to.repository/my_image` to retrieve an
    image from a specific Docker registry or `my_image` to retrieve the image from DockerHub.
    - **database.image.tag** - the tag for the image to retrieve
    - **database.image.pullPolicy** - the image pull policy to use when deploying to Kubernetes. I can be `Always`, `IfNotPresent`, or `Never`

## Testing Locally
Use the `localtest.sh` script to test your application on your local machine. Before testing please make sure all items listed
in the required software section are installed.

From a terminal run the following command
> sh localtest.sh

This script will install your application into the local single node Kubernetes cluster. Once the application is installed
you will be able to access the deployed application using the IP addressassigned to the Ambassador service, port 32080,
and the incomingPath you assigned to your application in your values.yaml.

See step 4 here: https://www.getambassador.io/user-guide/getting-started/ for instructions on getting the IP address
for your local Kubernetes cluster.
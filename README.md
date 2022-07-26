# tensorflow_serving_client
This repo contains clients to classify images using tensorflow serving.
Code base is using ResNet50 as base model. It is easy to update for different models.

TensorFlow serving supports gRPC and REST APIs to commuinicate with edge devices.
To cliassify images there are three scripts tfs-grpc.py  tfs-rest.py and tfs-client.py
tfs-client.py contains both types of communication and cane be selected with command line flag.

Usage: 
python <tfs-rest.py/grpc.py> -s <tensorflow serving servers ip/name> -i <comma separated list of images>
python <tfs-client.py> -s <tensorflow serving servers ip/name> -i <comma separated list of images> -i <rest/grpc>

gRPC
```
$ python tfs-grpc.py -s serverip/name -i image1.jpg,image2,image3
```
REST
```
$ python tfs-rest.py -s serverip/name -i image1.jpg,image2,image3
```
gRPC + REST
```
$ python tfs-client.py -s serverip/name -a rest -i image1.jpg,image2,image3
$ python tfs-client.py -s serverip/name -a grpc -i image1.jpg,image2,image3
```

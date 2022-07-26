from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc
import tensorflow as tf
import pre_process
import grpc

import numpy as np
import pre_process
import requests
import argparse
import json
import sys
import ast
import os


image_desc = 'imagenet1000_clsidx_to_labels.txt'

#image size for ResNet50
image_dims = [224, 224, 3]

def parse_args():
    parser = argparse.ArgumentParser(description="Classify jpeg images using TensorFlow Serving.")
    parser.add_argument("-s", "--server",
                        type=str,
                        help="Server IP address or host name")
    parser.add_argument("-i", "--images",
                        type=str,
                        help="List of comma seperated images (JPEG, PNG) files")
    parser.add_argument("-a", "--api",
                        type=str,
                        help="API used rest or grpc")
    return parser.parse_args()

def usage():
    print("Classify_images -s 'server ip or name' -i 'image1.jpeg, image2.jpg' -a rest or grpc")
    sys.exit(1)

def load_images(images):
    images = images.split(',')
    #make sure all the files are available
    for i in images:
        if os.path.exists(i) == False:
            print(f'Error File not found: {i}')
            sys.exit(2)
    #TODO check for valid images
    return images

def load_labels():
    try:
        with open(image_desc, 'r') as f:
            contents = f.read()
            dictionary = ast.literal_eval(contents)
    except:
        print('Error: Not able to open labels')
        sys.exit(3)
    return dictionary


def classify_rest(images, dictionary, server):
    HOST=server.strip()
    PORT='8501'
    #XXX this dependence on server configuration
    model_desc = '/v1/models/resnet50:predict'
    #create link to access TensorFlow Serving using REST APIs 
    SERVER_URL = 'http://'+HOST+':'+PORT+model_desc

    imgs = list()
    # preprocess and combine all the images into single batch
    for i in images:
        imgs.append(pre_process.process(i, dims=image_dims))

    images_batch=np.stack(imgs, axis=0)

    try:
        predict_request = json.dumps({'instances': images_batch.tolist()})
        response = requests.post(SERVER_URL, data=predict_request)
        response.raise_for_status()
        response = response.json()['predictions']
        image_index=0
        for rsp in response:
            print(f'Top predictions for {images[image_index]}')
            r = np.array(rsp)
            r = np.argsort(r)
            r = np.flip(r)
            for _r in r[0:5]:
                #threshold for %age 0.05 
                #change if need to display only higher value
                if(100*rsp[_r] > 0.05):
                    print(f' {100*rsp[_r]:6.2f}% {dictionary[_r-1]}')
            print()
            image_index = image_index + 1
    except:
        print(f'Error: Not able to connect to TensorFlow Serving @ {HOST}')
        sys.exit(4)

def classify_grpc(images, dictionary, server):
    HOST=server.strip()
    PORT='8500'

    imgs = list()
    # preprocess and combine all the images into single batch
    for i in images:
        imgs.append(pre_process.process(i, dims=image_dims))

    try:
        images_batch=np.stack(imgs, axis=0)
        channel = grpc.insecure_channel(HOST+':'+PORT)
        stub    = prediction_service_pb2_grpc.PredictionServiceStub(channel)
        request = predict_pb2.PredictRequest()
        request.model_spec.name = 'resnet50'
        request.model_spec.signature_name = 'serving_default'
        request.inputs['input_tensor'].CopyFrom(tf.make_tensor_proto(images_batch, shape=images_batch.shape))
        result_future=stub.Predict.future(request, 500.0)
        result = result_future.result()
        dims   = result.outputs["softmax_tensor"].tensor_shape.dim
        shape  = tuple(d.size for d in dims)
        response = np.reshape(result.outputs["softmax_tensor"].float_val, shape)

        image_index=0
        for rsp in response:
            print(f'Top predictions for {images[image_index]}')
            r = np.array(rsp)
            r = np.argsort(r)
            r = np.flip(r)
            for _r in r[0:5]:
                if(100*rsp[_r] > 0.05):
                    print(f' {100*rsp[_r]:6.2f}% {dictionary[_r-1]}')
            print()
        image_index = image_index + 1
    except:
        print(f'Error: Not able to connect to TensorFlow Serving @ {HOST}')
        sys.exit(4)

def main():
    args = parse_args()
    if args.server == None:
        usage()
    
    if args.images == None:
        usage()
    
    if args.api == None:
        usage()
    #load image labels
    dictionary = load_labels()
    # check and load all the images in the list
    images = load_images(args.images)
    if args.api == 'rest':
        classify_rest(images, dictionary, args.server)
    else:
        classify_grpc(images, dictionary, args.server)

if __name__ == "__main__":
    main()

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2022, Ampere Computing LLC

import numpy as np
import pre_process
import requests
import argparse
import json
import sys
import ast
import os

image_desc = 'imagenet1000_clsidx_to_labels.txt'
#XXX this dependence on server configuration
model_desc = '/v1/models/resnet50:predict'

def parse_args():
    parser = argparse.ArgumentParser(description="Classify jpeg images using TensorFlow Serving.")
    parser.add_argument("-s", "--server",
                        type=str,
                        help="Server IP address or host name")
    parser.add_argument("-i", "--images",
                        type=str,
                        help="List of comma seperated images (JPEG, PNG) files")
    return parser.parse_args()

def usage():
    print("Classify_images -s 'server ip or name' -i 'image1.jpeg, image2.jpg'")
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

#image size for ResNet50
image_dims = [224, 224, 3]

def classify(images, dictionary,server):
    HOST=server.strip()
    PORT='8501'
    #create link to access TensorFlow Serving using REST APIs 
    SERVER_URL = 'http://'+HOST+':'+PORT+model_desc

    imgs = list()
    # preprocess and combine all the images into single batch
    for i in images:
        imgs.append(pre_process.process(i, dims=image_dims))

    images_batch=np.stack(imgs, axis=0)
    #HOST='10.76.110.70'
    #PORT='8501'
    #SERVER_URL = 'http://'+HOST+':'+PORT+'/v1/models/resnet50:predict'

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


def main():
    args = parse_args()
    if args.server == None:
        usage()
    
    if args.images == None:
        usage()
    #load image labels
    dictionary = load_labels()
    # check and load all the images in the list
    images = load_images(args.images)
    classify(images, dictionary, args.server)

if __name__ == "__main__":
    main()

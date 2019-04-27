# ios-tensorflow-serving
Tutorial on how to use the tensorflow serving in an ios app

### Links

Tensorflow repository: 

https://github.com/tensorflow/tensorflow

Tensorflow examples:

https://github.com/tensorflow/tensorflow/tree/master/tensorflow/examples/ios

Tensorflow serving resnet client example: 

https://github.com/tensorflow/serving/blob/master/tensorflow_serving/example/resnet_client.cc

GRPC Route guide example (Objective C):

https://github.com/grpc/grpc/tree/master/examples/objective-c/route_guide

Tensorflow serving repository: 

https://github.com/tensorflow/serving

Tensorlfow serving with docker: 

https://github.com/tensorflow/serving/blob/master/tensorflow_serving/g3doc/docker.md

Proto Class Prefix (objc_class_prefix):

https://developers.google.com/protocol-buffers/docs/proto3

### Commands/Code used: 

Model details:

    saved_model_cli show --dir ./inception_saved_model/1 --all

Run tensorflow serving in docker 

    docker run -t --rm -p 8500:8500 -v "$(pwd)/data/inception_saved_model":/models/inception -e MODEL_NAME=inception tensorflow/serving

For prefix option, add following to serving/tensorflow_serving/apis/classification.proto (To fix "redefinition of Class error")

    option objc_class_prefix = "PRSV";
    

    NSString* kHostAddress = @"localhost:8500";
    
    // Prepare GRPC 
    [GRPCCall useInsecureConnectionsForHost:kHostAddress];
    [GRPCCall setDefaultCompressMethod:GRPCCompressGzip forhost:kHostAddress];
    
    // Init PredictionService
    PredictionService* client = [[PredictionService alloc] initWithHost:kHostAddress];
    PredictRequest* request = [PredictRequest message];
    
    request.modelSpec.name = @"inception"; // MODEL_NAME from docker call
    request.modelSpec.signatureName = @""; // signature name from our convert script / saved_model_cli
    
    TensorShapeProto_Dim* tensorsShapeDim1 = [[TensorShapeProto_Dim alloc] init];
    tensorsShapeDim1.size = 1;
    TensorShapeProto_Dim* tensorsShapeDim224 = [[TensorShapeProto_Dim alloc] init];
    tensorsShapeDim224.size = 224;
    TensorShapeProto_Dim* tensorsShapeDim3 = [[TensorShapeProto_Dim alloc] init];
    tensorsShapeDim3.size = 3;
    
    // Prepare Tensor data for the serving
    TensorProto* realImagesTensorProto = [[TensorProto alloc] init];
    realImagesTensorProto.dtype = DataType_DtFloat;
    [realImagesTensorProto.tensorShape.dimArray addObject:tensorsShapeDim1];
    [realImagesTensorProto.tensorShape.dimArray addObject:tensorsShapeDim224];
    [realImagesTensorProto.tensorShape.dimArray addObject:tensorsShapeDim224];
    [realImagesTensorProto.tensorShape.dimArray addObject:tensorsShapeDim3];
    GPBFloatArray* array1 = [[GPBFloatArray alloc] initWithValues:out count:wanted_height * wanted_width * wanted_channels];
    [realImagesTensorProto setFloatValArray:array1]; // Fill the tensor with our data
    
    [request.inputs setObject:realImagesTensorProto forKey:@"input"];
    
    // Send the request
    [client predictWithRequest:request handler:^(PredictResponse *response, NSError *error) {
        
        // Process response 
        if (response) {
            NSString* result = @"Received response";
            
            NSMutableDictionary* outputsDic = response.outputs;
            
            TensorProto* outputTensor = [outputsDic objectForKey:@"output"];
            GPBFloatArray* confidenceScores = outputTensor.floatValArray;
            
            // Re-use old code 
            
            // Update UI in the main thread
            dispatch_async(dispatch_get_main_queue(), ^{
                urlContentTextView.text = result;
            });
            
        } else {
            // Error
            dispatch_async(dispatch_get_main_queue(), ^{
                urlContentTextView.text = [NSString stringWithFormat:@"Request failed: %@", error];
            });
        }
        
        return;
    }];
    

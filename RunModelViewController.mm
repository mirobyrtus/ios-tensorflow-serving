// Copyright 2015 Google Inc. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#import "RunModelViewController.h"

#include <fstream>
#include <pthread.h>
#include <unistd.h>
#include <queue>
#include <sstream>
#include <string>

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/public/session.h"

#import <GRPCClient/GRPCCall+ChannelArg.h>
#import <GRPCClient/GRPCCall+Tests.h>

#import <tensorflow_serving/apis/PredictionService.pbrpc.h>
#import <tensorflow_serving/apis/Predict.pbobjc.h>
#import <tensorflow_serving/apis/Classification.pbobjc.h>
#import <tensorflow_serving/apis/GetModelMetadata.pbobjc.h>
#import <tensorflow_serving/apis/Inference.pbobjc.h>
#import <tensorflow_serving/apis/Input.pbobjc.h>
#import <tensorflow_serving/apis/Regression.pbobjc.h>
#import <tensorflow/core/framework/Tensor.pbobjc.h>
#import <tensorflow/core/framework/TensorShape.pbobjc.h>
#import <tensorflow/core/framework/Types.pbobjc.h>
#import <tensorflow_serving/apis/Model.pbobjc.h>

#include "ios_image_load.h"

void RunRemoteInferenceOnImage(UITextView *);

namespace {
class IfstreamInputStream : public ::google::protobuf::io::CopyingInputStream {
 public:
  explicit IfstreamInputStream(const std::string& file_name)
      : ifs_(file_name.c_str(), std::ios::in | std::ios::binary) {}
  ~IfstreamInputStream() { ifs_.close(); }

  int Read(void* buffer, int size) {
    if (!ifs_) {
      return -1;
    }
    ifs_.read(static_cast<char*>(buffer), size);
    return (int)ifs_.gcount();
  }

 private:
  std::ifstream ifs_;
};
}  // namespace

@interface RunModelViewController ()
@end

@implementation RunModelViewController {
}

- (IBAction)getUrl:(id)sender {
    RunRemoteInferenceOnImage(self.urlContentTextView);
}

@end

// Returns the top N confidence values over threshold in the provided vector,
// sorted by confidence in descending order.
static void GetTopNProto(GPBFloatArray *prediction,
                    const int num_results, const float threshold,
                    std::vector<std::pair<float, int> >* top_results) {
    // Will contain top N results in ascending order.
    std::priority_queue<std::pair<float, int>,
    std::vector<std::pair<float, int> >,
    std::greater<std::pair<float, int> > > top_result_pq;
    
    const long count = [prediction count];
    for (int i = 0; i < count; ++i) {
        const float value = [prediction valueAtIndex:i];
        
        // Only add it if it beats the threshold and has a chance at being in
        // the top N.
        if (value < threshold) {
            continue;
        }
        
        top_result_pq.push(std::pair<float, int>(value, i));
        
        // If at capacity, kick the smallest value out.
        if (top_result_pq.size() > num_results) {
            top_result_pq.pop();
        }
    }
    
    // Copy to output vector and reverse into descending order.
    while (!top_result_pq.empty()) {
        top_results->push_back(top_result_pq.top());
        top_result_pq.pop();
    }
    std::reverse(top_results->begin(), top_results->end());
}

NSString* FilePathForResourceName(NSString* name, NSString* extension) {
  NSString* file_path = [[NSBundle mainBundle] pathForResource:name ofType:extension];
  if (file_path == NULL) {
    LOG(FATAL) << "Couldn't find '" << [name UTF8String] << "."
	       << [extension UTF8String] << "' in bundle.";
  }
  return file_path;
}

void RunRemoteInferenceOnImage(UITextView *urlContentTextView) {
    // Read the label list
    NSString* labels_path = FilePathForResourceName(@"imagenet_comp_graph_label_strings", @"txt");
    std::vector<std::string> label_strings;
    std::ifstream t;
    t.open([labels_path UTF8String]);
    std::string line;
    while(t){
        std::getline(t, line);
        label_strings.push_back(line);
    }
    t.close();
    
    // Read the Grace Hopper image.
    NSString* image_path = FilePathForResourceName(@"grace_hopper", @"jpg");
    int image_width;
    int image_height;
    int image_channels;
    std::vector<tensorflow::uint8> image_data = LoadImageFromFile(
                                                                  [image_path UTF8String], &image_width, &image_height, &image_channels);
    const int wanted_width = 224;
    const int wanted_height = 224;
    const int wanted_channels = 3;
    const float input_mean = 117.0f;
    const float input_std = 1.0f;
    assert(image_channels >= wanted_channels);
    tensorflow::Tensor image_tensor(
                                    tensorflow::DT_FLOAT,
                                    tensorflow::TensorShape({
        1, wanted_height, wanted_width, wanted_channels}));
    auto image_tensor_mapped = image_tensor.tensor<float, 4>();
    tensorflow::uint8* in = image_data.data();
    // tensorflow::uint8* in_end = (in + (image_height * image_width * image_channels));
    float* out = image_tensor_mapped.data();
    for (int y = 0; y < wanted_height; ++y) {
        const int in_y = (y * image_height) / wanted_height;
        tensorflow::uint8* in_row = in + (in_y * image_width * image_channels);
        float* out_row = out + (y * wanted_width * wanted_channels);
        for (int x = 0; x < wanted_width; ++x) {
            const int in_x = (x * image_width) / wanted_width;
            tensorflow::uint8* in_pixel = in_row + (in_x * image_channels);
            float* out_pixel = out_row + (x * wanted_channels);
            for (int c = 0; c < wanted_channels; ++c) {
                out_pixel[c] = (in_pixel[c] - input_mean) / input_std;
            }
        }
    }
    
    std::string input_layer = "input";
    std::string output_layer = "output";
    std::vector<tensorflow::Tensor> outputs;
    
    NSString* kHostAddress = @"localhost:8500"; // 8500 is default GRPC port
    
    // Prepare GRPC
    [GRPCCall useInsecureConnectionsForHost:kHostAddress];
    [GRPCCall setDefaultCompressMethod:GRPCCompressGzip forhost:kHostAddress];
    
    // Init PredictionService
    PredictionService* client = [[PredictionService alloc] initWithHost:kHostAddress];
    PredictRequest* request = [PredictRequest message];
    
    request.modelSpec.name = @"inception"; // MODEL_NAME from docker call
    request.modelSpec.signatureName = @""; // signature name from our convert script / saved_model_cli
    
    // Prepare Tensor data for the serving
    TensorShapeProto_Dim* tensorsShapeDim1 = [[TensorShapeProto_Dim alloc] init];
    tensorsShapeDim1.size = 1;
    TensorShapeProto_Dim* tensorsShapeDim224 = [[TensorShapeProto_Dim alloc] init];
    tensorsShapeDim224.size = 224;
    TensorShapeProto_Dim* tensorsShapeDim3 = [[TensorShapeProto_Dim alloc] init];
    tensorsShapeDim3.size = 3;
    
    TensorProto* realImagesTensorProto = [[TensorProto alloc] init];
    realImagesTensorProto.dtype = DataType_DtFloat;
    [realImagesTensorProto.tensorShape.dimArray addObject:tensorsShapeDim1];
    [realImagesTensorProto.tensorShape.dimArray addObject:tensorsShapeDim224];
    [realImagesTensorProto.tensorShape.dimArray addObject:tensorsShapeDim224];
    [realImagesTensorProto.tensorShape.dimArray addObject:tensorsShapeDim3];
    GPBFloatArray* array1 = [[GPBFloatArray alloc] initWithValues:out count:wanted_height * wanted_width * wanted_channels];
    [realImagesTensorProto setFloatValArray:array1]; // Fill the tensor with our data
    
    [request.inputs setObject:realImagesTensorProto forKey:@"input"];
    // NSLog(@"%@", realImagesTensorProto.debugDescription);
    
    // Send the request
    [client predictWithRequest:request handler:^(PredictResponse *response, NSError *error) {
        
        // Process response
        if (response) {
            NSString* result = @"Received response";
            
            NSMutableDictionary* outputsDic = response.outputs;
            
            TensorProto* outputTensor = [outputsDic objectForKey:@"output"];
            GPBFloatArray* confidenceScores = outputTensor.floatValArray;
            
            // Re-used code
            result = [NSString stringWithFormat: @"%@ - %lu, %s - %dx%d", result,
                      label_strings.size(), label_strings[0].c_str(), image_width, image_height];
            
            const int kNumResults = 5;
            const float kThreshold = 0.1f;
            std::vector<std::pair<float, int> > top_results;
            GetTopNProto(confidenceScores, kNumResults, kThreshold, &top_results);
            
            std::stringstream ss;
            ss.precision(3);
            for (const auto& result : top_results) {
                const float confidence = result.first;
                const int index = result.second;
                
                ss << index << " " << confidence << "  ";
                
                // Write out the result as a string
                if (index < label_strings.size()) {
                    // just for safety: theoretically, the output is under 1000 unless there
                    // is some numerical issues leading to a wrong prediction.
                    ss << label_strings[index];
                } else {
                    ss << "Prediction: " << index;
                }
                
                ss << "\n";
            }
            
            LOG(INFO) << "Predictions: " << ss.str();
            
            tensorflow::string predictions = ss.str();
            result = [NSString stringWithFormat: @"%@ - %s", result, predictions.c_str()];
            
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
}

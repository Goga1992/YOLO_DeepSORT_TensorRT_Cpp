#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <vector>
#include <chrono>
#include "NvInfer.h"
#include "cuda_runtime_api.h"
#include "utils.h"
#include "logging.h"
#include "yololayer.h"
#include "calibrator.h"
#include "datatype.h"
#include "yolov3_NAB.h"
#define USE_FP16  // set USE_INT8 or USE_FP16 or USE_FP32
#define DEVICE 0  // GPU id
#define NMS_THRESH 0.4
#define BBOX_CONF_THRESH 0.8
#define BATCH_SIZE 1

using namespace nvinfer1;

// stuff we know about the network and the input/output blobs

static const int INPUT_H = Yolo::INPUT_H;
static const int INPUT_W = Yolo::INPUT_W;
static const int CLASS_NUM = Yolo::CLASS_NUM;
static const int DETECTION_SIZE = sizeof(Yolo::Detection) / sizeof(float);
static const int OUTPUT_SIZE = Yolo::MAX_OUTPUT_BBOX_COUNT * DETECTION_SIZE + 1; 
const char* INPUT_BLOB_NAME = "data";
const char* OUTPUT_BLOB_NAME = "prob";
static Logger gLogger;

cv::Rect get_rect(cv::Mat& img, float bbox[4]) {
    int l, r, t, b;
    float r_w = INPUT_W / (img.cols * 1.0);
    float r_h = INPUT_H / (img.rows * 1.0);
    if (r_h > r_w) {
        l = bbox[0] - bbox[2]/2.f;
        r = bbox[0] + bbox[2]/2.f;
        t = bbox[1] - bbox[3]/2.f - (INPUT_H - r_w * img.rows) / 2;
        b = bbox[1] + bbox[3]/2.f - (INPUT_H - r_w * img.rows) / 2;
        l = l / r_w;
        r = r / r_w;
        t = t / r_w;
        b = b / r_w;
    } else {
        l = bbox[0] - bbox[2]/2.f - (INPUT_W - r_h * img.cols) / 2;
        r = bbox[0] + bbox[2]/2.f - (INPUT_W - r_h * img.cols) / 2;
        t = bbox[1] - bbox[3]/2.f;
        b = bbox[1] + bbox[3]/2.f;
        l = l / r_h;
        r = r / r_h;
        t = t / r_h;
        b = b / r_h;
    }
    return cv::Rect(l, t, r-l, b-t);
}

float iou(float lbox[4], float rbox[4]) {
    float interBox[] = {
        std::max(lbox[0] - lbox[2]/2.f , rbox[0] - rbox[2]/2.f), //left
        std::min(lbox[0] + lbox[2]/2.f , rbox[0] + rbox[2]/2.f), //right
        std::max(lbox[1] - lbox[3]/2.f , rbox[1] - rbox[3]/2.f), //top
        std::min(lbox[1] + lbox[3]/2.f , rbox[1] + rbox[3]/2.f), //bottom
    };

    if(interBox[2] > interBox[3] || interBox[0] > interBox[1])
        return 0.0f;

    float interBoxS =(interBox[1]-interBox[0])*(interBox[3]-interBox[2]);
    return interBoxS/(lbox[2]*lbox[3] + rbox[2]*rbox[3] -interBoxS);
}

bool cmp(const Yolo::Detection& a, const Yolo::Detection& b) {
    return a.det_confidence > b.det_confidence;
}

// void nms(std::vector<Yolo::Detection>& res, float *output, float nms_thresh = NMS_THRESH) {
//     std::map<float, std::vector<Yolo::Detection>> m;
//     for (int i = 0; i < output[0] && i < 1000; i++) {
//         if (output[1 + 7 * i + 4] <= BBOX_CONF_THRESH) continue;
//         Yolo::Detection det;
//         memcpy(&det, &output[1 + 7 * i], 7 * sizeof(float));
//         if (m.count(det.class_id) == 0) m.emplace(det.class_id, std::vector<Yolo::Detection>());
//         m[det.class_id].push_back(det);
//     }
//     for (auto it = m.begin(); it != m.end(); it++) {
//         //std::cout << it->second[0].class_id << " --- " << std::endl;
//         auto& dets = it->second;
//         std::sort(dets.begin(), dets.end(), cmp);
//         for (size_t m = 0; m < dets.size(); ++m) {
//             auto& item = dets[m];
//             res.push_back(item);
//             for (size_t n = m + 1; n < dets.size(); ++n) {
//                 if (iou(item.bbox, dets[n].bbox) > nms_thresh) {
//                     dets.erase(dets.begin()+n);
//                     --n;
//                 }
//             }
//         }
//     }
// }

void nms(std::vector<Yolo::Detection>& res, float *output, float conf_thresh, float nms_thresh = 0.5) {
    int det_size = sizeof(Yolo::Detection) / sizeof(float);
    std::map<float, std::vector<Yolo::Detection>> m;
    for (int i = 0; i < output[0] && i < Yolo::MAX_OUTPUT_BBOX_COUNT; i++) {
        if (output[1 + det_size * i + 4] <= conf_thresh) continue;
        Yolo::Detection det;
        memcpy(&det, &output[1 + det_size * i], det_size * sizeof(float));
        if (m.count(det.class_id) == 0) m.emplace(det.class_id, std::vector<Yolo::Detection>());
        m[det.class_id].push_back(det);
    }
    for (auto it = m.begin(); it != m.end(); it++) {
        //std::cout << it->second[0].class_id << " --- " << std::endl;
        auto& dets = it->second;
        std::sort(dets.begin(), dets.end(), cmp);
        for (size_t m = 0; m < dets.size(); ++m) {
            auto& item = dets[m];
            res.push_back(item);
            for (size_t n = m + 1; n < dets.size(); ++n) {
                if (iou(item.bbox, dets[n].bbox) > nms_thresh) {
                    dets.erase(dets.begin() + n);
                    --n;
                }
            }
        }
    }
}

// TensorRT weight files have a simple space delimited format:
// [type] [size] <data x size in hex>
std::map<std::string, Weights> loadWeights(const std::string file) {
    std::cout << "Loading weights: " << file << std::endl;
    std::map<std::string, Weights> weightMap;

    // Open weights file
    std::ifstream input(file);
    assert(input.is_open() && "Unable to load weight file.");

    // Read number of weight blobs
    int32_t count;
    input >> count;
    assert(count > 0 && "Invalid weight map file.");

    while (count--)
    {
        Weights wt{DataType::kFLOAT, nullptr, 0};
        uint32_t size;

        // Read name and type of blob
        std::string name;
        input >> name >> std::dec >> size;
        wt.type = DataType::kFLOAT;

        // Load blob
        uint32_t* val = reinterpret_cast<uint32_t*>(malloc(sizeof(val) * size));
        for (uint32_t x = 0, y = size; x < y; ++x)
        {
            input >> std::hex >> val[x];
        }
        wt.values = val;
        
        wt.count = size;
        weightMap[name] = wt;
    }

    return weightMap;
}

IScaleLayer* addBatchNorm2d(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, std::string lname, float eps) {
    float *gamma = (float*)weightMap[lname + ".weight"].values;
    float *beta = (float*)weightMap[lname + ".bias"].values;
    float *mean = (float*)weightMap[lname + ".running_mean"].values;
    float *var = (float*)weightMap[lname + ".running_var"].values;
    int len = weightMap[lname + ".running_var"].count;

    float *scval = reinterpret_cast<float*>(malloc(sizeof(float) * len));
    for (int i = 0; i < len; i++) {
        scval[i] = gamma[i] / sqrt(var[i] + eps);
    }
    Weights scale{DataType::kFLOAT, scval, len};
    
    float *shval = reinterpret_cast<float*>(malloc(sizeof(float) * len));
    for (int i = 0; i < len; i++) {
        shval[i] = beta[i] - mean[i] * gamma[i] / sqrt(var[i] + eps);
    }
    Weights shift{DataType::kFLOAT, shval, len};

    float *pval = reinterpret_cast<float*>(malloc(sizeof(float) * len));
    for (int i = 0; i < len; i++) {
        pval[i] = 1.0;
    }
    Weights power{DataType::kFLOAT, pval, len};

    weightMap[lname + ".scale"] = scale;
    weightMap[lname + ".shift"] = shift;
    weightMap[lname + ".power"] = power;
    IScaleLayer* scale_1 = network->addScale(input, ScaleMode::kCHANNEL, shift, scale, power);
    assert(scale_1);
    return scale_1;
}

ILayer* convBnLeaky(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input,  int outch, int ksize, int s, int p, int linx) {
    Weights emptywts{DataType::kFLOAT, nullptr, 0};
    IConvolutionLayer* conv1 = network->addConvolutionNd(input, outch, DimsHW{ksize, ksize}, weightMap["module_list." + std::to_string(linx) + ".Conv2d.weight"], emptywts);
    assert(conv1);
    conv1->setStrideNd(DimsHW{s, s});
    conv1->setPaddingNd(DimsHW{p, p});

    IScaleLayer* bn1 = addBatchNorm2d(network, weightMap, *conv1->getOutput(0), "module_list." + std::to_string(linx) + ".BatchNorm2d", 1e-5);

    auto lr = network->addActivation(*bn1->getOutput(0), ActivationType::kLEAKY_RELU);
    lr->setAlpha(0.1);

    return lr;
}

// Creat the engine using only the API and not any parser.
ICudaEngine* createEngine(unsigned int maxBatchSize, IBuilder* builder, IBuilderConfig* config, DataType dt) {
    INetworkDefinition* network = builder->createNetworkV2(0U);

    // Create input tensor of shape {3, INPUT_H, INPUT_W} with name INPUT_BLOB_NAME
    ITensor* data = network->addInput(INPUT_BLOB_NAME, dt, Dims3{3, INPUT_H, INPUT_W});
    assert(data);

    std::map<std::string, Weights> weightMap = loadWeights("../../../nabTensorRTxResource/yolov3/weights/nab_yolov3_320.wts");
    Weights emptywts{DataType::kFLOAT, nullptr, 0};

    // Yeah I am stupid, I just want to expand the complete arch of darknet..
    auto lr0 = convBnLeaky(network, weightMap, *data, 32, 3, 1, 1, 0);
    auto lr1 = convBnLeaky(network, weightMap, *lr0->getOutput(0), 64, 3, 2, 1, 1);
    auto lr2 = convBnLeaky(network, weightMap, *lr1->getOutput(0), 32, 1, 1, 0, 2);
    auto lr3 = convBnLeaky(network, weightMap, *lr2->getOutput(0), 64, 3, 1, 1, 3);
    auto ew4 = network->addElementWise(*lr3->getOutput(0), *lr1->getOutput(0), ElementWiseOperation::kSUM);
    auto lr5 = convBnLeaky(network, weightMap, *ew4->getOutput(0), 128, 3, 2, 1, 5);
    auto lr6 = convBnLeaky(network, weightMap, *lr5->getOutput(0), 64, 1, 1, 0, 6);
    auto lr7 = convBnLeaky(network, weightMap, *lr6->getOutput(0), 128, 3, 1, 1, 7);
    auto ew8 = network->addElementWise(*lr7->getOutput(0), *lr5->getOutput(0), ElementWiseOperation::kSUM);
    auto lr9 = convBnLeaky(network, weightMap, *ew8->getOutput(0), 64, 1, 1, 0, 9);
    auto lr10 = convBnLeaky(network, weightMap, *lr9->getOutput(0), 128, 3, 1, 1, 10);
    auto ew11 = network->addElementWise(*lr10->getOutput(0), *ew8->getOutput(0), ElementWiseOperation::kSUM);
    auto lr12 = convBnLeaky(network, weightMap, *ew11->getOutput(0), 256, 3, 2, 1, 12);
    auto lr13 = convBnLeaky(network, weightMap, *lr12->getOutput(0), 128, 1, 1, 0, 13);
    auto lr14 = convBnLeaky(network, weightMap, *lr13->getOutput(0), 256, 3, 1, 1, 14);
    auto ew15 = network->addElementWise(*lr14->getOutput(0), *lr12->getOutput(0), ElementWiseOperation::kSUM);
    auto lr16 = convBnLeaky(network, weightMap, *ew15->getOutput(0), 128, 1, 1, 0, 16);
    auto lr17 = convBnLeaky(network, weightMap, *lr16->getOutput(0), 256, 3, 1, 1, 17);
    auto ew18 = network->addElementWise(*lr17->getOutput(0), *ew15->getOutput(0), ElementWiseOperation::kSUM);
    auto lr19 = convBnLeaky(network, weightMap, *ew18->getOutput(0), 128, 1, 1, 0, 19);
    auto lr20 = convBnLeaky(network, weightMap, *lr19->getOutput(0), 256, 3, 1, 1, 20);
    auto ew21 = network->addElementWise(*lr20->getOutput(0), *ew18->getOutput(0), ElementWiseOperation::kSUM);
    auto lr22 = convBnLeaky(network, weightMap, *ew21->getOutput(0), 128, 1, 1, 0, 22);
    auto lr23 = convBnLeaky(network, weightMap, *lr22->getOutput(0), 256, 3, 1, 1, 23);
    auto ew24 = network->addElementWise(*lr23->getOutput(0), *ew21->getOutput(0), ElementWiseOperation::kSUM);
    auto lr25 = convBnLeaky(network, weightMap, *ew24->getOutput(0), 128, 1, 1, 0, 25);
    auto lr26 = convBnLeaky(network, weightMap, *lr25->getOutput(0), 256, 3, 1, 1, 26);
    auto ew27 = network->addElementWise(*lr26->getOutput(0), *ew24->getOutput(0), ElementWiseOperation::kSUM);
    auto lr28 = convBnLeaky(network, weightMap, *ew27->getOutput(0), 128, 1, 1, 0, 28);
    auto lr29 = convBnLeaky(network, weightMap, *lr28->getOutput(0), 256, 3, 1, 1, 29);
    auto ew30 = network->addElementWise(*lr29->getOutput(0), *ew27->getOutput(0), ElementWiseOperation::kSUM);
    auto lr31 = convBnLeaky(network, weightMap, *ew30->getOutput(0), 128, 1, 1, 0, 31);
    auto lr32 = convBnLeaky(network, weightMap, *lr31->getOutput(0), 256, 3, 1, 1, 32);
    auto ew33 = network->addElementWise(*lr32->getOutput(0), *ew30->getOutput(0), ElementWiseOperation::kSUM);
    auto lr34 = convBnLeaky(network, weightMap, *ew33->getOutput(0), 128, 1, 1, 0, 34);
    auto lr35 = convBnLeaky(network, weightMap, *lr34->getOutput(0), 256, 3, 1, 1, 35);
    auto ew36 = network->addElementWise(*lr35->getOutput(0), *ew33->getOutput(0), ElementWiseOperation::kSUM);
    auto lr37 = convBnLeaky(network, weightMap, *ew36->getOutput(0), 512, 3, 2, 1, 37);
    auto lr38 = convBnLeaky(network, weightMap, *lr37->getOutput(0), 256, 1, 1, 0, 38);
    auto lr39 = convBnLeaky(network, weightMap, *lr38->getOutput(0), 512, 3, 1, 1, 39);
    auto ew40 = network->addElementWise(*lr39->getOutput(0), *lr37->getOutput(0), ElementWiseOperation::kSUM);
    auto lr41 = convBnLeaky(network, weightMap, *ew40->getOutput(0), 256, 1, 1, 0, 41);
    auto lr42 = convBnLeaky(network, weightMap, *lr41->getOutput(0), 512, 3, 1, 1, 42);
    auto ew43 = network->addElementWise(*lr42->getOutput(0), *ew40->getOutput(0), ElementWiseOperation::kSUM);
    auto lr44 = convBnLeaky(network, weightMap, *ew43->getOutput(0), 256, 1, 1, 0, 44);
    auto lr45 = convBnLeaky(network, weightMap, *lr44->getOutput(0), 512, 3, 1, 1, 45);
    auto ew46 = network->addElementWise(*lr45->getOutput(0), *ew43->getOutput(0), ElementWiseOperation::kSUM);
    auto lr47 = convBnLeaky(network, weightMap, *ew46->getOutput(0), 256, 1, 1, 0, 47);
    auto lr48 = convBnLeaky(network, weightMap, *lr47->getOutput(0), 512, 3, 1, 1, 48);
    auto ew49 = network->addElementWise(*lr48->getOutput(0), *ew46->getOutput(0), ElementWiseOperation::kSUM);
    auto lr50 = convBnLeaky(network, weightMap, *ew49->getOutput(0), 256, 1, 1, 0, 50);
    auto lr51 = convBnLeaky(network, weightMap, *lr50->getOutput(0), 512, 3, 1, 1, 51);
    auto ew52 = network->addElementWise(*lr51->getOutput(0), *ew49->getOutput(0), ElementWiseOperation::kSUM);
    auto lr53 = convBnLeaky(network, weightMap, *ew52->getOutput(0), 256, 1, 1, 0, 53);
    auto lr54 = convBnLeaky(network, weightMap, *lr53->getOutput(0), 512, 3, 1, 1, 54);
    auto ew55 = network->addElementWise(*lr54->getOutput(0), *ew52->getOutput(0), ElementWiseOperation::kSUM);
    auto lr56 = convBnLeaky(network, weightMap, *ew55->getOutput(0), 256, 1, 1, 0, 56);
    auto lr57 = convBnLeaky(network, weightMap, *lr56->getOutput(0), 512, 3, 1, 1, 57);
    auto ew58 = network->addElementWise(*lr57->getOutput(0), *ew55->getOutput(0), ElementWiseOperation::kSUM);
    auto lr59 = convBnLeaky(network, weightMap, *ew58->getOutput(0), 256, 1, 1, 0, 59);
    auto lr60 = convBnLeaky(network, weightMap, *lr59->getOutput(0), 512, 3, 1, 1, 60);
    auto ew61 = network->addElementWise(*lr60->getOutput(0), *ew58->getOutput(0), ElementWiseOperation::kSUM);
    auto lr62 = convBnLeaky(network, weightMap, *ew61->getOutput(0), 1024, 3, 2, 1, 62);
    auto lr63 = convBnLeaky(network, weightMap, *lr62->getOutput(0), 512, 1, 1, 0, 63);
    auto lr64 = convBnLeaky(network, weightMap, *lr63->getOutput(0), 1024, 3, 1, 1, 64);
    auto ew65 = network->addElementWise(*lr64->getOutput(0), *lr62->getOutput(0), ElementWiseOperation::kSUM);
    auto lr66 = convBnLeaky(network, weightMap, *ew65->getOutput(0), 512, 1, 1, 0, 66);
    auto lr67 = convBnLeaky(network, weightMap, *lr66->getOutput(0), 1024, 3, 1, 1, 67);
    auto ew68 = network->addElementWise(*lr67->getOutput(0), *ew65->getOutput(0), ElementWiseOperation::kSUM);
    auto lr69 = convBnLeaky(network, weightMap, *ew68->getOutput(0), 512, 1, 1, 0, 69);
    auto lr70 = convBnLeaky(network, weightMap, *lr69->getOutput(0), 1024, 3, 1, 1, 70);
    auto ew71 = network->addElementWise(*lr70->getOutput(0), *ew68->getOutput(0), ElementWiseOperation::kSUM);
    auto lr72 = convBnLeaky(network, weightMap, *ew71->getOutput(0), 512, 1, 1, 0, 72);
    auto lr73 = convBnLeaky(network, weightMap, *lr72->getOutput(0), 1024, 3, 1, 1, 73);
    auto ew74 = network->addElementWise(*lr73->getOutput(0), *ew71->getOutput(0), ElementWiseOperation::kSUM);
    auto lr75 = convBnLeaky(network, weightMap, *ew74->getOutput(0), 512, 1, 1, 0, 75);
    auto lr76 = convBnLeaky(network, weightMap, *lr75->getOutput(0), 1024, 3, 1, 1, 76);
    auto lr77 = convBnLeaky(network, weightMap, *lr76->getOutput(0), 512, 1, 1, 0, 77);
    auto lr78 = convBnLeaky(network, weightMap, *lr77->getOutput(0), 1024, 3, 1, 1, 78);
    auto lr79 = convBnLeaky(network, weightMap, *lr78->getOutput(0), 512, 1, 1, 0, 79);
    auto lr80 = convBnLeaky(network, weightMap, *lr79->getOutput(0), 1024, 3, 1, 1, 80);
    IConvolutionLayer* conv81 = network->addConvolutionNd(*lr80->getOutput(0), 3 * (Yolo::CLASS_NUM + 5), DimsHW{1, 1}, weightMap["module_list.81.Conv2d.weight"], weightMap["module_list.81.Conv2d.bias"]);
    assert(conv81);
    // 82 is yolo
    auto l83 = lr79;
    auto lr84 = convBnLeaky(network, weightMap, *l83->getOutput(0), 256, 1, 1, 0, 84);

    float *deval = reinterpret_cast<float*>(malloc(sizeof(float) * 256 * 2 * 2));
    for (int i = 0; i < 256 * 2 * 2; i++) {
        deval[i] = 1.0;
    }
    Weights deconvwts85{DataType::kFLOAT, deval, 256 * 2 * 2};
    IDeconvolutionLayer* deconv85 = network->addDeconvolutionNd(*lr84->getOutput(0), 256, DimsHW{2, 2}, deconvwts85, emptywts);
    assert(deconv85);
    deconv85->setStrideNd(DimsHW{2, 2});
    deconv85->setNbGroups(256);
    weightMap["deconv85"] = deconvwts85;

    ITensor* inputTensors[] = {deconv85->getOutput(0), ew61->getOutput(0)};
    auto cat86 = network->addConcatenation(inputTensors, 2);
    auto lr87 = convBnLeaky(network, weightMap, *cat86->getOutput(0), 256, 1, 1, 0, 87);
    auto lr88 = convBnLeaky(network, weightMap, *lr87->getOutput(0), 512, 3, 1, 1, 88);
    auto lr89 = convBnLeaky(network, weightMap, *lr88->getOutput(0), 256, 1, 1, 0, 89);
    auto lr90 = convBnLeaky(network, weightMap, *lr89->getOutput(0), 512, 3, 1, 1, 90);
    auto lr91 = convBnLeaky(network, weightMap, *lr90->getOutput(0), 256, 1, 1, 0, 91);
    auto lr92 = convBnLeaky(network, weightMap, *lr91->getOutput(0), 512, 3, 1, 1, 92);
    IConvolutionLayer* conv93 = network->addConvolutionNd(*lr92->getOutput(0), 3 * (Yolo::CLASS_NUM + 5), DimsHW{1, 1}, weightMap["module_list.93.Conv2d.weight"], weightMap["module_list.93.Conv2d.bias"]);
    assert(conv93);
    // 94 is yolo
    auto l95 = lr91;
    auto lr96 = convBnLeaky(network, weightMap, *l95->getOutput(0), 128, 1, 1, 0, 96);
    Weights deconvwts97{DataType::kFLOAT, deval, 128 * 2 * 2};
    IDeconvolutionLayer* deconv97 = network->addDeconvolutionNd(*lr96->getOutput(0), 128, DimsHW{2, 2}, deconvwts97, emptywts);
    assert(deconv97);
    deconv97->setStrideNd(DimsHW{2, 2});
    deconv97->setNbGroups(128);
    ITensor* inputTensors1[] = {deconv97->getOutput(0), ew36->getOutput(0)};
    auto cat98 = network->addConcatenation(inputTensors1, 2);
    auto lr99 = convBnLeaky(network, weightMap, *cat98->getOutput(0), 128, 1, 1, 0, 99);
    auto lr100 = convBnLeaky(network, weightMap, *lr99->getOutput(0), 256, 3, 1, 1, 100);
    auto lr101 = convBnLeaky(network, weightMap, *lr100->getOutput(0), 128, 1, 1, 0, 101);
    auto lr102 = convBnLeaky(network, weightMap, *lr101->getOutput(0), 256, 3, 1, 1, 102);
    auto lr103 = convBnLeaky(network, weightMap, *lr102->getOutput(0), 128, 1, 1, 0, 103);
    auto lr104 = convBnLeaky(network, weightMap, *lr103->getOutput(0), 256, 3, 1, 1, 104);
    IConvolutionLayer* conv105 = network->addConvolutionNd(*lr104->getOutput(0), 3 * (Yolo::CLASS_NUM + 5), DimsHW{1, 1}, weightMap["module_list.105.Conv2d.weight"], weightMap["module_list.105.Conv2d.bias"]);
    assert(conv105);

    auto creator = getPluginRegistry()->getPluginCreator("YoloLayer_TRT", "1");
    const PluginFieldCollection* pluginData = creator->getFieldNames();
    IPluginV2 *pluginObj = creator->createPlugin("yololayer", pluginData);
    ITensor* inputTensors_yolo[] = {conv81->getOutput(0), conv93->getOutput(0), conv105->getOutput(0)};
    auto yolo = network->addPluginV2(inputTensors_yolo, 3, *pluginObj);

    yolo->getOutput(0)->setName(OUTPUT_BLOB_NAME);
    network->markOutput(*yolo->getOutput(0));

    // Build engine
    builder->setMaxBatchSize(maxBatchSize);
    config->setMaxWorkspaceSize(16 * (1 << 20));  // 16MB
#if defined(USE_FP16)
    config->setFlag(BuilderFlag::kFP16);
#elif defined(USE_INT8)
    std::cout << "Your platform support int8: " << (builder->platformHasFastInt8() ? "true" : "false") << std::endl;
    assert(builder->platformHasFastInt8());
    config->setFlag(BuilderFlag::kINT8);
    Int8EntropyCalibrator2 *calibrator = new Int8EntropyCalibrator2(1, INPUT_W, INPUT_H, "./coco_calib/", "int8calib.table", INPUT_BLOB_NAME);
    config->setInt8Calibrator(calibrator);
#endif

    std::cout << "Building engine, please wait for a while..." << std::endl;
    ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);
    std::cout << "Build engine successfully!" << std::endl;

    // Don't need the network any more
    network->destroy();

    // Release host memory
    for (auto& mem : weightMap)
    {
        free((void*) (mem.second.values));
    }

    return engine;
}

void APIToModel(unsigned int maxBatchSize, IHostMemory** modelStream) {
    // Create builder
    IBuilder* builder = createInferBuilder(gLogger);
    IBuilderConfig* config = builder->createBuilderConfig();

    // Create model to populate the network, then set the outputs and create an engine
    ICudaEngine* engine = createEngine(maxBatchSize, builder, config, DataType::kFLOAT);
    assert(engine != nullptr);

    // Serialize the engine
    (*modelStream) = engine->serialize();

    // Close everything down
    engine->destroy();
    builder->destroy();
}

static void doInference(IExecutionContext& context, cudaStream_t& stream, void **buffers, float* input, float* output, int batchSize) {
    // DMA input batch data to device, infer on the batch asynchronously, and DMA output back to host
    CUDA_CHECK(cudaMemcpyAsync(buffers[0], input, batchSize * 3 * INPUT_H * INPUT_W * sizeof(float), cudaMemcpyHostToDevice, stream));
    //cudaMemcpyAsync(buffers[0], input, batchSize * 3 * INPUT_H * INPUT_W * sizeof(float), cudaMemcpyHostToDevice, stream);
    context.enqueue(batchSize, buffers, stream, nullptr);
    CUDA_CHECK(cudaMemcpyAsync(output, buffers[1], batchSize * OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost, stream));
    //cudaMemcpyAsync(output, buffers[1], batchSize * OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
}

typedef struct 
{
    float *data;
    float *prob;
    IRuntime *runtime;
    ICudaEngine *engine;
    IExecutionContext *exe_context;
    void* buffers[2];
    cudaStream_t cuda_stream;
    int inputIndex;
    int outputIndex;
    char result_json_str[16384];
 
}Yolov3TRTContext;

typedef struct{
	int class_id;
	int x1;
	int y1;
	int x2; 
	int y2;
	float conf;
	
}DeepsortContext;


void * yolov3_trt_create(const char * engine_name){
    size_t size{0};
    char *trtModelStream{nullptr};
    Yolov3TRTContext * trt_ctx = NULL;
    
    trt_ctx =  new Yolov3TRTContext();
    std::ifstream file(engine_name, std::ios::binary);
    if (file.good()) {
        file.seekg(0, file.end);
        size = file.tellg();
        file.seekg(0, file.beg);
        trtModelStream = new char[size];
        assert(trtModelStream);
        file.read(trtModelStream, size);
        file.close();
    }else{
        return NULL;
    }
    trt_ctx->data =  new float[BATCH_SIZE * 3 * INPUT_H * INPUT_W];  //===================
    trt_ctx->prob = new float[BATCH_SIZE * OUTPUT_SIZE];    //===================
    trt_ctx->runtime =  createInferRuntime(gLogger);    //===================
    assert(trt_ctx->runtime != nullptr);
    trt_ctx->engine = trt_ctx->runtime->deserializeCudaEngine(trtModelStream, size);    //===================
    assert(trt_ctx->engine != nullptr);
    trt_ctx->exe_context = trt_ctx->engine->createExecutionContext();   //===================
    assert(trt_ctx->exe_context != nullptr);
    delete[] trtModelStream;
    assert(trt_ctx->engine->getNbBindings() == 2);
    trt_ctx->inputIndex = trt_ctx->engine->getBindingIndex(INPUT_BLOB_NAME);
    trt_ctx->outputIndex = trt_ctx->engine->getBindingIndex(OUTPUT_BLOB_NAME);
    assert(trt_ctx->inputIndex == 0);
    assert(trt_ctx->outputIndex == 1);
    CUDA_CHECK(cudaMalloc(&trt_ctx->buffers[trt_ctx->inputIndex], BATCH_SIZE * 3 * INPUT_H * INPUT_W * sizeof(float)));

    CUDA_CHECK(cudaMalloc(&trt_ctx->buffers[trt_ctx->outputIndex], BATCH_SIZE * OUTPUT_SIZE * sizeof(float)));

    CUDA_CHECK(cudaStreamCreate(&trt_ctx->cuda_stream));
    return (void *)trt_ctx;
    

}

int yolov3_trt_detect(void *h, cv::Mat &img, float threshold,std::vector<DetectBox>& det){
    Yolov3TRTContext *trt_ctx;
    int i;
    int delay_preprocess;
    int delay_infer;
    trt_ctx = (Yolov3TRTContext *)h;
    trt_ctx->result_json_str[0] = 0;
 	// whether det is empty , if not, empty det
 	if (!det.empty()) det.clear();
    if (img.empty()) return 0;
 
    auto start0 = std::chrono::system_clock::now();
 
    //printf("yolov5_trt_detect start preprocess img \n");
    cv::Mat pr_img = preprocess_img(img, INPUT_W, INPUT_H);
 	//std::cout<<"after preprocess_img pr_img size:"<<pr_img.cols<<" "<<pr_img.rows<<std::endl;
 	//std::cout<<"after preprocess_img frame size:"<<img.cols<<" "<<img.rows<<std::endl;
 
 
    //printf("yolov5_trt_detect start convert img to float\n");
    // letterbox BGR to RGB
    i = 0;
    for (int row = 0; row < INPUT_H; ++row) {
        uchar* uc_pixel = pr_img.data + row * pr_img.step;
        for (int col = 0; col < INPUT_W; ++col) {
            trt_ctx->data[i] = (float)uc_pixel[2] / 255.0;
            trt_ctx->data[i + INPUT_H * INPUT_W] = (float)uc_pixel[1] / 255.0;
            trt_ctx->data[i + 2 * INPUT_H * INPUT_W] = (float)uc_pixel[0] / 255.0;
            uc_pixel += 3;
            ++i;
        }
    }
    auto end0 = std::chrono::system_clock::now();
 
    delay_preprocess =  std::chrono::duration_cast<std::chrono::milliseconds>(end0 - start0).count();

    auto start = std::chrono::system_clock::now();
    doInference(*trt_ctx->exe_context, trt_ctx->cuda_stream, trt_ctx->buffers, trt_ctx->data, trt_ctx->prob, BATCH_SIZE);
 
    auto end = std::chrono::system_clock::now();
    delay_infer = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
 
    float fps = 1 / (delay_infer*0.001);
    std::cout <<"delay_proress:" << delay_preprocess << "ms, " << "delay_infer:" << delay_infer << "ms" << " fps:" << fps << std::endl;
 
 
    int fcount = 1;
    int str_len;
    std::vector<std::vector<Yolo::Detection>> batch_res(1);
    auto& res = batch_res[0];
    nms(res, &trt_ctx->prob[0], threshold, NMS_THRESH);

 
    i = 0;
    for(i = 0 ; i < res.size(); i++){
        int x1, y1, x2, y2;
        int class_id;
        float conf;
        cv::Rect r = get_rect(img, res[i].bbox);
 
        DetectBox dd(r.x,r.y,r.x + r.width,r.y + r.height,(float)res[i].det_confidence,(int)res[i].class_id);
        det.push_back(dd);
    }
    return 1;
}
 
 
void yolov3_trt_destroy(void *h)
{
    Yolov3TRTContext *trt_ctx;
 
    trt_ctx = (Yolov3TRTContext *)h;
 
    // Release stream and buffers
    cudaStreamDestroy(trt_ctx->cuda_stream);
    CUDA_CHECK(cudaFree(trt_ctx->buffers[trt_ctx->inputIndex]));
    //cudaFree(trt_ctx->buffers[trt_ctx->inputIndex]);
    CUDA_CHECK(cudaFree(trt_ctx->buffers[trt_ctx->outputIndex]));
    //cudaFree(trt_ctx->buffers[trt_ctx->outputIndex])
    // Destroy the engine
    trt_ctx->exe_context->destroy();
    trt_ctx->engine->destroy();
    trt_ctx->runtime->destroy();
 
    delete trt_ctx->data;
    delete trt_ctx->prob;
 
    delete trt_ctx;
 
}



#pragma once

#include "OpenCVLibrary.h"

using namespace cv;
using namespace dnn;
using namespace std;

TArray<float> VACUNT;
bool DoEnhanceImage = false;
std::vector<std::string> LayersNames;//获取神经网络中的层级名称layout
std::vector<int> OutLayers;

bool UseTCP = false;
bool UseYolov5 = true;
bool UseYolov3 = false;
bool UseSSDRes = false;
vector<Mat> Yolov5Outs;

bool DoResizeImage = true;
bool DoKeepRatio = true;
int Yolov5Width = 640;
int Yolov5Height = 640;
int Yolov5StrideNum = 3;
int NewWidth = 0;
int NewHeight = 0;
int PaddingWidth = 0;
int PaddingHeight = 0;


int Yolov3Width = 608;
int Yolov3Height = 608;
vector<Mat> Yolov3Outs;
float Yolov3Confidence = 0.5;

int SSDResWidth = 300;
int SSDResHeight = 300;
float SSDResConfidence = 0.5;
TArray<float> SSDResFaceX;
TArray<float> SSDResFaceY;
TArray<float> SSDResFaceSize;

const float Anchors640[3][6] = { {10.0,  13.0, 16.0,  30.0,  33.0,  23.0},
							 {30.0,  61.0, 62.0,  45.0,  59.0,  119.0},
							 {116.0, 90.0, 156.0, 198.0, 373.0, 326.0} };

const float Anchors1280[4][6] = { {19, 27, 44, 40, 38, 94},{96, 68, 86, 152, 180, 137},{140, 301, 303, 264, 238, 542},
					   {436, 615, 739, 380, 925, 792} };


// const FVector4 RealWorldBoundary = FVector4(0, 0, 0, 0);
// const FVector4 VirtualWorldBoundary = FVector4(0, 0, 0, 0);
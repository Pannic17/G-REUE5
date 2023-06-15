// Fill out your copyright notice in the Description page of Project Settings.


#include "CVProcessor.h"
#include "DNNConfig.h"

// Sets default values
ACVProcessor::ACVProcessor()
{
 	// Set this actor to call Tick() every frame.  You can turn this off to improve performance if you don't need it.
	PrimaryActorTick.bCanEverTick = true;

	FString NetworkPath = FPaths::GameSourceDir() + "Network/";

	/* Yolov5 Model */
	FString Yolov5ModelPath = NetworkPath + "yolov5s.onnx";
	FString Yolov5ClassPath = NetworkPath + "class.names";
	this->Yolov5Net = readNet(TCHAR_TO_UTF8(*Yolov5ModelPath));
	Anchors = (float*) Anchors640;
	// this->Yolov5Net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
	// this->Yolov5Net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA_FP16);
    if (Yolov5Net.empty())
    {
	    UE_LOG(LogTemp, Warning, TEXT("Yolov5Net Did Not Load!!!"));
    }
    else
    {
	    UE_LOG(LogTemp, Warning, TEXT("Yolov5Net Loaded!!!"));
    }

	/* Yolov3 Model */
	FString Yolov3WeightPath = NetworkPath + "yolov3.weights";
	FString Yolov3CfgPath = NetworkPath + "yolov3.cfg";
	FString Yolov3ClassPath = NetworkPath + "coco.names";
	this->Yolov3Net = readNetFromDarknet(TCHAR_TO_UTF8(*Yolov3CfgPath), TCHAR_TO_UTF8(*Yolov3WeightPath));
	this->Yolov3Net.setPreferableBackend(cv::dnn::Backend::DNN_BACKEND_OPENCV);
	this->Yolov3Net.setPreferableTarget(cv::dnn::Target::DNN_TARGET_CPU);

	/* ResNet SSD Model */
	FString ResSSDModelPath = NetworkPath + "res10_300x300_ssd_iter_140000_fp16.caffemodel";
	FString ResSSDProtoPath = NetworkPath + "deploy.prototxt";
	UE_LOG(LogTemp, Warning, TEXT("file: %s"), *ResSSDModelPath);
	this->SSDResNet = readNetFromCaffe(TCHAR_TO_UTF8(*ResSSDProtoPath), TCHAR_TO_UTF8(*ResSSDModelPath));
	this->SSDResNet.setPreferableBackend(DNN_BACKEND_OPENCV);
	this->SSDResNet.setPreferableTarget(DNN_TARGET_CPU);
}

// Called when the game starts or when spawned
void ACVProcessor::BeginPlay()
{
	Super::BeginPlay();
	if (!UseTCP)
	{
		InitCameraAndThreadRunnable(0);
	}
}

// Called every frame
void ACVProcessor::Tick(float DeltaTime)
{
	Super::Tick(DeltaTime);

	if (Yolov5Count)
	{
		UE_LOG(LogTemp, Warning, TEXT("Detected Heads: %d"), Yolov5Count);
		TArray<float> xArray;
		TArray<float> yArray;
		TArray<float> wArray;
		TArray<float> hArray;
		for (int i = 0; i < Yolov5Result.boxes.size(); ++i)
		{
			xArray.Add(Yolov5Result.center[i][0]);
			yArray.Add(Yolov5Result.center[i][1]);
			UE_LOG(LogTemp, Warning, TEXT("Detected At X %f, Y %f."), Yolov5Result.center[i][0], Yolov5Result.center[i][1]);
			UE_LOG(LogTemp, Warning, TEXT("Detected Size W %f, H %f."), Yolov5Result.size[i][0], Yolov5Result.size[i][1]);
		}
		ShowYolov5Result(Yolov5Count, xArray, yArray);
	}
	if (UseYolov3)
	{
		UE_LOG(LogTemp, Warning, TEXT("Detected Bodies: %d"), Yolov3Count);
		ShowYolov3Result(Yolov3Count);
	}
	if (UseSSDRes)
	{
		UE_LOG(LogTemp, Warning, TEXT("Detected Faces: %d"), SSDResCount);
		// TODO: Show ResNet SSD Result
	}
}

void ACVProcessor::EndPlay(const EEndPlayReason::Type EndPlayReason)
{
	Super::EndPlay(EndPlayReason);
	if (ReadThread)
	{
		ReadThread->Stop();
		ReadThread = nullptr;

	}
	if (Camera.isOpened())
	{
		Camera.release();
	}
	Yolov5Count = 0;
	Yolov3Count = 0;
	SSDResCount = 0;
}

void ACVProcessor::ReadFrame()
{
	if (Camera.isOpened())
	{
		Mat frame;
		Camera.read(frame);
		if (frame.empty())
		{
			UE_LOG(LogTemp, Warning, TEXT("Frame is Empty !!!"));
			return;
		}
		if (frame.channels() == 4)
		{
			cvtColor(frame, frame, COLOR_BGRA2BGR);
		}
		
		
		if (DoEnhanceImage)
		{
			// TODO: EnhanceImage
		}
		
		AsyncTask(ENamedThreads::GameThread, [=]()
		{
			UTexture2D* OutTexture = ConvertMat2Texture2D(frame);
			// Show Native Capture Image
			ShowNativeImage(OutTexture);
		});

		if (UseYolov5)
		{
			UE_LOG(LogTemp, Warning, TEXT("Use Yolov5 Model"));
			Yolov5Count = 0;
			DetectYolov5Head(frame);
		}
		if (UseYolov3)
		{
			UE_LOG(LogTemp, Warning, TEXT("Use Yolov3 Model"));
			Yolov3Count = 0;
			DetectYolov3Body(frame);
		}
		// if (UseSSDRes)
		// {
		// 	UE_LOG(LogTemp, Warning, TEXT("Use ResNet SSD Model"));
		// 	SSDResCount = 0;
		// 	DetectSSDResFace(frame);
		// }
	}
}

void ACVProcessor::DetectYolov5Head(Mat& Frame)
{
	// Detect With Yolov5 Model
	if (Frame.empty()) return;
	Yolov5Count = 0;
	int Width = Frame.cols;
	int Height = Frame.rows;
	if (DoResizeImage)
	{
		
		Mat Resized = ResizeImage(Frame, &NewWidth, &NewHeight, &PaddingHeight, &PaddingWidth);
		
		Mat Yolov5Bolb = blobFromImage(Resized, 1 / 255.0, Size(Yolov5Width, Yolov5Height), Scalar(0, 0, 0), true, false);

		Yolov5Net.setInput(Yolov5Bolb);
		UE_LOG(LogTemp, Warning, TEXT("Yolov5Outs Size %llu"), Yolov5Outs.size());
		for (size_t i = 0; i < Yolov5Outs.size(); ++i)
		{
			Mat output = Yolov5Outs[i];
			UE_LOG(LogTemp, Warning, TEXT("Data Size %d"), *output.size);
			UE_LOG(LogTemp, Warning, TEXT("Data Addr %p"), output.data);
		}
		
		Yolov5Net.forward(Yolov5Outs, GetOutputsNames(Yolov5Net));
		// Yolov5Net.forward(Yolov5Outs, Yolov5Net.getUnconnectedOutLayersNames());

		const int NumProposal = Yolov5Outs[0].size[1];
		// UE_LOG(LogTemp, Warning, TEXT("NumProposal: %d"), NumProposal);
		int OutLength = Yolov5Outs[0].size[2];
		if (Yolov5Outs[0].dims > 2)
		{
			Yolov5Outs[0] = Yolov5Outs[0].reshape(0, NumProposal);
		}
		float RatioWidth = static_cast<float>(Width) / NewWidth;
		float RatioHeight = static_cast<float>(Height) / NewHeight;
		// int xMin = 0, yMin = 0, xMax = 0, yMax = 0, Index = 0; 
		int RowIndex = 0;
		float* Prediction = (float*)Yolov5Outs[0].data;

		DetectionResult RawResult;

		for (int lS = 0; lS < Yolov5StrideNum; lS++)   ///����ͼ�߶�
		{
			const float Stride = pow(2, lS + 3);
			int GridXNum = static_cast<int>(ceil(Yolov5Width / Stride));
			int GridYNum = static_cast<int>(ceil(Yolov5Height / Stride));
			for (int lA = 0; lA < 3; lA++)    ///anchor
			{
				const float AnchorWidth = this->Anchors[lS * 6 + lA * 2];
				const float AnchorHeight = this->Anchors[lS * 6 + lA * 2 + 1];
				for (int lY = 0; lY < GridYNum; lY++)
				{
					for (int lX = 0; lX < GridXNum; lX++)
					{
						float BoxScore = Prediction[4];
						if (BoxScore > ObjectThreshold)
						{
							/* For specific case of head detection, class number is only 1, so col5 is used */
							// Mat ClassScores = Yolov5Outs[0].row(RowIndex).colRange(5, nout);
							// Point classIdPoint
							// Get the value and location of the maximum score
							// minMaxLoc(ClassScores, 0, &MaxClassScore, 0, &classIdPoint);
							double ClassScore = Prediction[5];
							ClassScore *= BoxScore;
							if (ClassScore > ConfigThreshold)
							{ 
								// const int class_idx = classIdPoint.x;
								float centerX = (Prediction[0] * 2.f - 0.5f + lX) * Stride;  ///cx
								float centerY = (Prediction[1] * 2.f - 0.5f + lY) * Stride;   ///cy
								float boxWidth = powf(Prediction[2] * 2.f, 2.f) * AnchorWidth;   ///w
								float boxHeight = powf(Prediction[3] * 2.f, 2.f) * AnchorHeight;  ///h

								int leftBound = static_cast<int>((centerX - PaddingWidth - 0.5 * boxWidth) * RatioWidth);
								int topBound = static_cast<int>((centerY - PaddingHeight - 0.5 * boxHeight) * RatioHeight);

								RawResult.confidences.push_back(static_cast<float>(ClassScore));
								RawResult.boxes.push_back(cv::Rect(leftBound, topBound, static_cast<int>(boxWidth * RatioWidth), static_cast<int>(boxHeight * RatioHeight)));
								RawResult.classID.push_back(0);
								RawResult.center.push_back({ centerX, centerY });
								RawResult.size.push_back({ boxWidth, boxHeight });
							}
						}
						RowIndex++;
						Prediction += OutLength;
					}
				}
			}
		}

		vector<int> indices;
		NMSBoxes(RawResult.boxes, RawResult.confidences, ConfigThreshold, NMSThreshold, indices);
		for (size_t i = 0; i < indices.size(); ++i)
		{
			int index = indices[i];
			cv::Rect box = RawResult.boxes[index];
			Yolov5Result.boxes.push_back(box);
			Yolov5Result.confidences.push_back(RawResult.confidences[index]);
			Yolov5Result.classID.push_back(0);
			Yolov5Result.center.push_back(RawResult.center[index]);
			Yolov5Result.size.push_back(RawResult.size[index]);
			Yolov5Count++;
			UE_LOG(LogTemp, Warning, TEXT("Detected At X %d, Y %d, W %d, H %d."), box.x, box.y, box.width, box.height);
		}
		UE_LOG(LogTemp, Warning, TEXT("Detected %d Head(s)."), Yolov5Count);

		if (!Yolov5Result.boxes.empty()) Yolov5Result.boxes.resize(0);
		if (!Yolov5Result.confidences.empty()) Yolov5Result.confidences.resize(0);
		if (!Yolov5Result.classID.empty()) Yolov5Result.classID.resize(0);
		if (!Yolov5Result.center.empty()) Yolov5Result.center.resize(0);
		if (!Yolov5Result.size.empty()) Yolov5Result.size.resize(0);
		if (!indices.empty()) indices.resize(0);
		if(!Yolov5Bolb.empty()) Yolov5Bolb.resize(0);
		// if (!Yolov5Bolb.empty()) Yolov5Bolb.resize(0);
	}
	else
	{
		Mat Yolov5Bolb = blobFromImage(Frame, 1 / 255.0, Size(Yolov5Width, Yolov5Height), Scalar(0, 0, 0), true, false);
		Yolov5Net.setInput(Yolov5Bolb);
		Yolov5Net.forward(Yolov5Outs, GetOutputsNames(Yolov5Net));
		PostProcessing(Yolov5Outs, Width, Height, Yolov5Width, Yolov5Height, Yolov5Result, Yolov5Count);
	}
	
}

// Detect With Yolov3 Model
void ACVProcessor::DetectYolov3Body(Mat& Frame)
{
	if (Frame.empty()) return;
	Yolov3Count = 0;
	int Width = Frame.cols;
	int Height = Frame.rows;
	Mat Yolov3Bolb = blobFromImage(Frame, 1 / 255.0, Size(Yolov3Width, Yolov3Height), Scalar(0, 0, 0),false, false);
	Yolov3Net.setInput(Yolov3Bolb);
	// TODO: Get Yolov3 Output
	Yolov3Net.forward(Yolov3Outs, GetOutputsNames(Yolov3Net));

	DetectionResult RawResult;
	
	for (size_t i = 0; i < Yolov3Outs.size(); ++i)
	{
		// Scan through all the bounding boxes output from the network and keep only the
		// ones with high confidence scores. Assign the box's class label as the class
		// with the highest score for the box.
		float* data = (float*)Yolov3Outs[i].data;
		for (int j = 0; j < Yolov3Outs[i].rows; ++j, data += Yolov3Outs[i].cols)
		{
			cv::Mat scores = Yolov3Outs[i].row(j).colRange(5, Yolov3Outs[i].cols);
			cv::Point classIdPoint;
			double confidence;
			// Get the value and location of the maximum score
			cv::minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);

			if (confidence > Yolov3Confidence)
			{
				float centerX = data[0] * Width;
				float centerY = data[1] * Height;
				float width = data[2] * Width;
				float height = data[3] * Height;
				int left = centerX - width / 2;
				int top = centerY - height / 2;

				RawResult.classID.push_back(classIdPoint.x);
				RawResult.confidences.push_back((float)confidence);
				RawResult.boxes.push_back(cv::Rect(left, top, width, height));
				RawResult.center.push_back({ centerX, centerY });
				RawResult.size.push_back({ width, height });
			}
		}
	}

	// Perform non maximum suppression to eliminate redundant overlapping boxes with
	// lower confidences
	vector<int> indices;
	NMSBoxes(RawResult.boxes, RawResult.confidences, Yolov3Confidence, NMSThreshold, indices);
	for (size_t i = 0; i < indices.size(); ++i)
	{
		int index = indices[i];
		Yolov3Count++;
		Yolov3Result.boxes.push_back(RawResult.boxes[index]);
		Yolov3Result.confidences.push_back(RawResult.confidences[index]);
		Yolov3Result.classID.push_back(0);
		Yolov3Result.center.push_back(RawResult.center[index]);
		Yolov3Result.size.push_back(RawResult.size[index]);
		//drawPred(classIds[idx], confidences[idx], box.x, box.y,box.x + box.width, box.y + box.height, frame);
	}
	if (!Yolov3Result.boxes.empty()) Yolov3Result.boxes.resize(0);
	if (!Yolov3Result.confidences.empty()) Yolov3Result.confidences.resize(0);
	if (!Yolov3Result.classID.empty()) Yolov3Result.classID.resize(0);
	if (!Yolov3Result.center.empty()) Yolov3Result.center.resize(0);
	if (!Yolov3Result.size.empty()) Yolov3Result.size.resize(0);
	if (!indices.empty()) indices.resize(0);
	if(!Yolov3Bolb.empty()) Yolov3Bolb.resize(0);
}

// Detect With ResNet SSD Model
void ACVProcessor::DetectSSDResFace(Mat& Frame)
{
	if (Frame.empty()) return;
	TArray<UTexture2D*> SSDResOuts;//类似std::Vector动态数组
	SSDResCount = 0;
	SSDResFaceX = VACUNT;
	SSDResFaceY = VACUNT;
	SSDResFaceSize = VACUNT;
	int Width = Frame.cols;
	int Height = Frame.rows;
	Mat SSDResBlob = blobFromImage(Frame, 0.5, Size(SSDResWidth, SSDResHeight), Scalar(0, 0, 0), false, false);
	SSDResNet.setInput(SSDResBlob, "data");
	Mat FaceDetection = SSDResNet.forward("detection_out");
	Mat Detections(FaceDetection.size[2], FaceDetection.size[3], CV_32F, FaceDetection.ptr<float>());
	

	UE_LOG(LogTemp, Warning, TEXT("####Detected: %d"), Detections.rows);

	for (int i = 0; i < Detections.rows; i++)
	{
		const float Confidence = Detections.at<float>(i, 2);
		if(Confidence > SSDResConfidence)
		{
			++SSDResCount;

			float xTL = Detections.at<float>(i, 3);
			float yTL = Detections.at<float>(i, 4);
			float xBR = Detections.at<float>(i, 5);
			float yBR = Detections.at<float>(i, 6);

			float w = xBR - xTL;
			float h = yBR - yTL;

			float centerX = (xTL + w / 2) * 1920;
			float centerY = (yTL + h / 2) * 1080;
			SSDResFaceX.Add(centerX);
			SSDResFaceY.Add(centerY);

			float size = sqrt(w * w + h * h);
			SSDResFaceSize.Add(size);

			UE_LOG(LogTemp, Warning, TEXT("####Face At X:%f, Y:%f, S:%f"), centerX, centerY, size);
		}
	}
}

FVector2D ACVProcessor::Convert2MapPosition(float InputX, float InputY)
{
	float OutputX = InputX;
	float OutputY = InputY;
	if (InputX < SceneLeftBound)
	{
		OutputX = (float) SceneLeftBound;
	} else if (InputX > SceneRightBound)
	{
		OutputX = (float) SceneRightBound;
	}
	if (InputY < SceneTopBound)
	{
		OutputY = (float) SceneTopBound;
	} else if (InputY > SceneBottomBound)
	{
		OutputY = (float) SceneBottomBound;
	}
	float SceneWidth = (float) (SceneRightBound - SceneLeftBound);
	float SceneHeight = (float) (SceneBottomBound - SceneTopBound);
	float MapWidth = (float) (MapRightBound - MapLeftBound);
	float MapHeight = (float) (MapBottomBound - MapTopBound);
	OutputX = (float) (MapLeftBound + (OutputX - SceneLeftBound) / SceneWidth * MapWidth);
	OutputY = (float) (MapTopBound + (OutputY - SceneTopBound) / SceneHeight * MapHeight);
	return FVector2D(OutputX, OutputY);
}

float ACVProcessor::CalculateHeight(float InputW, float InputH)
{
	float Size = sqrt(InputW * InputW + InputH * InputH);
	return Size * HeightCoefficient + HeightOffset;
}

// Convert captured image from OpenCV Mat to Texture2D for Unreal
UTexture2D* ACVProcessor::ConvertMat2Texture2D(const Mat& InMat)
{
	int32 Width = InMat.cols;
	int32 Height = InMat.rows;
	int32 Channels = InMat.channels();
	Mat OutMat;
	cvtColor(InMat, OutMat, CV_RGB2RGBA);
	UTexture2D* OutTexture = UTexture2D::CreateTransient(Width, Height);
	OutTexture->SRGB = 0;
	//NewTexture >CompressionSettings = TextureCompressionSettings::TC_Displacementmap;
	const int DataSize = Width * Height * 4;
	void* TextureData = OutTexture->PlatformData->Mips[0].BulkData.Lock(LOCK_READ_WRITE);
	FMemory::Memmove(TextureData, OutMat.data, DataSize);
	OutTexture->PlatformData->Mips[0].BulkData.Unlock();
	OutTexture->UpdateResource();
	return OutTexture;
}

// Initialize Camera and Thread Runnable
void ACVProcessor::InitCameraAndThreadRunnable(uint32 index)
{
	Async<>(EAsyncExecution::Thread, [=]()
	{
		if (Camera.open(index))
		{
			UE_LOG(LogTemp, Warning, TEXT("Open Camera Sucessful !!!"));
			Camera.set(CV_CAP_PROP_FRAME_WIDTH,1920);
			Camera.set(CV_CAP_PROP_FRAME_HEIGHT,1080);
			Camera.set(CV_CAP_PROP_FPS, 30);
			ReadThread = FReadImageRunnable::InitReadRunnable(this);
		}
		else
		{
			UE_LOG(LogTemp, Warning, TEXT("Open Camera Failed !!!"));
		}
		FPlatformProcess::Sleep(0.01);
	});
}

Mat ACVProcessor::ResizeImage(Mat InMat, int *Width, int *Height, int *Top, int *Left)
{
	const int InWidth = InMat.cols;
	const int InHeight = InMat.rows;
	*Width = Yolov5Width;
	*Height = Yolov5Height;
	Mat OutMat;
	if (DoKeepRatio && InHeight != InWidth) {
		const float InScale = static_cast<float>(InHeight) / InWidth;
		if (InScale > 1) {
			*Width = static_cast<int>(Yolov5Width / InScale);
			*Height = Yolov5Height;
			resize(InMat, OutMat, Size(*Width, *Height), INTER_AREA);
			*Left = static_cast<int>((Yolov5Width - *Width) * 0.5);
			copyMakeBorder(OutMat, OutMat, 0, 0, *Left, Yolov5Width - *Width - *Left, BORDER_CONSTANT, 114);
		}
		else {
			*Width = Yolov5Width;
			*Height = static_cast<int>(Yolov5Height * InScale);
			resize(InMat, OutMat, Size(*Width, *Height), INTER_AREA);
			*Top = static_cast<int>((Yolov5Height - *Height) * 0.5);
			copyMakeBorder(OutMat, OutMat, *Top, Yolov5Height - *Height - *Top, 0, 0, BORDER_CONSTANT, 114);
		}
	}
	else {
		resize(InMat, OutMat, Size(*Width, *Height), INTER_AREA);
	}
	return OutMat;
}

void ACVProcessor::PostProcessing(vector<Mat>& Outs, int Width, int Height, int InWidth, int InHeight, DetectionResult &Result, int &Count)
{
	float xRatio = static_cast<float>(Width / InWidth);
	float yRatio = static_cast<float>(Height / InHeight);

	float* Prediction = (float*) Outs[0].data;

	const int dimensions = 85;
	const int rows = 25200;

	DetectionResult RawResult;

	for (int i = 0; i < rows; ++i) {

		float BoxScore = Prediction[4];
		if (BoxScore >= ObjectThreshold) {

			double ClassScore = Prediction[5];
			// ClassScore *= BoxScore;
			// float* classes_score = data + 5;
			// cv::Mat scores(1, className.size(), CV_32FC1, classes_scores);
			// cv::Point class_id;
			// double max_class_score = classes_score;
			// minMaxLoc(scores, 0, &max_class_score, 0, &class_id);
			
			if (ClassScore > ConfigThreshold) {
				float x = float(Prediction[0]);
				float y = float(Prediction[1]);
				float w = float(Prediction[2]);
				float h = float(Prediction[3]);
				int leftBound = int((x - 0.5 * w) * xRatio);
				int topBound = int((y - 0.5 * h) * yRatio);
				float boxWidth = w * xRatio;
				float boxHeight = h * yRatio;
				RawResult.confidences.push_back(static_cast<float>(ClassScore));
				RawResult.boxes.push_back(cv::Rect(leftBound, topBound, static_cast<int>(boxWidth * xRatio), static_cast<int>(boxHeight * yRatio)));
				RawResult.classID.push_back(0);
				RawResult.center.push_back({ x, y });
				RawResult.size.push_back({ boxWidth, boxHeight });
			}
		}
		Prediction += dimensions;
	}
	vector<int> indices;
	NMSBoxes(RawResult.boxes, RawResult.confidences, ConfigThreshold, NMSThreshold, indices);
	for (size_t i = 0; i < indices.size(); ++i)
	{
		int index = indices[i];
		cv::Rect box = RawResult.boxes[index];
		Result.boxes.push_back(box);
		Result.confidences.push_back(RawResult.confidences[index]);
		Result.classID.push_back(0);
		Result.center.push_back(RawResult.center[index]);
		Result.size.push_back(RawResult.size[index]);
		Count++;
		UE_LOG(LogTemp, Warning, TEXT("Detected At X %d, Y %d, W %d, H %d."), box.x, box.y, box.width, box.height);
	}
	UE_LOG(LogTemp, Warning, TEXT("Detected %d Head(s)."), Count);

	// delete Prediction;
	Yolov5Outs.clear();
}

vector<String> ACVProcessor::GetOutputsNames(const Net& net)
{
	static std::vector<cv::String> names;
	if (names.empty())
	{
		//Get the indices of the output layers, i.e. the layers with unconnected outputs
		OutLayers = net.getUnconnectedOutLayers();

		//get the names of all the layers in the network
		LayersNames = net.getLayerNames();

		// Get the names of the output layers in names
		names.resize(OutLayers.size());
		for (size_t i = 0; i < OutLayers.size(); ++i)
			names[i] = LayersNames[OutLayers[i] - 1];
	}
	return names;
}


/*Thread Instance*/
FReadImageRunnable*  FReadImageRunnable::ReadInstance = nullptr;


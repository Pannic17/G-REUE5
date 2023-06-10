// Fill out your copyright notice in the Description page of Project Settings.

#pragma once

#include "CoreMinimal.h"
#include "GameFramework/Actor.h"


#include "OpenCVLibrary.h"
#include "Runtime/Core/Public/HAL/RunnableThread.h"
#include "Runtime/Core/Public/HAL/Runnable.h"

#include "CVProcessor.generated.h"

class FRunnable;
class FReadImageRunnable;
using namespace cv;
using namespace dnn;
using namespace std;

struct NetConfig
{
	float confThreshold; // Confidence threshold
	float nmsThreshold;  // Non-maximum suppression threshold
	float objThreshold;  //Object Confidence threshold
	string modelpath;
};

struct DetectionResult
{
	// int count;
	vector<float> confidences;
	vector<cv::Rect> boxes;
	vector<int> classID;
	vector<vector<float>> center;
	vector<vector<float>> size;
};



UCLASS()
class G_520TEST_API ACVProcessor : public AActor
{
	GENERATED_BODY()
	
public:	
	// Sets default values for this actor's properties
	ACVProcessor();

	/* Define Networks */
	Net Yolov5Net;
	Net Yolov3Net;
	Net SSDResNet;

	/* Camera */
	VideoCapture Camera;
	FReadImageRunnable* ReadThread;

	/* Actor Default */
	// Called when the game starts or when spawned
	virtual void BeginPlay() override;
	// Called every frame
	virtual void Tick(float DeltaTime) override;
	// Called when the game ends
	virtual void EndPlay(const EEndPlayReason::Type EndPlayReason) override;

	/* Core */
	void ReadFrame();

	/* Result Struct */
	DetectionResult Yolov5Result;
	DetectionResult Yolov3Result;

	/* Result Var - UPROPERTY */
	UPROPERTY(VisibleAnywhere, BlueprintReadOnly)
	int Yolov5Count = 0;
	UPROPERTY(VisibleAnywhere, BlueprintReadOnly)
	int Yolov3Count = 0;
	UPROPERTY(VisibleAnywhere, BlueprintReadOnly)
	int SSDResCount = 0;

	/* Show Event - UFUNCTION */
	UFUNCTION(BlueprintImplementableEvent)
	void ShowImage(UTexture2D* outRGB,int Width,int Height);
	UFUNCTION(BlueprintImplementableEvent)
	void ShowCutImage( UTexture2D* outRGB);
	UFUNCTION(BlueprintImplementableEvent)
	void ShowNativeImage(UTexture2D* outRGB);

	// TODO: Yolov5 Result
	UFUNCTION(BlueprintImplementableEvent)
	void ShowYolov5Result(int Count, const TArray<float>& CenterX, const TArray<float>& CenterY);
	// Yolov3
	UFUNCTION(BlueprintImplementableEvent)
	void ShowYolov3Result(int Count);
	// ResNet SSD
	UFUNCTION(BlueprintImplementableEvent)
	void ShowSSDResResult(int Count, const TArray<float>& FaceX, const TArray<float>& FaceY, const TArray<float>& FaceSize);
	
	/* Detections */
	void DetectYolov5Head(Mat& Frame);
	void DetectYolov3Body(Mat& Frame);
	void DetectSSDResFace(Mat& Frame);

private:
	// Define private variables and helper functions
	float* Anchors;
	
	static UTexture2D* ConvertMat2Texture2D(const Mat& InMat);
	void InitCameraAndThreadRunnable(uint32 index);

	static Mat ResizeImage(Mat InMat, int *Width, int *Height, int *Top, int *Left);
	// void CutImage(const Mat inMat, FVector2D inPos);
	// void CutImageRect(const Mat inMat, cv::Rect inRect);

	void PostProcessing(vector<Mat>& Outs, int Width, int Height, int InWidth, int InHeight, DetectionResult& Result, int& Count);

	static vector<String> GetOutputsNames(const Net& net);
	// static TArray<any> ConvertVector2TArray(const vector<any>& Vectors);
};




class G_520TEST_API FReadImageRunnable :public FRunnable
{
public:
	static FReadImageRunnable* InitReadRunnable(ACVProcessor* inActor)
	{
		if (!ReadInstance&&FPlatformProcess::SupportsMultithreading())
		{
			ReadInstance=new FReadImageRunnable(inActor);
		}
		return ReadInstance;
	}

public:

	virtual bool Init() override
	{
		StopThreadCounter.Increment();
		return true;
	}

	virtual uint32 Run() override
	{
		if (!ReadActor)
		{
			UE_LOG(LogTemp, Warning, TEXT("AHikVisionActor Actor is not spawn"));
			return 1;
		}

		
		while (StopThreadCounter.GetValue())
		{
			// UE_LOG(LogTemp, Warning, TEXT("ReadFrame"));
			// std::lock_guard<std::mutex> lock(mutex);
			ReadActor->ReadFrame();
		}

		/*
			double StartTime = FDateTime::Now().GetTimeOfDay().GetTotalMilliseconds();
			while (StopThreadCounter.GetValue())
			{
				double EndTime = FDateTime::Now().GetTimeOfDay().GetTotalMilliseconds();
				if (EndTime - StartTime > 1000)
				{
					ReadActor->ReadFrame();
					StartTime = FDateTime::Now().GetTimeOfDay().GetTotalMilliseconds();
				}
				else
				{
					continue;
				}
			}
			*/
		return 0;
	}


	virtual void Exit() override
	{
		
	}

	virtual void Stop() override
	{
		if(ReadInstance)
		{
			ReadInstance->EnsureThread();
			delete ReadInstance;
			ReadInstance = nullptr;
		}

	}
	void EnsureThread()
	{
		StopThreadCounter.Decrement();
		if (ReadImageThread) {
			ReadImageThread->WaitForCompletion();
		}
	}
protected:
	FReadImageRunnable(ACVProcessor* inReadActor) 
	{
		
		ReadActor = inReadActor;
		ReadImageThread = FRunnableThread::Create(this, TEXT("ReadImageRunnable"));
	}


	~FReadImageRunnable() {

	};


private:
	FRunnableThread* ReadImageThread;
	ACVProcessor* ReadActor;
	static FReadImageRunnable* ReadInstance;
	FThreadSafeCounter StopThreadCounter;
	std::mutex mutex;
};

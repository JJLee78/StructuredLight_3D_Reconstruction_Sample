#pragma once
#include <opencv2/core.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <fstream>
#include <nlohmann/json.hpp>

using namespace std;
using namespace cv;
using json = nlohmann::json;

class camParam 
{
	public:
	camParam();
	int nWidth;
	int nHeight;
	int nSize;
	Mat* mIntrinsic;
	Mat* mDistortion;
	Mat* mExtrinsic;
	Mat* mCenter;
	Mat* mLine;
	Mat* mDepthMap;
	Mat* mBackgroundImage;
	Mat* mBackgroundMask;

	Mat* mPatternImage;
	Mat* mDecodedImage;
	Mat* mPatternMask;

	private:
	double dGain;
	double dExposure;
	double dFramerate;
	double dWBRed;
	double dWBBlu;
	double dWBGrn;

};

class projParam
{
	public:
	projParam();
	int nWidth;
	int nHeight;
	int nSize;
	Mat* mIntrinsic;
	Mat* mDistortion;
	Mat* mExtrinsic;
	Mat* mCenter;
	Mat* mLine;
	Mat* mColPlanes;
	Mat* mBackgroundImage;
	Mat* mBackgroundMask;

	int nPatternColumn;
	int nPatternShift;
	Mat* mGrayCode;
};


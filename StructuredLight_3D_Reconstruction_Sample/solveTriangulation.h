#pragma once
#include <iostream>
#include "configDevice.h"
#include <opencv2/core.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include "../cuda_algs/cuda_algs.cuh"
using namespace cv;
using namespace std;
int solveParameters(camParam* CamParam, projParam* ProjParam);
int decodeGrayCodes(camParam* CamParam, projParam* ProjParam);
int recostruct3D(camParam* CamParam, projParam* ProjParam);
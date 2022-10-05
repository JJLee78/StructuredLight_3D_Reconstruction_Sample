#pragma once
#include <iostream>
#include "configDevice.h"
#include <opencv2/core.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

using namespace cv;
using namespace std;
int solveParameters(camParam* CamParam, projParam* ProjParam);
int decodeGrayCodes(camParam* CamParam, projParam* ProjParam);
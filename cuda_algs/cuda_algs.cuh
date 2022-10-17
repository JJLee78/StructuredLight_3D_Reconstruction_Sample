#pragma once

#include <vector>

bool bIsHaveCuda();

void undistortFunc(unsigned char* input, int bytes_per_line, int width, int height,
	float fx, float fy, float cx, float cy, float k1, float k2);

void GTensorInitialize();
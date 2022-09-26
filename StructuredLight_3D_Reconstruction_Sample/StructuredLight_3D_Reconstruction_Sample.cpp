// StructuredLight_3D_Reconstruction_Sample.cpp : 이 파일에는 'main' 함수가 포함됩니다. 거기서 프로그램 실행이 시작되고 종료됩니다.
//

#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

using namespace cv;
using namespace std;

#include "configDevice.h"
#include "solveTriangulation.h"


int main()
{
    camParam cam_1;
    projParam proj_1;
    solveParameters(&cam_1, &proj_1);
    //InitCamera(&cam_1);
    //InitProjector(&proj_1);
    //InitGreyCode(proj_1.nWidth);

    std::cout << "Hello World!\n";
}

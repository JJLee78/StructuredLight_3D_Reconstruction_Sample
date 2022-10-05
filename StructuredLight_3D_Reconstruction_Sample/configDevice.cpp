#include "configDevice.h"


camParam::camParam()
{
	std::ifstream f("..\\config.json");
	json configFile = json::parse(f);

	float intrinsicVal[9];
	float distortionVal[5];
	float rotationVal[9];
	float translationVal[3];

	for (int i = 0; i < configFile["Cam_Intri"].size(); i++)
		intrinsicVal[i] = configFile["Cam_Intri"][i];
	for (int i = 0; i < configFile["Cam_Disto"].size(); i++)
		distortionVal[i] = configFile["Cam_Disto"][i];
	for (int i = 0; i < configFile["Cam_R"].size(); i++)
		rotationVal[i] = configFile["Cam_R"][i];
	for (int i = 0; i < configFile["Cam_T"].size(); i++)
		translationVal[i] = configFile["Cam_T"][i];


	nWidth = configFile["CamWidth"]; //카메라 가로 해상도
	nHeight = configFile["CamHeight"]; //카메라 세로 해상도
	nSize = nWidth * nHeight; //카메라 모든 픽셀 수
	mIntrinsic = new Mat(Mat::zeros(3, 3, CV_32F)); //카메라 내부 파라미터
	mDistortion = new Mat(Mat::zeros(1, 5, CV_32F)); //카메라 왜곡 계수
	mExtrinsic = new Mat(Mat::zeros(2, 3, CV_32F)); //카메라 외부 파라미터
	mCenter = new Mat(Mat::zeros(3, 1, CV_32F)); //카메라 중심 좌표
	mLine = new Mat(Mat::zeros(3, nSize, CV_32F)); //카메라 광축 행렬(단위 방향벡터)
	mDepthMap = new Mat(Mat::zeros(nWidth, nHeight, CV_32F)); //깊이 정보(Depthmap)
	mBackgroundImage = new Mat(Mat::zeros(nWidth, nHeight, CV_8UC3)); //카메라 이미지 배경
	mBackgroundMask = new Mat(nWidth, nHeight, CV_8UC1, cv::Scalar(255)); //카메라 이미지 배경(마스크)

	memcpy(mIntrinsic->data, intrinsicVal, sizeof(intrinsicVal));
	memcpy(mDistortion->data, distortionVal, sizeof(distortionVal));

	cout << "--Camera Intrinsic--" << endl;
	cout << *mIntrinsic << endl;
	Mat rotmat(3, 3, CV_32F, rotationVal);
	Mat rvec = Mat::zeros(1, 3, CV_32FC1);
	Rodrigues(rotmat, rvec);
	for (int i = 0; i < 3; i++)
	{
		mExtrinsic->at<float>(0, i) = rvec.at<float>(0, i);
		mExtrinsic->at<float>(1, i) = translationVal[i];
	}
	cout << "--Camera Extrinsic--" << endl;
	cout << *mExtrinsic << endl;
	cout << "--Camera Distortion--" << endl;
	cout << *mDistortion << endl;

}

projParam::projParam()
{
	std::ifstream f("..\\config.json");
	json configFile = json::parse(f);

	float intrinsicVal[9];
	float distortionVal[5];
	float rotationVal[9];
	float translationVal[3];

	for (int i = 0; i < configFile["Proj_Intri"].size(); i++)
		intrinsicVal[i] = configFile["Proj_Intri"][i];
	for (int i = 0; i < configFile["Proj_Disto"].size(); i++)
		distortionVal[i] = configFile["Proj_Disto"][i];
	for (int i = 0; i < configFile["Proj_R"].size(); i++)
		rotationVal[i] = configFile["Proj_R"][i];
	for (int i = 0; i < configFile["Proj_T"].size(); i++)
		translationVal[i] = configFile["Proj_T"][i];

	nWidth = configFile["ProjWidth"]; // 프로젝터의 가로 해상도
	nHeight = configFile["ProjHeight"]; // 프로젝터의 세로 해상도
	nSize = nWidth * nHeight; //프로젝터 픽셀 수(가로 x 세로)
	mIntrinsic = new Mat(Mat::zeros(3, 3, CV_32F)); //프로젝터 내부 파라미터
	mDistortion = new Mat(Mat::zeros(1, 5, CV_32F)); //프로젝터 왜곡 계수
	mExtrinsic = new Mat(Mat::zeros(2, 3, CV_32F)); //프로젝터 외부 파라미터
	mCenter = new Mat(Mat::zeros(3, 1, CV_32F)); //프로젝터 중심 좌표 (이 프로그램은 프로젝터 좌표를 원점 0,0,0 으로 놓기로 함)
	mLine = new Mat(Mat::zeros(3, nSize, CV_32F)); //프로젝터 광축 행렬(단위 방향벡터)
	mColPlanes = new Mat(Mat::zeros(nWidth, 4, CV_32F)); //프로젝터 패턴 한 줄마다의 평면

	memcpy(mIntrinsic->data, intrinsicVal, sizeof(intrinsicVal));
	memcpy(mDistortion->data, distortionVal, sizeof(distortionVal));

	cout << "--Projector Intrinsic--" << endl;
	cout << *mIntrinsic << endl;

	Mat rotmat(3, 3, CV_32F, rotationVal);
	Mat rvec = Mat::zeros(1, 3, CV_32FC1);
	Rodrigues(rotmat, rvec);
	for (int i = 0; i < 3; i++)
	{
		mExtrinsic->at<float>(0, i) = rvec.at<float>(0, i);
		mExtrinsic->at<float>(1, i) = translationVal[i];
	}
	cout << "--Projera Extrinsic--" << endl;
	cout << *mExtrinsic << endl;


	// Gray code 생성부 (매번 생성할 필요는 없고 초기에만 필요하므로, 추후에는 이미지 파일을 로딩하는 방식으로 변경해야 됨)
	Mat* mGrayCode;
	nPatternColumn = ceil(log2(nWidth));
	nPatternShift = floor((pow(2, nPatternColumn) - nWidth) / 2);
	const int allPatternNum = 2 * (nPatternColumn) + 2;
	cout << "nPatternColumn / nPatternShift : " << nPatternColumn << " " << nPatternShift << endl;
	mGrayCode = new Mat[allPatternNum];
	for (int i = 0; i < allPatternNum; i++)
		mGrayCode[i] = Mat::zeros(nHeight, nWidth, CV_8UC1);
	mGrayCode[0] = Scalar(255); //+
	mGrayCode[1] = Scalar(0); //+
	int step = mGrayCode[0].cols / sizeof(uchar);

	for (int c = 0; c < nWidth; c++) { // 0 ~ 1280

		for (int i = 0; i < allPatternNum-2; i += 2) { // 0 - 22
			uchar* data = (uchar*)mGrayCode[i + 2].data;
			if (i > 0)
				data[c] = (((c + nPatternShift) >> (nPatternColumn - i / 2 - 1)) & 1) ^ (((c + nPatternShift) >> (nPatternColumn - i / 2)) & 1); //(0~1280 + 384) >> ((11 - 0~22 / 2)) & 1
			else
				data[c] = (((c + nPatternShift) >> (nPatternColumn - i / 2 - 1)) & 1);
			data[c] *= 255;
			for (int r = 1; r < nHeight; r++)
				data[r * step + c] = data[c];
			
			mGrayCode[i + 3] = ~mGrayCode[i + 2];
		}
	}
	for (int i = 0; i < allPatternNum; i++)
	{
	//	cv::imshow("mGrayCode", mGrayCode[i]);
	//	cv::waitKey(1000);
	}
}


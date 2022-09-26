#include "solveTriangulation.h"


int solveParameters(camParam* CamParam, projParam* ProjParam)
{
	Mat rotationTemp(1, 3, CV_32F);
	Mat camRotation(3, 3, CV_32FC1);
	Mat camTranslation(3, 1, CV_32FC1);
	Mat proRotation(3, 3, CV_32FC1);
	Mat proTranslation(3, 1, CV_32FC1);
	rotationTemp = CamParam->mExtrinsic->row(0);
	Rodrigues(rotationTemp, camRotation);
	rotationTemp = ProjParam->mExtrinsic->row(0);
	Rodrigues(rotationTemp, proRotation);

	transpose(CamParam->mExtrinsic->row(1), camTranslation);
	*CamParam->mCenter = camRotation.inv() * camTranslation * -1; // Camera position = -R^-1 * (0-T) = -R^-1 * T

	// 영상 좌표를 카메라 좌표로 전환	
	Mat camDistorted(CamParam->nSize, 1, CV_32FC2);
	Mat camUndistorted(CamParam->nSize, 1, CV_32FC2);
	for (int y = 0; y < CamParam->nHeight; y++)
	{
		for (int x = 0; x < CamParam->nWidth; x++)
		{
			Vec2f& p1 = camDistorted.at<Vec2f>(CamParam->nWidth * y + x);
			p1[0] = float(x);
			p1[1] = float(y);
		}
	}
	undistortPoints(camDistorted, camUndistorted, *CamParam->mIntrinsic, *CamParam->mDistortion);

	//카메라 좌표계를 기준으로, 카메라 투사중심을 통과하는 방향벡터를 계산함
	for (int i = 0; i < CamParam->nSize; i++) {
		Vec2f& p1 = camUndistorted.at<Vec2f>(i);
		float norm = (float)sqrt(pow(p1.val[0], 2) + pow(p1.val[1], 2) + 1);
		CamParam->mLine->at<float>(0, i) = (float)p1.val[0] / norm;
		CamParam->mLine->at<float>(1, i) = (float)p1.val[1] / norm;
		CamParam->mLine->at<float>(2, i) = (float)1.0 / norm;
	}
	

	//cout << "----Projector parameters----" << endl;
	//cout << *ProjParam->mIntrinsic << endl;
	//cout << *ProjParam->mDistortion << endl;
	//cout << *ProjParam->mExtrinsic << endl;
	//cout << "----Projector parameters----" << endl;
	Mat proj_dist_points(ProjParam->nSize, 1, CV_32FC2);
	Mat proj_undist_points(ProjParam->nSize, 1, CV_32FC2);
	for (int y = 0; y < ProjParam->nHeight; y++)
	{
		for (int x = 0; x < ProjParam->nWidth; x++)
		{
			Vec2f& p1 = proj_dist_points.at<Vec2f>(ProjParam->nWidth * y + x);
			p1[0] = float(x);
			p1[1] = float(y);
		}
	}
	undistortPoints(proj_dist_points, proj_undist_points, *ProjParam->mIntrinsic, *ProjParam->mDistortion);

	//프로젝터 좌표계를 기준으로, 프로젝터 투사중심을 통과하는 방향벡터 구해짐
	for (int i = 0; i < ProjParam->nSize; i++) {
		Vec2f& p1 = proj_undist_points.at<Vec2f>(i);
		float norm = (float)sqrt(pow(p1.val[0], 2) + pow(p1.val[1], 2) + 1);
		ProjParam->mLine->at<float>(0, i) = (float)p1.val[0] / norm;
		ProjParam->mLine->at<float>(1, i) = (float)p1.val[1] / norm;
		ProjParam->mLine->at<float>(2, i) = (float)1.0 / norm;
	} 
	
	//프로젝터의 광축을 카메라 좌표계로 전환
	Mat R = Mat::eye(3, 3, CV_32FC1);
	R = camRotation * proRotation.t();
	*ProjParam->mLine = R * (*ProjParam->mLine);

	// 프로젝터 광평면(세로 줄)의 방정식 구하기 (카메라 좌표계 내)
	for (int x = 0; x < ProjParam->nWidth; x++) { // 프로젝터 가로 해상도
		Mat points = cv::Mat(ProjParam->nHeight + 1, 3, CV_32FC1); //프로젝터 광평면 내 하나의 점                           
		for (int y = 0; y < ProjParam->nHeight; y++) // 프로젝터 세로 해상도
		{
			int ri = (ProjParam->nWidth) * y + x;
			for (int i = 0; i < 3; i++)
			{
				points.at<float>(3 * y + i) = ProjParam->mCenter->at<float>(i, 1) + ProjParam->mLine->at<float>(ri + ProjParam->nSize * i); //하나의 광선은 광선상 하나의 점과 광선방향벡터의 실수배를 합을 만족하는 모든 점들의 집합으로 표현됨
			}
		}

		// centroid : 광선평면상의 하나의 3D 점을 프로젝터 평면의 세로줄 수로 나눔. matching points 의 geometric centroids (6.2 바로 위에 식 있음)
		// point2 : 광선평면상의 하나의 3D 점에서 centroid를 뺌
		// A : points2^T * points
		float plane[4];
		Mat centroid = Mat::zeros(1, points.cols, CV_32FC1);
		for (int x = 0; x < points.cols; x++) { // 0 - 3
			for (int y = 0; y < points.rows; y++) // 0 - height
				centroid.at<float>(x) += points.at<float>(points.cols * y + x);
			centroid.at<float>(x) /= points.rows; // Gₓ = (x₁ + x₂ + x₃ + ... + xk) / k -> 논문 6.2의 식.
		}

		//식 6.2에서 각 3D 점에 대응하여 geometric centroid를 뺌.
		Mat points2 = Mat::zeros(points.rows, points.cols, CV_32FC1);
		for (int r = 0; r < points.rows; r++)  // 0 - height
			for (int c = 0; c < points.cols; c++) // 0 - 3
				points2.at<float>(points.cols * r + c) = points.at<float>(points.cols * r + c) - centroid.at<float>(c);

		Mat A = Mat::zeros(points.cols, points.cols, CV_32FC1);
		Mat W = Mat::zeros(points.cols, points.cols, CV_32FC1);
		Mat U = Mat::zeros(points.cols, points.cols, CV_32FC1);
		Mat V = Mat::zeros(points.cols, points.cols, CV_32FC1);
		
		A = points2.t() * points;
		SVDecomp(A, W, U, V); //3차원 점들이 여러개 주어지면 가장 적합한 평면의 방정식을 svd를 통해 도출해 낸다.(Best Fit K-Plane)
		plane[points.cols] = 0;
		for (int c = 0; c < points.cols; c++) 
		{
			plane[c] = V.at<float>(points.cols * (points.cols - 1) + c);
			plane[points.cols] += plane[c] * centroid.at<float>(c);
		}
		//cout << "cent rows/cols : "<< centroid.rows << " " << centroid.cols  << endl;
		//cout << centroid << endl;
		for (int i = 0; i < 4; i++)
			ProjParam->mColPlanes->at<float>(4 * x + i) = plane[i];
	}
	return 0;
}
#include "solveTriangulation.h"


//T//
#include <chrono>
chrono::system_clock::time_point StartTime;
chrono::system_clock::time_point EndTime;
float FinalTime;
#define TSTART StartTime = chrono::system_clock::now();
#define TEND EndTime = chrono::system_clock::now();
#define TPRINT cout <<  "ELPASED " << chrono::duration_cast<chrono::milliseconds>(EndTime - StartTime).count() << " ms" <<endl;
//T//


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
	
	//transpose(ProjParam->mExtrinsic->row(1), proTranslation);
	//*ProjParam->mCenter = proRotation.inv() * proTranslation * -1; // Camera position = -R^-1 * (0-T) = -R^-1 * T

	transpose(CamParam->mExtrinsic->row(1), camTranslation);
	*CamParam->mCenter = camRotation.inv() * camTranslation * -1; // Camera position = -R^-1 * (0-T) = -R^-1 * T
	if (bIsHaveCuda())
		cout << "CUDA ENABLE" << endl;
	else
		cout << "CUDA DISABLE" << endl;

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
	TSTART;
	undistortPoints(camDistorted, camUndistorted, *CamParam->mIntrinsic, *CamParam->mDistortion);
	TEND;
	TPRINT;

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

		for (int i = 0; i < 4; i++)
		{
			ProjParam->mColPlanes->at<float>(4 * x + i) = plane[i];
		}
		if (x == 640)
		{
			//cout << "---640---" << endl;
			//for (int i = 0; i < 4; i++)
			//{
			//	//cout << "plane[i] : " << plane[i] << endl;
			//	//cout << ProjParam->mColPlanes->data[4 * x + i];
			//}
			//cout << endl;
			//for (int i = 0; i < 4; i++)
			//{
			//	//cout << "plane[i] : " << plane[i] << endl;
			//	cout << ProjParam->mColPlanes->at<float>(4 * x + i);
			//}
			//cout << endl;
			//cout << "---end---" << endl;
		}
	}


	return 0;
}

int decodeGrayCodes(camParam* CamParam, projParam* ProjParam)
{
	int allPatternNum = 2 * (ProjParam->nPatternColumn + 1);
	CamParam->mPatternImage = new Mat[allPatternNum];
	CamParam->mPatternMask = new Mat(1, CamParam->nSize, CV_32FC1, Scalar(0));
	CamParam->mDecodedImage = new Mat(CamParam->nHeight, CamParam->nWidth, CV_16UC1, Scalar(0));
//	CamParam->mTextureColor = new Mat(3, CamParam->nSize, CV_32FC1);
	TSTART;
	//CamParam->mTextureImage = new Mat(CamParam->nHeight, CamParam->nWidth);
	for (int i = 0; i < allPatternNum; i++)
	{
		CamParam->mPatternImage[i] = Mat(CamParam->nHeight, CamParam->nWidth, CV_8UC1, 1);
		string s("..\\Data_cartman\\" + to_string(i) + ".bmp");
		CamParam->mPatternImage[i] = cv::imread(s, IMREAD_GRAYSCALE);
		if (CamParam->mPatternImage[i].empty())
		{
			cout << "FAILED TO READ PATTERN IMAGE" << endl;
			exit(0);
		}
		//if( i == 0)
		//	*CamParam->mTextureImage = cv::imread(s, IMREAD_COLOR);
	}

	shared_ptr<Mat> mPatternOld(new Mat(CamParam->nHeight, CamParam->nWidth, CV_8UC1));
	shared_ptr<Mat> mPatternNew(new Mat(CamParam->nHeight, CamParam->nWidth, CV_8UC1));
	shared_ptr<Mat> mMask(new Mat(CamParam->nHeight, CamParam->nWidth, CV_8UC1));
	shared_ptr<Mat> mBitPlane(new Mat(CamParam->nHeight, CamParam->nWidth, CV_8UC1));
	shared_ptr<Mat> temp(new Mat(CamParam->nHeight, CamParam->nWidth, CV_8UC1));

	double threshold = 32; //디코딩 스레숄드, 값이 높을수록 디코딩 영역이 넓어짐
	*mPatternOld = CamParam->mPatternImage[0];
	*mPatternNew = CamParam->mPatternImage[1];

	cv::absdiff(*mPatternOld, *mPatternNew, *temp);
	cv::compare(*temp, threshold, *temp, CMP_GE);
	cv::bitwise_or(*temp, *CamParam->mBackgroundMask, *CamParam->mBackgroundMask);

	for (int i = 0; i < ProjParam->nPatternColumn; i++) {
		*mPatternOld = CamParam->mPatternImage[2 * (i + 1)];
		*mPatternNew = CamParam->mPatternImage[(2 * (i + 1)) + 1];

		cv::compare(*mPatternOld, *mPatternNew, *mBitPlane, CMP_GE); //gray1이 gray2의 픽셀값 이상일 때(빛이 있을 때) 결과를 bit_plane_2에 저장

		if (i == 0)
			mBitPlane->copyTo(*mMask);
		else
			cv::bitwise_xor(*mMask, *mBitPlane, *mMask);

		add(*CamParam->mDecodedImage, Scalar(pow(2.0, ProjParam->nPatternColumn - i - 1)), *CamParam->mDecodedImage, *mMask); //bit_plane_1로 마스크된 각 픽셀마다 패턴 장수별로 1024, 512, 256 ..등등의 스칼라 값을 더함. bit_plane_1은 8비트 이미지므로, 동일한 픽셀에서 bit_plane_1에선 255면 값을 더하고, 0이면 더하지 않는다.
	}
	//정리하자면 결국 모든 장수에서 흰색 줄이 highlited 되면 그만큼의 가중치를 더하게 된다.
	//낮고, 듬성듬성한 패턴에서는 가중치가 높고, 높고 세밀한 패턴에서는 가중치가 그만큼 낮다.
	// 가중치는 1024, 512, 256 ... 등등 2의 n제곱 형태로 나타난다.
	subtract(*CamParam->mDecodedImage, Scalar(ProjParam->nPatternShift), *CamParam->mDecodedImage);
	cv::compare(*CamParam->mDecodedImage, Scalar(ProjParam->nWidth - 1), *temp, CMP_LE);
	cv::bitwise_and(*temp, *CamParam->mBackgroundMask, *CamParam->mBackgroundMask);
	cv::compare(*CamParam->mDecodedImage, 0, *temp, CMP_GE);
	cv::bitwise_and(*temp, *CamParam->mBackgroundMask, *CamParam->mBackgroundMask);
	cv::bitwise_not(*CamParam->mBackgroundMask, *temp);
	TEND;
	TPRINT;
	return 0;
}

int recostruct3D(camParam* CamParam, projParam* ProjParam)
{
	shared_ptr<Mat> points(new Mat(3, CamParam->nSize, CV_32FC1));
	shared_ptr<Mat> colors(new Mat(3, CamParam->nSize, CV_32FC1));
	shared_ptr<Mat> depth_map(new Mat(CamParam->nHeight, CamParam->nWidth, CV_32FC1));
	shared_ptr<Mat> mask(new Mat(1, CamParam->nSize, CV_32FC1, Scalar(0)));


	int correspond_max = 100;
	int correspond_min = 100;
	int depth_counter = 0;
	uchar* gray_mask_data;
	TSTART;
	for (int r = 0; r < CamParam->nHeight; r++) 
	{
		for (int c = 0; c < CamParam->nWidth; c++) 
		{ 
			if (CamParam->mBackgroundMask->at<uchar>(r * CamParam->nWidth + c)) //mBackgroundMask가 0이 아닐 시
			{
				float pointCol[3], point_rows[3];
				float depthCol, depth_rows;

				// 픽셀별 카메라 line(ray)과 plane equation을 intersection, ray-plane intersection
				float q[3], v[3], w[4];
				int rc = (CamParam->nWidth) * r + c; //r : cam_h / c : cam_w
				for (int i = 0; i < 3; i++) {
					q[i] = CamParam->mCenter->at<float>(i);
					v[i] = CamParam->mLine->at<float>(rc + CamParam->nSize * i);
				}

				int corresponding_column = CamParam->mDecodedImage->at<unsigned short>(r * CamParam->nWidth + c); //디코딩 된 이미지에서 pixel 단위로 가져와 최대/최소값 계산.
				if (corresponding_column > correspond_max)
					correspond_max = corresponding_column;
				if (corresponding_column < correspond_min && corresponding_column != 0)
					correspond_min = corresponding_column;

				for (int i = 0; i < 4; i++)
					w[i] = ProjParam->mColPlanes->at<float>(4 * corresponding_column + i); // Best Fit K-plane 결과값으로 얻은 plane

				/*if (r == 500 && c == 600) //검증용
				{
					cout << " q[0] v[0] w[0] w[1] : " << q[0] << " / " << v[0] << " / " << w[0] << " / " << w[1] << endl;
					cout << " q[1] v[1] w[2]	  : " << q[1] << " / " << v[1] << " / " << w[2] << endl;
					cout << " q[2] v[2] w[3]	  : " << q[2] << " / " << v[2] << " / " << w[3] << endl;
				}*/


				float nq = 0, nv = 0;
				for (int i = 0; i < 3; i++) {
					nq += w[i] * q[i]; //q : mCenter
					nv += w[i] * v[i]; //v : mLine
				}

				// plane(w-nq)  ray(nv) intersection
				depthCol = (w[3] - nq) / nv;
				for (int i = 0; i < 3; i++)
					pointCol[i] = q[i] + depthCol * v[i];

				// 삼각측량으로 depth를 계산
				depth_map->at<float>(CamParam->nWidth * r + c) = depthCol;
				for (int i = 0; i < 3; i++)
					points->at<float>(CamParam->nWidth * r + c + CamParam->nSize * i) = pointCol[i];

				if (r == 1024 && c == 1536)
				{
					cout << "depth_cols : " << depth_map->at<float>(CamParam->nWidth * r + c) << endl;
					cout << "points : " << points->at<float>(CamParam->nWidth * r + c + CamParam->nSize * 0) << " "
						<< points->at<float>(CamParam->nWidth * r + c + CamParam->nSize * 1) << " "
						<< points->at<float>(CamParam->nWidth * r + c + CamParam->nSize * 2) << endl;
				}
				
				CamParam->mPatternMask->at<float>(CamParam->nWidth * r + c) = 1;

				// 절두체를 만들기 위해 fDistance의 거리 내에 있는 depth만 출력
				if (depth_map->at<float>(CamParam->nWidth * r + c) < CamParam->fDistanceMin ||
					depth_map->at<float>(CamParam->nWidth * r + c) > CamParam->fDistanceMax)
				{
					depth_counter++;
					CamParam->mBackgroundMask->at<uchar>(r * CamParam->nWidth + c) = 0;
					CamParam->mPatternMask->at<float>(CamParam->nWidth * r + c) = 0;
					depth_map->at<float>(CamParam->nWidth * r + c) = 0;
					for (int i = 0; i < 3; i++)
						points->at<float>(CamParam->nWidth * r + c + CamParam->nSize * i) = 0;
					//텍스처 컬러링
				}

			}
		}
	}
	TEND;
	TPRINT;

	//cout << "depth_counter : " << depth_counter << endl;
	char str[1024], outputDir[1024]; int scanindex = 3;
	//cv::FileStorage file("depth_map.ext", cv::FileStorage::WRITE);
	//file << "matName" << *depth_map;
	string s("..\\Data\\");

	sprintf(outputDir, "%stestobject.ply", s);
	TSTART;
	Mat depth_map_image = Mat(CamParam->nHeight, CamParam->nWidth, CV_8U, 1);
	for (int r = 0; r < CamParam->nHeight; r++) {
		for (int c = 0; c < CamParam->nWidth; c++) {
			if (CamParam->mPatternMask->at<float>(r * CamParam->nWidth + c))
			{
				depth_map_image.at<unsigned char>(CamParam->nWidth* r + c) = 
				(uchar)(255 - int(255 * (depth_map->at<float>(CamParam->nWidth * r + c) - CamParam->fDistanceMin) / (CamParam->fDistanceMax - CamParam->fDistanceMin)));
			}
			else
				depth_map_image.at<unsigned char>(CamParam->nWidth* r + c) = 0;
		}
	}
	TEND;
	TPRINT;
	cv::namedWindow("depthmap", CV_WINDOW_AUTOSIZE);

	Mat camrt2 = depth_map_image;
	resize(camrt2, camrt2, Size(CamParam->nWidth / 4, CamParam->nHeight / 4));
	imshow("depthmap", camrt2);
	waitKey(0);

	FILE* pFile = fopen(outputDir, "w");
	if (pFile == NULL) {
		fprintf(stderr, "\n");
		return -1;
	}
	fprintf(pFile, "ply\n");
	fprintf(pFile, "format ascii 1.0\n");
	
	if (points != NULL) {
		int cnt = 0;
		for (int c = 0; c < points->cols; c++) {
			if (CamParam->mPatternMask == NULL || CamParam->mPatternMask->at<float>(c) != 0)
				cnt++;
		}
		fprintf(pFile, "element vertex %d\n", cnt);
		fprintf(pFile, "property float x\nproperty float y\nproperty float z\n");
		fprintf(pFile, "end_header\n");
		cout << "vertex : " << cnt << endl;
		for (int c = 0; c < points->cols; c++) {
			if (CamParam->mPatternMask == NULL || CamParam->mPatternMask->at<float>(c) != 0) {

				for (int r = 0; r < points->rows; r++) {
					if (r != 1)
						fprintf(pFile, "    %f ", points->at<float>(c + points->cols * r));
					else
						fprintf(pFile, "    %f ", points->at<float>(c + points->cols * r));
				}
				fprintf(pFile, "\n");
			}
		}
	}

	return 0;
}

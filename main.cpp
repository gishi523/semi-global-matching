#include <iostream>
#include <chrono>
#include <opencv2/opencv.hpp>

#include "semi_global_matching.h"

int main(int argc, char* argv[])
{
	if (argc < 4)
	{
		std::cout << "usage: " << argv[0] << " left-image-format right-image-format camera.xml" << std::endl;
		return -1;
	}

	SemiGlobalMatching::Parameters param;
	// param.numPaths = 8;
	SemiGlobalMatching sgm(param);
	
	for (int frameno = 1;; frameno++)
	{
		char buf1[256];
		char buf2[256];
		sprintf(buf1, argv[1], frameno);
		sprintf(buf2, argv[2], frameno);

		cv::Mat I1 = cv::imread(buf1, -1);
		cv::Mat I2 = cv::imread(buf2, -1);

		if (I1.empty() || I2.empty())
		{
			std::cerr << "imread failed." << std::endl;
			break;
		}

		const auto t1 = std::chrono::system_clock::now();

		cv::Mat disparity = sgm.compute(I1, I2);

		const auto t2 = std::chrono::system_clock::now();
		const auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
		std::cout << "disparity computation time: " << duration << "[msec]" << std::endl;
		
		disparity.convertTo(disparity, CV_32F, 1. / SemiGlobalMatching::DISP_SCALE);
		if (I1.type() == CV_16U) cv::normalize(I1, I1, 0, 255, cv::NORM_MINMAX, CV_8U);

		cv::imshow("image", I1);
		cv::imshow("disparity", disparity / param.numDisparities);
		const char c = cv::waitKey(1);
		if (c == 27)
			break;
	}
}

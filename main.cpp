#include <opencv2/opencv.hpp>
#include <chrono>
#include "semi_global_matching.h"

int main(int argc, char* argv[])
{
	if (argc < 3)
	{
		std::cout << "usage: " << argv[0] << " left-image-format right-image-format" << std::endl;
		return -1;
	}

	SemiGlobalMatching::Parameters param;
	SemiGlobalMatching sgm(param);
	cv::Mat D1, D2;
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

		CV_Assert(I1.size() == I2.size() && I1.type() == I2.type());
		if (I1.type() == CV_16U)
		{
			cv::normalize(I1, I1, 0, 255, cv::NORM_MINMAX);
			cv::normalize(I2, I2, 0, 255, cv::NORM_MINMAX);
			I1.convertTo(I1, CV_8U);
			I2.convertTo(I2, CV_8U);
		}

		const auto t1 = std::chrono::system_clock::now();

		sgm.compute(I1, I2, D1, D2);

		const auto t2 = std::chrono::system_clock::now();
		const auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
		std::cout << "disparity computation time: " << duration << "[msec]" << std::endl;

		D1.setTo(0, D1 == SemiGlobalMatching::DISP_INV);
		D1.convertTo(D1, CV_32F, 1. / SemiGlobalMatching::DISP_SCALE);

		cv::imshow("image", I1);
		cv::imshow("disparity", D1 / param.numDisparities);
		const char c = cv::waitKey(1);
		if (c == 27)
			break;
	}
}

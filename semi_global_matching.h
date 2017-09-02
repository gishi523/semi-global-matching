#ifndef __SEMI_GLOBAL_MATCHING_H__
#define __SEMI_GLOBAL_MATCHING_H__

#include <opencv2/opencv.hpp>

class SemiGlobalMatching
{
public:

	static const int DISP_SCALE = 16;

	struct Parameters
	{
		int P1;
		int P2;
		int numDisparities;
		int numPaths;

		// default settings
		Parameters()
		{
			P1 = 20;
			P2 = 100;
			numDisparities = 64;
			numPaths = 4;
		}
	};

	SemiGlobalMatching(const Parameters& param = Parameters());
	cv::Mat compute(const cv::Mat& I1, const cv::Mat& I2);
	
private:

	Parameters param_;
};

#endif // !__SEMI_GLOBAL_MATCHING_H__
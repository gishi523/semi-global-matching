#ifndef __SEMI_GLOBAL_MATCHING_H__
#define __SEMI_GLOBAL_MATCHING_H__

#include <opencv2/opencv.hpp>

class SemiGlobalMatching
{
public:

	static const int DISP_SHIFT = 4;
	static const int DISP_SCALE = (1 << DISP_SHIFT);
	static const int DISP_INV = static_cast<ushort>(-1);

	enum CensusType
	{
		CENSUS_9x7,
		SYMMETRIC_CENSUS_9x7,
	};

	struct Parameters
	{
		int P1;
		int P2;
		int numDisparities;
		int max12Diff;
		int medianKernelSize;
		CensusType censusType;

		// default settings
		Parameters()
		{
			P1 = 20;
			P2 = 100;
			numDisparities = 64;
			max12Diff = 5;
			medianKernelSize = 3;
			censusType = SYMMETRIC_CENSUS_9x7;
		}
	};

	SemiGlobalMatching(const Parameters& param = Parameters());
	cv::Mat compute(const cv::Mat& I1, const cv::Mat& I2);

private:

	Parameters param_;
};

#endif // !__SEMI_GLOBAL_MATCHING_H__
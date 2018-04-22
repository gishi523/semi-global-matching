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
	void compute(const cv::Mat& I1, const cv::Mat& I2, cv::Mat& D1, cv::Mat& D2);

private:

	cv::Mat_<uint32_t> census32[2];
	cv::Mat_<uint64_t> census64[2];
	cv::Mat1w MC, S;
	std::vector<cv::Mat1w> L, minL;

	Parameters param_;
};

#endif // !__SEMI_GLOBAL_MATCHING_H__
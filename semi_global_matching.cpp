#include "semi_global_matching.h"
#include "timer.h"

#ifdef _WIN32
#define popcnt32 __popcnt
#define popcnt64 __popcnt64
#else
#define popcnt32 __builtin_popcount
#define popcnt64 __builtin_popcountll
#endif

namespace cv
{
	using Mat1u32 = Mat_<uint32_t>;
}

static inline int HammingDistance32(uint32_t c1, uint32_t c2)
{
	return static_cast<int>(popcnt32(c1 ^ c2));
};

static inline int HammingDistance64(uint64_t c1, uint64_t c2)
{
	return static_cast<int>(popcnt64(c1 ^ c2));
};

static void symmetricCensus9x7(const cv::Mat& src, cv::Mat1u32& dst)
{
	const int RADIUS_U = 9 / 2;
	const int RADIUS_V = 7 / 2;

	int v;
	for (v = RADIUS_V; v < src.rows - RADIUS_V; v++)
	{
		for (int u = RADIUS_U; u < src.cols - RADIUS_U; u++)
		{
			uint32_t c = 0;
			for (int dv = -RADIUS_V; dv <= -1; dv++)
			{
				for (int du = -RADIUS_U; du <= RADIUS_U; du++)
				{
					const int v1 = v + dv;
					const int v2 = v - dv;
					const int u1 = u + du;
					const int u2 = u - du;
					c <<= 1;
					c += src.ptr(v1)[u1] <= src.ptr(v2)[u2] ? 0 : 1;
				}
			}
			{
				int dv = 0;
				for (int du = -RADIUS_U; du <= -1; du++)
				{
					const int v1 = v + dv;
					const int v2 = v - dv;
					const int u1 = u + du;
					const int u2 = u - du;
					c <<= 1;
					c += src.ptr(v1)[u1] <= src.ptr(v2)[u2] ? 0 : 1;
				}
			}

			dst(v, u) = c;
		}
	}
}

static void calcMatchingCost(const cv::Mat1u32& census1, const cv::Mat1u32& census2, cv::Mat1w& MC, int n)
{
	int v;
//#pragma omp parallel for
	for (v = 0; v < census1.rows; v++)
	{
		for (int u = 0; u < n; u++)
		{
			for (int d = 0; d <= u; d++)
			{
				const uint32_t c1 = census1(v, u);
				const uint32_t c2 = census1(v, u - d);
				MC(v, u, d) = static_cast<uint16_t>(HammingDistance32(c1, c2));
			}
			for (int d = u + 1; d < n; d++)
				MC(v, u, d) = 64;
		}
		for (int u = n; u < census1.cols; u++)
		{
			for (int d = 0; d < n; d++)
			{
				const uint32_t c1 = census1(v, u);
				const uint32_t c2 = census1(v, u - d);
				MC(v, u, d) = static_cast<uint16_t>(HammingDistance32(c1, c2));
			}
		}
	}
}

SemiGlobalMatching::SemiGlobalMatching(const Parameters & param) : param_(param)
{
}

void SemiGlobalMatching::compute(const cv::Mat& I1, const cv::Mat& I2, cv::Mat& D1)
{
	CV_Assert(I1.type() == CV_8U && I2.type() == CV_8U);
	CV_Assert(I1.size() == I2.size());

	const int h = I1.rows;
	const int w = I1.cols;
	const int n = param_.numDisparities;
	const int dims[3] = { h, w, n };

	Timer t;

	t.start("census");

	// census transform
	census32[0].create(h, w);
	census32[1].create(h, w);
	symmetricCensus9x7(I1, census32[0]);
	symmetricCensus9x7(I2, census32[1]);

	t.start("matching cost");

	// calculate matching cost
	MC.create(3, dims);
	calcMatchingCost(census32[0], census32[1], MC, n);

	t.plot();

	D1.create(h, w, CV_16U);
	D1 = 0;
}

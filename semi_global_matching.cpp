#include "semi_global_matching.h"
#include "timer.h"
#include <thread>

#define USE_OPENMP
#if defined(_OPENMP) && defined(USE_OPENMP)
#define OMP_PARALLEL_FOR omp parallel for
#else
#define OMP_PARALLEL_FOR
#endif

#ifdef _WIN32
#define popcnt32 __popcnt
#define popcnt64 __popcnt64
#else
#define popcnt32 __builtin_popcount
#define popcnt64 __builtin_popcountll
#endif

#define WITH_SSE
#ifdef WITH_SSE
#include <smmintrin.h>
static inline int _mm_hmin_epu16(__m128i v)
{
	const __m128i minpos = _mm_minpos_epu16(v);
	return _mm_extract_epi16(minpos, 0);
}
#endif

namespace cv
{
	using Mat1u32 = Mat_<uint32_t>;
	using Mat1u64 = Mat_<uint64_t>;
}

template <typename T>
static inline int HammingDistance(T c1, T c2) { return 0; }

template <>
static inline int HammingDistance(uint64_t c1, uint64_t c2) { return static_cast<int>(popcnt64(c1 ^ c2)); }

template <>
static inline int HammingDistance(uint32_t c1, uint32_t c2) { return static_cast<int>(popcnt32(c1 ^ c2)); }

static inline int min4(int x, int y, int z, int w)
{
	return std::min(std::min(x, y), std::min(z, w));
};

static void census9x7(const cv::Mat& src, cv::Mat1u64& dst)
{
	const int RADIUS_U = 9 / 2;
	const int RADIUS_V = 7 / 2;
	int v;
#pragma OMP_PARALLEL_FOR
	for (v = RADIUS_V; v < src.rows - RADIUS_V; v++)
	{
		for (int u = RADIUS_U; u < src.cols - RADIUS_U; u++)
		{
			uint64_t c = 0;
			for (int dv = -RADIUS_V; dv <= RADIUS_V; dv++)
			{
				for (int du = -RADIUS_U; du <= RADIUS_U; du++)
				{
					c <<= 1;
					c += src.ptr(v)[u] <= src.ptr(v + dv)[u + du] ? 0 : 1;
				}
			}
			dst(v, u) = c;
		}
	}
}

static void symmetricCensus9x7(const cv::Mat& src, cv::Mat1u32& dst)
{
	const int RADIUS_U = 9 / 2;
	const int RADIUS_V = 7 / 2;

	int v;
#pragma OMP_PARALLEL_FOR
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

template <typename T>
static void calcMatchingCost(const cv::Mat_<T>& census1, const cv::Mat_<T>& census2, cv::Mat1w& MC, int n)
{
	const int h = census1.rows;
	const int w = census1.cols;
	int v;
#pragma OMP_PARALLEL_FOR
	for (v = 0; v < h; v++)
	{
		const T* _census1 = census1.ptr<T>(v);
		const T* _census2 = census2.ptr<T>(v);
		for (int u = 0; u < n; u++)
		{
			ushort* _MC = MC.ptr<ushort>(v, u);
			for (int d = 0; d <= u; d++)
			{
				const T c1 = _census1[u];
				const T c2 = _census2[u - d];
				_MC[d] = static_cast<ushort>(HammingDistance(c1, c2));
			}
			for (int d = u + 1; d < n; d++)
			{
				_MC[d] = 64;
			}
		}
		for (int u = n; u < w; u++)
		{
			ushort* _MC = MC.ptr<ushort>(v, u);
			for (int d = 0; d < n; d++)
			{
				const T c1 = _census1[u];
				const T c2 = _census2[u - d];
				_MC[d] = static_cast<ushort>(HammingDistance(c1, c2));
			}
		}
	}
}

static inline ushort updateCost(const ushort* MC, const ushort* Lp, ushort* Lc, int n, ushort minLp, int P1, int P2)
{
	ushort minLc = std::numeric_limits<ushort>::max();

#ifdef WITH_SSE

	const __m128i _minLp = _mm_set1_epi16(minLp);
	const __m128i _P1 = _mm_set1_epi16(P1);
	const __m128i _P2 = _mm_set1_epi16(P2);
	const __m128i _INF = _mm_set1_epi16(-1);
	__m128i _minLc = _mm_set1_epi16(-1);
	for (int d = 0; d < n; d += 8)
	{
		const __m128i _MC = _mm_load_si128((__m128i*)&MC[d]);
		__m128i _Lp0 = _mm_load_si128((__m128i*)&Lp[d]);
		__m128i _Lp1 = d > 0 ? _mm_loadu_si128((__m128i*)&Lp[d - 1]) : _mm_alignr_epi8(_Lp0, _INF, 14);
		__m128i _Lp2 = d < n - 8 ? _mm_loadu_si128((__m128i*)&Lp[d + 1]) : _mm_alignr_epi8(_INF, _Lp0, 2);
		__m128i _Lp3 = _mm_adds_epu16(_minLp, _P2);
		_Lp0 = _mm_min_epu16(_Lp0, _mm_adds_epu16(_Lp1, _P1));
		_Lp1 = _mm_min_epu16(_Lp3, _mm_adds_epu16(_Lp2, _P1));
		_Lp0 = _mm_min_epu16(_Lp0, _Lp1);
		const __m128i _Lc = _mm_subs_epu16(_mm_adds_epu16(_MC, _Lp0), _minLp);
		_mm_store_si128((__m128i*)&Lc[d], _Lc);
		_minLc = _mm_min_epu16(_minLc, _Lc);
	}

	minLc = _mm_hmin_epu16(_minLc);

#else

	for (int d = 0; d < n; d++)
	{
		const int Lp0 = Lp[d];
		const int Lp1 = d > 0 ? Lp[d - 1] + P1 : 0xFFFF;
		const int Lp2 = d < n - 1 ? Lp[d + 1] + P1 : 0xFFFF;
		const int Lp3 = minLp + P2;
		Lc[d] = static_cast<ushort>(MC[d] + min4(Lp0, Lp1, Lp2, Lp3) - minLp);
		minLc = std::min(minLc, Lc[d]);
	}

#endif

	return minLc;
}

static inline ushort updateCost(const ushort* MC, ushort* Lc, int n)
{
	ushort minLc = std::numeric_limits<ushort>::max();

#ifdef WITH_SSE

	__m128i _minLc = _mm_set1_epi16(-1);
	for (int d = 0; d < n; d += 8)
	{
		const __m128i _Lc = _mm_load_si128((__m128i*)&MC[d]);
		_mm_store_si128((__m128i*)&Lc[d], _Lc);
		_minLc = _mm_min_epu16(_minLc, _Lc);
	}

	minLc = _mm_hmin_epu16(_minLc);

#else

	for (int d = 0; d < n; d++)
	{
		Lc[d] = MC[d];
		minLc = std::min(minLc, Lc[d]);
	}

#endif

	return minLc;
}

static void scanCost(const cv::Mat1w& MC, cv::Mat1w& L, cv::Mat1w& minL, int P1, int P2, int ru, int rv)
{
	const int h = MC.size[0];
	const int w = MC.size[1];
	const int n = MC.size[2];

	const bool forward = rv > 0 || (rv == 0 && ru > 0);
	int u0 = 0, u1 = w, du = 1, v0 = 0, v1 = h, dv = 1;
	if (!forward)
	{
		u0 = w - 1; u1 = -1; du = -1;
		v0 = h - 1; v1 = -1; dv = -1;
	}

	for (int vc = v0; vc != v1; vc += dv)
	{
		const int vp = vc - rv;
		ushort* _minLc = minL.ptr<ushort>(vc);
		ushort* _minLp = (ushort*)(minL.data + minL.step.p[0] * vp); // for CV_DbgAssert avoidance
		for (int uc = u0; uc != u1; uc += du)
		{
			const int up = uc - ru;
			const ushort* _MC = MC.ptr<ushort>(vc, uc);
			ushort* _Lc = L.ptr<ushort>(vc, uc);
			ushort* _Lp = (ushort*)(L.data + vp * L.step.p[0] + up * L.step.p[1]); // for CV_DbgAssert avoidance

			const bool inside = vp >= 0 && vp < h && up >= 0 && up < w;
			const ushort minLc = inside ? updateCost(_MC, _Lp, _Lc, n, _minLp[up], P1, P2) : updateCost(_MC, _Lc, n);
			_minLc[uc] = minLc;
		}
	}
}

static int winnerTakesAll(const ushort* L0, const ushort* L1, const ushort* L2, const ushort* L3,
	const ushort* L4, const ushort* L5, const ushort* L6, const ushort* L7, ushort* S, int n)
{
	int minS = std::numeric_limits<int>::max();
	int disp = 0;

#ifdef WITH_SSE

	for (int d = 0; d < n; d += 8)
	{
		__m128i _L0 = _mm_load_si128((__m128i*)&L0[d]);
		__m128i _L1 = _mm_load_si128((__m128i*)&L1[d]);
		__m128i _L2 = _mm_load_si128((__m128i*)&L2[d]);
		__m128i _L3 = _mm_load_si128((__m128i*)&L3[d]);
		__m128i _L4 = _mm_load_si128((__m128i*)&L4[d]);
		__m128i _L5 = _mm_load_si128((__m128i*)&L5[d]);
		__m128i _L6 = _mm_load_si128((__m128i*)&L6[d]);
		__m128i _L7 = _mm_load_si128((__m128i*)&L7[d]);

		_L0 = _mm_adds_epu16(_L0, _L1);
		_L2 = _mm_adds_epu16(_L2, _L3);
		_L4 = _mm_adds_epu16(_L4, _L5);
		_L6 = _mm_adds_epu16(_L6, _L7);

		_L0 = _mm_adds_epu16(_L0, _L2);
		_L4 = _mm_adds_epu16(_L4, _L6);

		const __m128i _S = _mm_adds_epu16(_L0, _L4);
		const __m128i _minS = _mm_minpos_epu16(_S);
		_mm_store_si128((__m128i*)&S[d], _S);
		const int S = _mm_extract_epi16(_minS, 0);
		if (S < minS)
		{
			minS = S;
			disp = _mm_extract_epi16(_minS, 1) + d;
		}
	}

#else

	for (int d = 0; d < n; d++)
	{
		S[d] = L0[d] + L1[d] + L2[d] + L3[d] + L4[d] + L5[d] + L6[d] + L7[d];
		if (S[d] < minS)
		{
			minS = S[d];
			disp = d;
		}
	}

#endif

	return disp;
}

static void calcDisparity(const std::vector<cv::Mat1w>& L, cv::Mat& D1, cv::Mat& D2, cv::Mat1w& S, int DISP_SCALE)
{
	const int h = S.size[0];
	const int w = S.size[1];
	const int n = S.size[2];
	S = 0;

	int v;
#pragma OMP_PARALLEL_FOR
	for (v = 0; v < h; v++)
	{
		ushort* _D1 = D1.ptr<ushort>(v);
		ushort* _D2 = D2.ptr<ushort>(v);
		for (int u = 0; u < w; u++)
		{
			ushort* _S = S.ptr<ushort>(v, u);
			const ushort* _L0 = L[0].ptr<ushort>(v, u);
			const ushort* _L1 = L[1].ptr<ushort>(v, u);
			const ushort* _L2 = L[2].ptr<ushort>(v, u);
			const ushort* _L3 = L[3].ptr<ushort>(v, u);
			const ushort* _L4 = L[4].ptr<ushort>(v, u);
			const ushort* _L5 = L[5].ptr<ushort>(v, u);
			const ushort* _L6 = L[6].ptr<ushort>(v, u);
			const ushort* _L7 = L[7].ptr<ushort>(v, u);

			int disp = winnerTakesAll(_L0, _L1, _L2, _L3, _L4, _L5, _L6, _L7, _S, n);

			// sub-pixel interpolation 
			if (disp > 0 && disp < n - 1)
			{
				const int numer = _S[disp - 1] - _S[disp + 1];
				const int denom = _S[disp - 1] - 2 * _S[disp] + _S[disp + 1];
				disp = disp * DISP_SCALE + (DISP_SCALE * numer + denom) / (2 * denom);
			}
			else
			{
				disp *= DISP_SCALE;
			}

			_D1[u] = static_cast<ushort>(disp);
		}

		// calculate right disparity
		for (int u = 0; u < w; u++)
		{
			int minS = std::numeric_limits<int>::max();
			int disp = 0;
			for (int d = 0; d < n && u + d < w; d++)
			{
				const ushort _S = S(v, u + d, d);
				if (_S < minS)
				{
					minS = _S;
					disp = d;
				}
			}
			_D2[u] = static_cast<ushort>(DISP_SCALE * disp);
		}
	}
}

static void LRConsistencyCheck(cv::Mat& D1, cv::Mat& D2, int max12Diff, int DISP_SHIFT, int DISP_INV)
{
	const int h = D1.rows;
	const int w = D1.cols;
	int v;
#pragma OMP_PARALLEL_FOR
	for (v = 0; v < h; v++)
	{
		ushort* _D1 = D1.ptr<ushort>(v);
		ushort* _D2 = D2.ptr<ushort>(v);
		for (int u = 0; u < w; u++)
		{
			const int d = _D1[u] >> DISP_SHIFT;
			if (u - d >= 0 && std::abs(_D1[u] - _D2[u - d]) > max12Diff)
				_D1[u] = DISP_INV;
		}
	}
}

SemiGlobalMatching::SemiGlobalMatching(const Parameters & param) : param_(param)
{
}

void SemiGlobalMatching::compute(const cv::Mat& I1, const cv::Mat& I2, cv::Mat& D1, cv::Mat& D2)
{
	CV_Assert(I1.type() == CV_8U && I2.type() == CV_8U);
	CV_Assert(I1.size() == I2.size());
	CV_Assert(param_.numDisparities % 16 == 0);

	const int h = I1.rows;
	const int w = I1.cols;
	const int n = param_.numDisparities;
	const int dims[3] = { h, w, n };

	const int NUM_DIRECTIONS = 8;
	const int ru[NUM_DIRECTIONS] = { 1, 1, 0, -1, -1, -1, 0, 1 };
	const int rv[NUM_DIRECTIONS] = { 0, 1, 1, 1, 0, -1, -1, -1 };

	Timer t;

	MC.create(3, dims);
	if (param_.censusType == CENSUS_9x7)
	{
		t.start("census");

		census64[0].create(h, w);
		census64[1].create(h, w);
		census9x7(I1, census64[0]);
		census9x7(I2, census64[1]);

		t.start("matching cost");

		calcMatchingCost(census64[0], census64[1], MC, n);
	}
	else if (param_.censusType == SYMMETRIC_CENSUS_9x7)
	{
		t.start("census");

		census32[0].create(h, w);
		census32[1].create(h, w);
		symmetricCensus9x7(I1, census32[0]);
		symmetricCensus9x7(I2, census32[1]);

		t.start("matching cost");

		calcMatchingCost(census32[0], census32[1], MC, n);
	}
	else
	{
		CV_Error(cv::Error::StsInternal, "No such mode");
	}

	t.start("scan cost");

	L.resize(NUM_DIRECTIONS);
	minL.resize(NUM_DIRECTIONS);

	std::vector<std::thread> threads(NUM_DIRECTIONS);
	for (int dir = 0; dir < NUM_DIRECTIONS; dir++)
	{
		L[dir].create(3, dims);
		minL[dir].create(3, dims);
		threads[dir] = std::thread(scanCost, std::cref(MC), std::ref(L[dir]), std::ref(minL[dir]),
			param_.P1, param_.P2, ru[dir], rv[dir]);
	}
	for (int dir = 0; dir < NUM_DIRECTIONS; dir++)
		threads[dir].join();

	t.start("winner takes all");

	D1.create(h, w, CV_16U);
	D2.create(h, w, CV_16U);
	S.create(3, dims);
	calcDisparity(L, D1, D2, S, DISP_SCALE);

	t.start("median");

	if (param_.medianKernelSize > 0)
	{
		cv::medianBlur(D1, D1, param_.medianKernelSize);
		cv::medianBlur(D2, D2, param_.medianKernelSize);
	}

	t.start("LR consistensy check");

	const int max12Diff = param_.max12Diff << DISP_SHIFT;
	if (max12Diff >= 0)
	{
		LRConsistencyCheck(D1, D2, max12Diff, DISP_SHIFT, DISP_INV);
	}

	t.plot();
}
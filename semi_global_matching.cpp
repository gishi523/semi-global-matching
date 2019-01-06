#include "semi_global_matching.h"

#include <opencv2/imgproc.hpp>

#define USE_OPENMP
#if defined(_OPENMP) && defined(USE_OPENMP)
#ifdef _WIN32
#define OMP_PARALLEL_FOR __pragma(omp parallel for)
#else
#define OMP_PARALLEL_FOR _Pragma("omp parallel for")
#endif
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

static inline int _mm_hmin_epu8(__m128i v)
{
	v = _mm_min_epu8(v, _mm_shuffle_epi32(v, _MM_SHUFFLE(3, 2, 3, 2)));
	v = _mm_min_epu8(v, _mm_shuffle_epi32(v, _MM_SHUFFLE(1, 1, 1, 1)));
	v = _mm_min_epu8(v, _mm_shufflelo_epi16(v, _MM_SHUFFLE(1, 1, 1, 1)));
	v = _mm_min_epu8(v, _mm_srli_epi16(v, 8));
	return static_cast<uchar>(_mm_cvtsi128_si32(v));
}

static inline __m128i _mm_set1_epi_(uint32_t v) { return _mm_set1_epi32(v); }
static inline __m128i _mm_set1_epi_(uint64_t v) { return _mm_set1_epi64x(v); }

#endif

static const int DEFAULT_MC = 64;

static inline int HammingDistance(uint64_t c1, uint64_t c2) { return static_cast<int>(popcnt64(c1 ^ c2)); }

static inline int HammingDistance(uint32_t c1, uint32_t c2) { return static_cast<int>(popcnt32(c1 ^ c2)); }

static inline int min4(int x, int y, int z, int w)
{
	return std::min(std::min(x, y), std::min(z, w));
};

template <int VIEW = 0>
static void census9x7(const cv::Mat& src, cv::Mat& dst)
{
	CV_Assert(dst.elemSize() == 8);

	memset(dst.data, 0, dst.rows * dst.cols * sizeof(uint64_t));

	const int RADIUS_U = 9 / 2;
	const int RADIUS_V = 7 / 2;
	int v;
OMP_PARALLEL_FOR
	for (v = RADIUS_V; v < src.rows - RADIUS_V; v++)
	{
		uint64_t* dstptr = dst.ptr<uint64>(v);
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
			if (VIEW == 0)
				dstptr[u] = c;
			else
				dstptr[src.cols - 1 - u] = c;
		}
	}
}

template <int VIEW = 0>
static void symmetricCensus9x7(const cv::Mat& src, cv::Mat& dst)
{
	CV_Assert(dst.elemSize() == 4);

	memset(dst.data, 0, dst.rows * dst.cols * sizeof(uint32_t));

	const int RADIUS_U = 9 / 2;
	const int RADIUS_V = 7 / 2;

	int v;
OMP_PARALLEL_FOR
	for (v = RADIUS_V; v < src.rows - RADIUS_V; v++)
	{
		uint32_t* dstptr = dst.ptr<uint32_t>(v);
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
			if (VIEW == 0)
				dstptr[u] = c;
			else
				dstptr[src.cols - 1 - u] = c;
		}
	}
}

#ifdef WITH_SSE
static inline void calcMatchingCost16(__m128i _census1, const uint32_t* census2, uchar* MC)
{
	for (int i = 0; i < 16; i += 4)
	{
		__m128i _census2 = _mm_loadu_si128((__m128i*)&census2[i]);
		__m128i _diff = _mm_xor_si128(_census1, _census2);
		MC[i + 0] = static_cast<uchar>(popcnt32(_mm_extract_epi32(_diff, 0)));
		MC[i + 1] = static_cast<uchar>(popcnt32(_mm_extract_epi32(_diff, 1)));
		MC[i + 2] = static_cast<uchar>(popcnt32(_mm_extract_epi32(_diff, 2)));
		MC[i + 3] = static_cast<uchar>(popcnt32(_mm_extract_epi32(_diff, 3)));
	}
}

static inline void calcMatchingCost16(__m128i _census1, const uint64_t* census2, uchar* MC)
{
	for (int i = 0; i < 16; i += 2)
	{
		__m128i _census2 = _mm_loadu_si128((__m128i*)&census2[i]);
		__m128i _diff = _mm_xor_si128(_census1, _census2);
		MC[i + 0] = static_cast<uchar>(popcnt64(_mm_extract_epi64(_diff, 0)));
		MC[i + 1] = static_cast<uchar>(popcnt64(_mm_extract_epi64(_diff, 1)));
	}
}
#endif

template <typename T>
static inline void updateCost(T census1, const T* census2, const uchar* Lp, uchar* Lc, int u, int n, int P1, int P2)
{

#ifdef WITH_SSE

	__m128i _minLp = _mm_set1_epi8(-1);
	for (int d = 0; d < n; d += 16)
	{
		__m128i _Lp = _mm_load_si128((__m128i*)&Lp[d]);
		_minLp = _mm_min_epu8(_minLp, _Lp);
	}

	const uchar minLp = _mm_hmin_epu8(_minLp);
	P1 -= minLp;

	const __m128i _census1 = _mm_set1_epi_(census1);
	const __m128i _P1 = _mm_set1_epi8(P1);
	const __m128i _P2 = _mm_set1_epi8(P2);
	const __m128i _INF = _mm_set1_epi8(255 - P1);
	_minLp = _mm_set1_epi8(minLp);

	alignas(16) uchar MC[16];
	for (int d = 0; d < n; d += 16)
	{
		if (u >= n - 1)
		{
			calcMatchingCost16(_census1, census2 + d, MC);
		}
		else
		{
			for (int i = 0; i < 16; i++)
				MC[i] = u - (d + i) >= 0 ? HammingDistance(census1, census2[d + i]) : DEFAULT_MC;
		}

		const __m128i _MC = _mm_load_si128((__m128i*)MC);

		__m128i _Lp0 = _mm_load_si128((__m128i*)&Lp[d]);
		__m128i _Lp1 = d > 0 ? _mm_loadu_si128((__m128i*)&Lp[d - 1]) : _mm_alignr_epi8(_Lp0, _INF, 15);
		__m128i _Lp2 = d < n - 16 ? _mm_loadu_si128((__m128i*)&Lp[d + 1]) : _mm_alignr_epi8(_INF, _Lp0, 1);

		_Lp0 = _mm_sub_epi8(_Lp0, _minLp);
		_Lp1 = _mm_add_epi8(_Lp1, _P1);
		_Lp2 = _mm_add_epi8(_Lp2, _P1);

		_Lp0 = _mm_min_epu8(_Lp0, _P2);
		_Lp1 = _mm_min_epu8(_Lp1, _Lp2);

		_Lp0 = _mm_min_epu8(_Lp0, _Lp1);

		const __m128i _Lc = _mm_adds_epu8(_MC, _Lp0);
		_mm_store_si128((__m128i*)&Lc[d], _Lc);
	}

#else

	uchar minLp = std::numeric_limits<uchar>::max();
	for (int d = 0; d < n; d++)
		minLp = std::min(minLp, Lp[d]);

	uchar _P1 = P1 - minLp;
	for (int d = 0; d < n; d++)
	{
		const uchar MC = u - d >= 0 ? HammingDistance(census1, census2[d]) : DEFAULT_MC;
		const uchar Lp0 = Lp[d] - minLp;
		const uchar Lp1 = d > 0 ? Lp[d - 1] + _P1 : 0xFF;
		const uchar Lp2 = d < n - 1 ? Lp[d + 1] + _P1 : 0xFF;
		const uchar Lp3 = P2;
		Lc[d] = static_cast<uchar>(MC + min4(Lp0, Lp1, Lp2, Lp3));
	}

#endif
}

template <typename T>
static inline void updateCost(T census1, const T* census2, uchar* Lc, int u, int n)
{
	for (int d = 0; d < n; d++)
	{
		const int MC = u - d >= 0 ? HammingDistance(census1, census2[d]) : DEFAULT_MC;
		Lc[d] = MC;
	}
}

template <typename T>
static void scanCost(const cv::Mat& C1, const cv::Mat& C2, cv::Mat1b& L, int P1, int P2, int ru, int rv)
{
	const int h = L.size[0];
	const int w = L.size[1];
	const int n = L.size[2];

	const bool forward = rv > 0 || (rv == 0 && ru > 0);
	int u0 = 0, u1 = w, du = 1, v0 = 0, v1 = h, dv = 1;
	if (!forward)
	{
		u0 = w - 1; u1 = -1; du = -1;
		v0 = h - 1; v1 = -1; dv = -1;
	}

	for (int vc = v0; vc != v1; vc += dv)
	{
		const T* _census1 = C1.template ptr<T>(vc);
		const T* _census2 = C2.template ptr<T>(vc) + w - 1;
		for (int uc = u0; uc != u1; uc += du)
		{
			const int vp = vc - rv;
			const int up = uc - ru;
			const bool inside = vp >= 0 && vp < h && up >= 0 && up < w;

			uchar* _Lc = L.ptr<uchar>(vc, uc);
			uchar* _Lp = (uchar*)(L.data + vp * L.step.p[0] + up * L.step.p[1]); // for CV_DbgAssert avoidance

			if (inside)
				updateCost(_census1[uc], _census2 - uc, _Lp, _Lc, uc, n, P1, P2);
			else
				updateCost(_census1[uc], _census2 - uc, _Lc, uc, n);
		}
	}
}

static inline int winnerTakesAll(const uchar* L0, const uchar* L1, const uchar* L2, const uchar* L3,
	const uchar* L4, const uchar* L5, const uchar* L6, const uchar* L7, ushort* S, int n)
{
	int minS = std::numeric_limits<int>::max();
	int disp = 0;

#ifdef WITH_SSE

	__m128i _zero = _mm_setzero_si128();
	for (int d = 0; d < n; d += 16)
	{
		__m128i _L0 = _mm_load_si128((__m128i*)&L0[d]);
		__m128i _L1 = _mm_load_si128((__m128i*)&L1[d]);
		__m128i _L2 = _mm_load_si128((__m128i*)&L2[d]);
		__m128i _L3 = _mm_load_si128((__m128i*)&L3[d]);
		__m128i _L4 = _mm_load_si128((__m128i*)&L4[d]);
		__m128i _L5 = _mm_load_si128((__m128i*)&L5[d]);
		__m128i _L6 = _mm_load_si128((__m128i*)&L6[d]);
		__m128i _L7 = _mm_load_si128((__m128i*)&L7[d]);

		// sign extension
		__m128i _L0_0 = _mm_unpacklo_epi8(_L0, _zero);
		__m128i _L0_1 = _mm_unpackhi_epi8(_L0, _zero);
		__m128i _L1_0 = _mm_unpacklo_epi8(_L1, _zero);
		__m128i _L1_1 = _mm_unpackhi_epi8(_L1, _zero);
		__m128i _L2_0 = _mm_unpacklo_epi8(_L2, _zero);
		__m128i _L2_1 = _mm_unpackhi_epi8(_L2, _zero);
		__m128i _L3_0 = _mm_unpacklo_epi8(_L3, _zero);
		__m128i _L3_1 = _mm_unpackhi_epi8(_L3, _zero);
		__m128i _L4_0 = _mm_unpacklo_epi8(_L4, _zero);
		__m128i _L4_1 = _mm_unpackhi_epi8(_L4, _zero);
		__m128i _L5_0 = _mm_unpacklo_epi8(_L5, _zero);
		__m128i _L5_1 = _mm_unpackhi_epi8(_L5, _zero);
		__m128i _L6_0 = _mm_unpacklo_epi8(_L6, _zero);
		__m128i _L6_1 = _mm_unpackhi_epi8(_L6, _zero);
		__m128i _L7_0 = _mm_unpacklo_epi8(_L7, _zero);
		__m128i _L7_1 = _mm_unpackhi_epi8(_L7, _zero);

		// add costs
		_L0_0 = _mm_adds_epu16(_L0_0, _L1_0);
		_L0_1 = _mm_adds_epu16(_L0_1, _L1_1);
		_L2_0 = _mm_adds_epu16(_L2_0, _L3_0);
		_L2_1 = _mm_adds_epu16(_L2_1, _L3_1);
		_L4_0 = _mm_adds_epu16(_L4_0, _L5_0);
		_L4_1 = _mm_adds_epu16(_L4_1, _L5_1);
		_L6_0 = _mm_adds_epu16(_L6_0, _L7_0);
		_L6_1 = _mm_adds_epu16(_L6_1, _L7_1);

		_L0_0 = _mm_adds_epu16(_L0_0, _L2_0);
		_L0_1 = _mm_adds_epu16(_L0_1, _L2_1);
		_L4_0 = _mm_adds_epu16(_L4_0, _L6_0);
		_L4_1 = _mm_adds_epu16(_L4_1, _L6_1);

		const __m128i _S_0 = _mm_adds_epu16(_L0_0, _L4_0);
		const __m128i _S_1 = _mm_adds_epu16(_L0_1, _L4_1);

		const __m128i _minS_0 = _mm_minpos_epu16(_S_0);
		const __m128i _minS_1 = _mm_minpos_epu16(_S_1);
		_mm_store_si128((__m128i*)&S[d + 0], _S_0);
		_mm_store_si128((__m128i*)&S[d + 8], _S_1);

		const int S_0 = _mm_extract_epi16(_minS_0, 0);
		const int S_1 = _mm_extract_epi16(_minS_1, 0);

		if (S_0 < minS)
		{
			minS = S_0;
			disp = _mm_extract_epi16(_minS_0, 1) + d + 0;
		}
		if (S_1 < minS)
		{
			minS = S_1;
			disp = _mm_extract_epi16(_minS_1, 1) + d + 8;
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

static void calcDisparity(const std::vector<cv::Mat1b>& L, cv::Mat1w& S, cv::Mat& D1, cv::Mat& D2)
{
	const int DISP_SCALE = SemiGlobalMatching::DISP_SCALE;

	const int h = S.size[0];
	const int w = S.size[1];
	const int n = S.size[2];

	int v;
OMP_PARALLEL_FOR
	for (v = 0; v < h; v++)
	{
		short* _D1 = D1.ptr<short>(v);
		short* _D2 = D2.ptr<short>(v);
		for (int u = 0; u < w; u++)
		{
			ushort* _S = S.ptr<ushort>(v, u);
			const uchar* _L0 = L[0].ptr<uchar>(v, u);
			const uchar* _L1 = L[1].ptr<uchar>(v, u);
			const uchar* _L2 = L[2].ptr<uchar>(v, u);
			const uchar* _L3 = L[3].ptr<uchar>(v, u);
			const uchar* _L4 = L[4].ptr<uchar>(v, u);
			const uchar* _L5 = L[5].ptr<uchar>(v, u);
			const uchar* _L6 = L[6].ptr<uchar>(v, u);
			const uchar* _L7 = L[7].ptr<uchar>(v, u);

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

			_D1[u] = static_cast<short>(disp);
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
			_D2[u] = static_cast<short>(DISP_SCALE * disp);
		}
	}
}

static void LRConsistencyCheck(cv::Mat& D1, cv::Mat& D2, int max12Diff)
{
	const int DISP_SHIFT = SemiGlobalMatching::DISP_SHIFT;
	const int DISP_INV = SemiGlobalMatching::DISP_INV;

	const int h = D1.rows;
	const int w = D1.cols;
	int v;
OMP_PARALLEL_FOR
	for (v = 0; v < h; v++)
	{
		short* _D1 = D1.ptr<short>(v);
		short* _D2 = D2.ptr<short>(v);
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

	const int MAX_DIRECTIONS = 8;
	const int ru[MAX_DIRECTIONS] = { +1, -1, +0, +0, +1, -1, -1, +1 };
	const int rv[MAX_DIRECTIONS] = { +0, +0, +1, -1, +1, +1, -1, -1 };
	L.resize(MAX_DIRECTIONS);

	if (param_.censusType == CENSUS_9x7)
	{
		census[0].create(h, w, CV_64F);
		census[1].create(h, w, CV_64F);
		census9x7<0>(I1, census[0]);
		census9x7<1>(I2, census[1]);

		int dir;
OMP_PARALLEL_FOR
		for (dir = 0; dir < MAX_DIRECTIONS; dir++)
		{
			L[dir].create(3, dims);
			scanCost<uint64_t>(census[0], census[1], L[dir], param_.P1, param_.P2, ru[dir], rv[dir]);
		}
	}
	else if (param_.censusType == SYMMETRIC_CENSUS_9x7)
	{
		census[0].create(h, w, CV_32S);
		census[1].create(h, w, CV_32S);
		symmetricCensus9x7<0>(I1, census[0]);
		symmetricCensus9x7<1>(I2, census[1]);

		int dir;
OMP_PARALLEL_FOR
		for (dir = 0; dir < MAX_DIRECTIONS; dir++)
		{
			L[dir].create(3, dims);
			scanCost<uint32_t>(census[0], census[1], L[dir], param_.P1, param_.P2, ru[dir], rv[dir]);
		}
	}
	else
	{
		CV_Error(cv::Error::StsInternal, "No such mode");
	}

	S.create(3, dims);
	D1.create(h, w, CV_16S);
	D2.create(h, w, CV_16S);
	calcDisparity(L, S, D1, D2);

	if (param_.medianKernelSize > 0)
	{
		cv::medianBlur(D1, D1, param_.medianKernelSize);
		cv::medianBlur(D2, D2, param_.medianKernelSize);
	}

	const int max12Diff = param_.max12Diff << DISP_SHIFT;
	if (max12Diff >= 0)
	{
		LRConsistencyCheck(D1, D2, max12Diff);
	}
}

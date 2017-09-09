#include "semi_global_matching.h"

#ifdef _WIN32
#define popcnt64 __popcnt64
#else
#define popcnt64 __builtin_popcountll
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

#define USE_SSE
#ifdef USE_SSE
#include <smmintrin.h>
static inline int _mm_hmin_epu16(__m128i v)
{
	const __m128i minpos = _mm_minpos_epu16(v);
	return _mm_extract_epi16(minpos, 0);
}
#endif

static inline int min4(int x, int y, int z, int w)
{
	return std::min(std::min(x, y), std::min(z, w));
};

static void census9x7(const uchar* src, uint64_t* dst, int h, int w)
{
	const int RADIUS_U = 9 / 2;
	const int RADIUS_V = 7 / 2;
	int v;
#pragma omp parallel for
	for (v = RADIUS_V; v < h - RADIUS_V; v++)
	{
		for (int u = RADIUS_U; u < w - RADIUS_U; u++)
		{
			uint64_t c = 0;
			for (int dv = -RADIUS_V; dv <= RADIUS_V; dv++)
			{
				for (int du = -RADIUS_U; du <= RADIUS_U; du++)
				{
					c <<= 1;
					c += src[v * w + u] <= src[(v + dv) * w + (u + du)] ? 0 : 1;
				}
			}
			dst[v * w + u] = c;
		}
	}
}

static void symmetricCensus9x7(const uchar* src, uint64_t* dst, int h, int w)
{
	const int RADIUS_U = 9 / 2;
	const int RADIUS_V = 7 / 2;
	int v;
#pragma omp parallel for
	for (v = RADIUS_V; v < h - RADIUS_V; v++)
	{
		for (int u = RADIUS_U; u < w - RADIUS_U; u++)
		{
			uint64_t c = 0;
			for (int dv = -RADIUS_V; dv <= -1; dv++)
			{
				for (int du = -RADIUS_U; du <= RADIUS_U; du++)
				{
					const int v1 = v + dv;
					const int v2 = v - dv;
					const int u1 = u + du;
					const int u2 = u - du;
					c <<= 1;
					c += src[v1 * w + u1] <= src[v2 * w + u2] ? 0 : 1;
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
					c += src[v1 * w + u1] <= src[v2 * w + u2] ? 0 : 1;
				}
			}

			dst[v * w + u] = c;
		}
	}
}

static inline int hammingDistance(uint64_t c1, uint64_t c2)
{
	return static_cast<int>(popcnt64(c1 ^ c2));
};

static void matchingCost(const uint64_t* census1, const uint64_t* census2, ushort* C, int h, int w, int n)
{
	int v;
#pragma omp parallel for
	for (v = 0; v < h; v++)
	{
		for (int u = 0; u < n; u++)
		{
			for (int d = 0; d <= u; d++)
			{
				const uint64_t c1 = census1[v * w + u];
				const uint64_t c2 = census2[v * w + u - d];
				C[(v * w + u) * n + d] = static_cast<uint16_t>(hammingDistance(c1, c2));
			}
			for (int d = u + 1; d < n; d++)
				C[(v * w + u) * n + d] = 64;
		}
		for (int u = n; u < w; u++)
		{
			for (int d = 0; d < n; d++)
			{
				const uint64_t c1 = census1[v * w + u];
				const uint64_t c2 = census2[v * w + u - d];
				C[(v * w + u) * n + d] = static_cast<uint16_t>(hammingDistance(c1, c2));
			}
		}
	}
}

SemiGlobalMatching::SemiGlobalMatching(const Parameters& param) : param_(param)
{
}

cv::Mat SemiGlobalMatching::compute(const cv::Mat& I1, const cv::Mat& I2)
{
	const int h = I1.rows;
	const int w = I1.cols;
	const int n = param_.numDisparities;

	const int P1 = param_.P1;
	const int P2 = param_.P2;
	const int max12Diff = param_.max12Diff << DISP_SHIFT;

	// census transform
	uint64_t* census1 = (uint64_t*)malloc(sizeof(uint64_t) * h * w);
	uint64_t* census2 = (uint64_t*)malloc(sizeof(uint64_t) * h * w);
	if (param_.censusType == CENSUS_9x7)
	{
		census9x7((uchar*)I1.data, census1, h, w);
		census9x7((uchar*)I2.data, census2, h, w);
	}
	if (param_.censusType == SYMMETRIC_CENSUS_9x7)
	{
		symmetricCensus9x7((uchar*)I1.data, census1, h, w);
		symmetricCensus9x7((uchar*)I2.data, census2, h, w);
	}

	// calculate matching cost
	ushort* Cbuf = (ushort*)malloc(sizeof(ushort) * h * w * n);
	matchingCost(census1, census2, Cbuf, h, w, n);

	// aggregate cost
	ushort* Lbuf = (ushort*)malloc(sizeof(ushort) * h * w * n * 8);
	ushort* minLbuf = (ushort*)malloc(sizeof(ushort) * h * w * 8);

	const int ru[8] = { 1, 1, 0, -1, -1, -1, 0, 1 };
	const int rv[8] = { 0, 1, 1, 1, 0, -1, -1, -1 };

#ifdef USE_SSE
	const __m128i vP1 = _mm_set1_epi16(P1);
	const __m128i vP2 = _mm_set1_epi16(P2);
	const __m128i vINV = _mm_set1_epi16(-1);
#endif

	int k;
#pragma omp parallel for
	for (k = 0; k < 8; k++)
	{
		const int u1 = k < 4 ? 0 : w - 1;
		const int u2 = k < 4 ? w : -1;
		const int v1 = k < 4 ? 0 : h - 1;
		const int v2 = k < 4 ? h : -1;
		const int du = k < 4 ? 1 : -1;
		const int dv = k < 4 ? 1 : -1;

		for (int vc = v1; vc != v2; vc += dv)
		{
			for (int uc = u1; uc != u2; uc += du)
			{
				const int vp = vc - rv[k];
				const int up = uc - ru[k];

				const int pc = vc * w + uc; // current pixel index
				const int pp = vp * w + up; // previous pixel index

				const ushort* C = Cbuf + pc * n;
				ushort* Lc = Lbuf + k * (h * w * n) + pc * n;
				ushort* Lp = Lbuf + k * (h * w * n) + pp * n;
				ushort* minL = minLbuf + k * (h * w);

				int minLc = std::numeric_limits<int>::max();

				// the case where previous pixel is inside the image
				if (vp >= 0 && vp < h && up >= 0 && up < w)
				{
					const int minLp = minL[pp];
#ifdef USE_SSE
					const __m128i vminLp = _mm_set1_epi16(minLp);
					__m128i vminLc = _mm_set1_epi16(-1);
					for (int d = 0; d < n; d += 8)
					{
						const __m128i vC = _mm_load_si128((__m128i*)&C[d]);
						__m128i vLc = _mm_load_si128((__m128i*)&Lp[d]);
						__m128i vLc_m = d > 0 ? _mm_loadu_si128((__m128i*)&Lp[d - 1]) : _mm_alignr_epi8(vLc, vINV, 14);
						__m128i vLc_p = d < n - 8 ? _mm_loadu_si128((__m128i*)&Lp[d + 1]) : _mm_alignr_epi8(vINV, vLc, 2);
						vLc = _mm_min_epu16(vLc, _mm_adds_epu16(vLc_m, vP1));
						vLc = _mm_min_epu16(vLc, _mm_adds_epu16(vLc_p, vP1));
						vLc = _mm_min_epu16(vLc, _mm_adds_epu16(vminLp, vP2));
						vLc = _mm_subs_epu16(_mm_adds_epu16(vC, vLc), vminLp);
						_mm_store_si128((__m128i*)&Lc[d], vLc);
						vminLc = _mm_min_epu16(vminLc, vLc);
					}
					minLc = _mm_hmin_epu16(vminLc);
#else
					for (int d = 0; d < n; d++)
					{
						const int Lp_m_P1 = d > 0 ? Lp[d - 1] + P1 : 0xffff;
						const int Lp_p_P1 = d < n - 1 ? Lp[d + 1] + P1 : 0xffff;
						const int _Lc = C[d] + min4(Lp[d], Lp_m_P1, Lp_p_P1, minLp + P2) - minLp;
						Lc[d] = static_cast<ushort>(_Lc);
						minLc = std::min(minLc, _Lc);
					}
#endif // USE_SSE	
				}
				else
				{
#ifdef USE_SSE
					__m128i vminLc = _mm_set1_epi16(-1);
					for (int d = 0; d < n; d += 8)
					{
						const __m128i vLc = _mm_load_si128((__m128i*)&C[d]);
						_mm_store_si128((__m128i*)&Lc[d], vLc);
						vminLc = _mm_min_epu16(vminLc, vLc);
					}
					minLc = _mm_hmin_epu16(vminLc);
#else
					for (int d = 0; d < n; d++)
					{
						const int _Lc = C[d];
						Lc[d] = static_cast<ushort>(_Lc);
						minLc = std::min(minLc, _Lc);
					}
#endif // USE_SSE
				}

				minL[pc] = static_cast<ushort>(minLc);

			}
		}
	}

	// calculate disparity
	cv::Mat1w disparity1(h, w);
	cv::Mat1w disparity2(h, w);
	ushort* Sbuf = (ushort*)malloc(sizeof(ushort) * h * w * n);
	{
		int vc;
#pragma omp parallel for
		for (vc = 0; vc < h; vc++)
		{
			ushort* D1 = disparity1.ptr<ushort>(vc);
			ushort* D2 = disparity2.ptr<ushort>(vc);
			for (int uc = 0; uc < w; uc++)
			{
				const int pc = vc * w + uc;

				ushort* S = Sbuf + pc * n;
				ushort* L0 = Lbuf + 0 * (h * w * n) + pc * n;
				ushort* L1 = Lbuf + 1 * (h * w * n) + pc * n;
				ushort* L2 = Lbuf + 2 * (h * w * n) + pc * n;
				ushort* L3 = Lbuf + 3 * (h * w * n) + pc * n;
				ushort* L4 = Lbuf + 4 * (h * w * n) + pc * n;
				ushort* L5 = Lbuf + 5 * (h * w * n) + pc * n;
				ushort* L6 = Lbuf + 6 * (h * w * n) + pc * n;
				ushort* L7 = Lbuf + 7 * (h * w * n) + pc * n;

				// winner takes all
				int minS = std::numeric_limits<int>::max();
				int disp = 0;
#ifdef USE_SSE
				for (int d = 0; d < n; d += 8)
				{
					const __m128i vL0 = _mm_load_si128((__m128i*)&L0[d]);
					const __m128i vL1 = _mm_load_si128((__m128i*)&L1[d]);
					const __m128i vL2 = _mm_load_si128((__m128i*)&L2[d]);
					const __m128i vL3 = _mm_load_si128((__m128i*)&L3[d]);
					const __m128i vL4 = _mm_load_si128((__m128i*)&L4[d]);
					const __m128i vL5 = _mm_load_si128((__m128i*)&L5[d]);
					const __m128i vL6 = _mm_load_si128((__m128i*)&L6[d]);
					const __m128i vL7 = _mm_load_si128((__m128i*)&L7[d]);
					const __m128i vS = _mm_adds_epu16(
						_mm_adds_epu16(_mm_adds_epu16(vL0, vL1), _mm_adds_epu16(vL2, vL3)),
						_mm_adds_epu16(_mm_adds_epu16(vL4, vL5), _mm_adds_epu16(vL6, vL7)));
					const __m128i vminS = _mm_minpos_epu16(vS);
					_mm_store_si128((__m128i*)&S[d], vS);
					const int _minS = _mm_extract_epi16(vminS, 0);
					const int _disp = _mm_extract_epi16(vminS, 1) + d;
					if (_minS < minS)
					{
						minS = _minS;
						disp = _disp;
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

				// sub-pixel interpolation 
				if (disp > 0 && disp < n - 1)
				{
					const int numer = S[disp - 1] - S[disp + 1];
					const int denom = S[disp - 1] - 2 * S[disp] + S[disp + 1];
					disp = disp * DISP_SCALE + (DISP_SCALE * numer + denom) / (2 * denom);
				}
				else
				{
					disp *= DISP_SCALE;
				}

				D1[uc] = static_cast<ushort>(disp);
			}

			// calculate right disparity
			for (int uc = 0; uc < w; uc++)
			{
				int minS = std::numeric_limits<int>::max();
				int disp = 0;
				for (int d = 0; d < n && uc + d < w; d++)
				{
					ushort* S = Sbuf + (vc * w + uc + d) * n;
					if (S[d] < minS)
					{
						minS = S[d];
						disp = d;
					}
				}
				D2[uc] = static_cast<ushort>(DISP_SCALE * disp);
			}
		}
	}

	if (param_.medianKernelSize > 0)
	{
		cv::medianBlur(disparity1, disparity1, param_.medianKernelSize);
		cv::medianBlur(disparity2, disparity2, param_.medianKernelSize);
	}

	// consistency check
	if (max12Diff >= 0)
	{
		int vc;
#pragma omp parallel for
		for (vc = 0; vc < h; vc++)
		{
			ushort* D1 = disparity1.ptr<ushort>(vc);
			ushort* D2 = disparity2.ptr<ushort>(vc);
			for (int uc = 0; uc < w; uc++)
			{
				const int d = D1[uc] >> DISP_SHIFT;
				if (uc - d >= 0 && std::abs(D1[uc] - D2[uc - d]) > max12Diff)
					D1[uc] = DISP_INV;
			}
		}
	}

	free(census1);
	free(census2);
	free(Cbuf);
	free(Sbuf);
	free(Lbuf);
	free(minLbuf);

	return disparity1;
}
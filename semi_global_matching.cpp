#include "semi_global_matching.h"
#include <array>
#ifdef _OPENMP
#include <omp.h>
#endif

#define WITH_AVX2
#ifdef WITH_AVX2
#include <immintrin.h>
#endif

#ifdef _WIN32
#define popcnt64 __popcnt64
#else
#define popcnt64 __builtin_popcountll
#endif

static const int r[8][2] = { { 0, 1 }, { 1, 0 }, { 0, -1 }, { -1, 0 }, { 1, 1 }, { 1, -1 }, { -1, 1 }, { -1, -1 } };

template <typename T>
static void census9x7(const T* src, uint64_t* dst, int w, int h)
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
				C[(v * w + u) * n + d] = static_cast<uint16_t>(popcnt64(c1 ^ c2));
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
				C[(v * w + u) * n + d] = static_cast<uint16_t>(popcnt64(c1 ^ c2));
			}
		}
	}
}

#ifdef WITH_AVX2
static inline uint16_t _mm256_hmin_epu16(__m256i v)
{
	__m128i vmin = _mm_min_epu16(_mm256_extracti128_si256(v, 0), _mm256_extracti128_si256(v, 1));
	vmin = _mm_min_epu16(vmin, _mm_alignr_epi8(vmin, vmin, 2));
	vmin = _mm_min_epu16(vmin, _mm_alignr_epi8(vmin, vmin, 4));
	vmin = _mm_min_epu16(vmin, _mm_alignr_epi8(vmin, vmin, 8));
	vmin = _mm_min_epu16(vmin, _mm_alignr_epi8(vmin, vmin, 16));
	return _mm_extract_epi16(vmin, 0);
}
static inline __m256i _mm256_fsr_16(__m256i a, __m256i b)
{
	return _mm256_alignr_epi8(a, _mm256_permute2x128_si256(b, a, 0x21), 14);
}
static inline __m256i _mm256_fsl_16(__m256i a, __m256i b)
{
	return _mm256_alignr_epi8(_mm256_permute2x128_si256(a, b, 0x21), a, 2);
}
#endif // WITH_AVX2

static void costAggregation(const ushort* C, ushort* L, ushort* minL, int h, int w, int n, int rv, int ru, int P1, int P2)
{
	const bool forward = rv > 0 || (rv == 0 && ru > 0);
	const int u1 = forward ? 0 : w - 1;
	const int u2 = forward ? w : -1;
	const int du = forward ? 1 : -1;
	const int v1 = forward ? 0 : h - 1;
	const int v2 = forward ? h : -1;
	const int dv = forward ? 1 : -1;

#ifdef WITH_AVX2
	const __m256i VP1 = _mm256_set1_epi16(P1);
	const __m256i VP2 = _mm256_set1_epi16(P2);
	const __m256i VINV = _mm256_set1_epi16(std::numeric_limits<ushort>::max() - P1);
#endif // WITH_AVX2

	for (int vc = v1; vc != v2; vc += dv)
	{
		for (int uc = u1; uc != u2; uc += du)
		{
			const int vp = vc - rv;
			const int up = uc - ru;

			if (vp >= 0 && vp < h && up >= 0 && up < w)
			{
				const ushort minLp = minL[vp * w + up];
				ushort minLc = std::numeric_limits<ushort>::max();
#ifdef WITH_AVX2
				__m256i vminLc = _mm256_set1_epi16(-1);
				const __m256i vminLp = _mm256_set1_epi16(minLp);

				for (int d = 0; d < n; d += 16)
				{
					const __m256i vLp1 = _mm256_load_si256((__m256i*)&L[(vp * w + up) * n + d]);
					const __m256i vLp2 = d > 0 ? _mm256_loadu_si256((__m256i*)&L[(vp * w + up) * n + d - 1]) : _mm256_fsr_16(vLp1, VINV);
					const __m256i vLp3 = d < n - 16 ? _mm256_loadu_si256((__m256i*)&L[(vp * w + up) * n + d + 1]) : _mm256_fsl_16(vLp1, VINV);

					const __m256i vcost1 = _mm256_load_si256((__m256i*)&C[(vc * w + uc) * n + d]);
					const __m256i vcost2 = vLp1;
					const __m256i vcost3 = _mm256_add_epi16(vLp2, VP1);
					const __m256i vcost4 = _mm256_add_epi16(vLp3, VP1);
					const __m256i vcost5 = _mm256_add_epi16(vminLp, VP2);
					const __m256i vcost6 = _mm256_min_epu16(_mm256_min_epu16(vcost2, vcost3), _mm256_min_epu16(vcost4, vcost5));
					const __m256i vLc = _mm256_sub_epi16(_mm256_add_epi16(vcost1, vcost6), vminLp);

					_mm256_store_si256((__m256i*)&L[(vc * w + uc) * n + d], vLc);
					vminLc = _mm256_min_epu16(vminLc, vLc);
				}
				minLc = _mm256_hmin_epu16(vminLc);
#else
				for (int d = 0; d < n; d++)
				{
					const int cost1 = C[(vc * w + uc) * n + d];
					const int cost2 = L[(vp * w + up) * n + d];
					const int cost3 = d - 1 >= 0 ? L[(vp * w + up) * n + d - 1] + P1 : std::numeric_limits<int>::max();
					const int cost4 = d + 1 < n ? L[(vp * w + up) * n + d + 1] + P1 : std::numeric_limits<int>::max();
					const int cost5 = minLp + P2;
					const ushort Lc = static_cast<ushort>(cost1 + std::min(std::min(cost2, cost3), std::min(cost4, cost5)) - minLp);
					L[(vc * w + uc) * n + d] = Lc;
					minLc = std::min(minLc, Lc);
				}
#endif // WITH_AVX2
				minL[vc * w + uc] = minLc;
			}
			else
			{
				ushort minLc = std::numeric_limits<ushort>::max();
#ifdef WITH_AVX2
				__m256i vminLc = _mm256_set1_epi16(-1);
				for (int d = 0; d < n; d += 16)
				{
					const __m256i vLc = _mm256_load_si256((__m256i*)&C[(vc * w + uc) * n + d]);
					_mm256_store_si256((__m256i*)&L[(vc * w + uc) * n + d], vLc);
					vminLc = _mm256_min_epu16(vminLc, vLc);
				}
				minLc = _mm256_hmin_epu16(vminLc);
#else
				for (int d = 0; d < n; d++)
				{
					const ushort Lc = C[(vc * w + uc) * n + d];
					L[(vc * w + uc) * n + d] = Lc;
					minLc = std::min(minLc, Lc);
				}
#endif // WITH_AVX2
				minL[vc * w + uc] = minLc;
			}
		}
	}
}

static void winnerTakesAll(const std::vector<const ushort*>& L, ushort* S, ushort* D, int h, int w, int n, int scale)
{
	if (L.size() == 4)
	{
		int v;
#pragma omp parallel for
		for (v = 0; v < h; v++)
		{
			for (int u = 0; u < w; u++)
			{
				ushort minS = std::numeric_limits<ushort>::max();
				ushort disp = 0;

				ushort* _S = &S[(v * w + u) * n];

#ifdef WITH_AVX2
				for (int d = 0; d < n; d += 16)
				{
					const __m256i vL0 = _mm256_load_si256((__m256i*)&L[0][(v * w + u) * n + d]);
					const __m256i vL1 = _mm256_load_si256((__m256i*)&L[1][(v * w + u) * n + d]);
					const __m256i vL2 = _mm256_load_si256((__m256i*)&L[2][(v * w + u) * n + d]);
					const __m256i vL3 = _mm256_load_si256((__m256i*)&L[3][(v * w + u) * n + d]);

					const __m256i vS = _mm256_add_epi16(_mm256_add_epi16(vL0, vL1), _mm256_add_epi16(vL2, vL3));

					_mm256_store_si256((__m256i*)&_S[d], vS);

					const __m128i vS0 = _mm256_extracti128_si256(vS, 0);
					const __m128i vS1 = _mm256_extracti128_si256(vS, 1);
					const __m128i vminS0 = _mm_minpos_epu16(vS0);
					const __m128i vminS1 = _mm_minpos_epu16(vS1);
					const ushort minS0 = _mm_extract_epi16(vminS0, 0);
					const ushort mind0 = _mm_extract_epi16(vminS0, 1) + d;
					const ushort minS1 = _mm_extract_epi16(vminS1, 0);
					const ushort mind1 = _mm_extract_epi16(vminS1, 1) + d + 8;
					if (minS0 < minS)
					{
						minS = minS0;
						disp = mind0;
					}
					if (minS1 < minS)
					{
						minS = minS1;
						disp = mind1;
					}
				}
#else
				for (int d = 0; d < n; d++)
				{
					_S[d] = L[0][(v * w + u) * n + d] + L[1][(v * w + u) * n + d] + L[2][(v * w + u) * n + d] + L[3][(v * w + u) * n + d];

					if (_S[d] < minS)
					{
						minS = _S[d];
						disp = d;
					}
				}
#endif // WITH_AVX2

				if (disp > 0 && disp < n - 1)
				{
					const int hdenom = _S[disp - 1] - 2 * _S[disp] + _S[disp + 1];
					disp = disp * scale + (scale * (_S[disp - 1] - _S[disp + 1]) + hdenom) / (2 * hdenom);
				}
				else
				{
					disp *= scale;
				}

				D[v * w + u] = disp;
			}
		}
	}
	else
	{
		int v;
#pragma omp parallel for
		for (v = 0; v < h; v++)
		{
			for (int u = 0; u < w; u++)
			{
				ushort minS = std::numeric_limits<ushort>::max();
				ushort disp = 0;

				ushort* _S = &S[(v * w + u) * n];

#ifdef WITH_AVX2
				for (int d = 0; d < n; d += 16)
				{
					const __m256i vL0 = _mm256_load_si256((__m256i*)&L[0][(v * w + u) * n + d]);
					const __m256i vL1 = _mm256_load_si256((__m256i*)&L[1][(v * w + u) * n + d]);
					const __m256i vL2 = _mm256_load_si256((__m256i*)&L[2][(v * w + u) * n + d]);
					const __m256i vL3 = _mm256_load_si256((__m256i*)&L[3][(v * w + u) * n + d]);
					const __m256i vL4 = _mm256_load_si256((__m256i*)&L[4][(v * w + u) * n + d]);
					const __m256i vL5 = _mm256_load_si256((__m256i*)&L[5][(v * w + u) * n + d]);
					const __m256i vL6 = _mm256_load_si256((__m256i*)&L[6][(v * w + u) * n + d]);
					const __m256i vL7 = _mm256_load_si256((__m256i*)&L[7][(v * w + u) * n + d]);

					const __m256i vS = _mm256_add_epi16(
						_mm256_add_epi16(_mm256_add_epi16(vL0, vL1), _mm256_add_epi16(vL2, vL3)),
						_mm256_add_epi16(_mm256_add_epi16(vL4, vL5), _mm256_add_epi16(vL6, vL7)));

					_mm256_store_si256((__m256i*)&S[(v * w + u) * n + d], vS);

					const __m128i vS0 = _mm256_extracti128_si256(vS, 0);
					const __m128i vS1 = _mm256_extracti128_si256(vS, 1);
					const __m128i vminS0 = _mm_minpos_epu16(vS0);
					const __m128i vminS1 = _mm_minpos_epu16(vS1);
					const ushort minS0 = _mm_extract_epi16(vminS0, 0);
					const ushort mind0 = _mm_extract_epi16(vminS0, 1) + d;
					const ushort minS1 = _mm_extract_epi16(vminS1, 0);
					const ushort mind1 = _mm_extract_epi16(vminS1, 1) + d + 8;
					if (minS0 < minS)
					{
						minS = minS0;
						disp = mind0;
					}
					if (minS1 < minS)
					{
						minS = minS1;
						disp = mind1;
					}
				}
#else
				for (int d = 0; d < n; d++)
				{
					_S[d]
						= L[0][(v * w + u) * n + d] + L[1][(v * w + u) * n + d] + L[2][(v * w + u) * n + d] + L[3][(v * w + u) * n + d];
						+ L[4][(v * w + u) * n + d] + L[5][(v * w + u) * n + d] + L[6][(v * w + u) * n + d] + L[7][(v * w + u) * n + d];

					if (_S[d] < minS)
					{
						minS = _S[d];
						disp = d;
					}
				}
#endif // WITH_AVX2

				if (disp > 0 && disp < n - 1)
				{
					const int hdenom = _S[disp - 1] - 2 * _S[disp] + _S[disp + 1];
					disp = disp * scale + (scale * (_S[disp - 1] - _S[disp + 1]) + hdenom) / (2 * hdenom);
				}
				else
				{
					disp *= scale;
				}

				D[v * w + u] = disp;
			}
		}
	}
}

SemiGlobalMatching::SemiGlobalMatching(const Parameters& param) : param_(param)
{
	CV_Assert(param.numPaths == 4 || param.numPaths == 8);
	CV_Assert(param.numDisparities >= 64 && param.numDisparities % 16 == 0);
}

cv::Mat SemiGlobalMatching::compute(const cv::Mat& I1, const cv::Mat& I2)
{
	CV_Assert(I1.size() == I2.size() && I1.type() == I2.type());
	CV_Assert(I1.type() == CV_8U || I1.type() == CV_16U);

	const int w = I1.cols;
	const int h = I1.rows;
	const int n = param_.numDisparities;

	// census transform
	cv::Mat_<uint64_t> census1 = cv::Mat_<uint64_t>::zeros(h, w);
	cv::Mat_<uint64_t> census2 = cv::Mat_<uint64_t>::zeros(h, w);
	if (I1.type() == CV_8U)
	{
		census9x7((uchar*)I1.data, (uint64_t*)census1.data, w, h);
		census9x7((uchar*)I2.data, (uint64_t*)census2.data, w, h);
	}
	else
	{
		census9x7((ushort*)I1.data, (uint64_t*)census1.data, w, h);
		census9x7((ushort*)I2.data, (uint64_t*)census2.data, w, h);
	}

	// calculate matching cost
	cv::Mat1w C(3, std::array<int, 3>{ h, w, n }.data());
	matchingCost((uint64_t*)census1.data, (uint64_t*)census2.data, (ushort*)C.data, h, w, n);

	// aggregate cost
	std::vector<cv::Mat1w> L(param_.numPaths);
	std::vector<cv::Mat1w> minL(param_.numPaths);
	std::vector<const ushort*> Ldata(param_.numPaths);
	int i;
#pragma omp parallel for
	for (i = 0; i < param_.numPaths; i++)
	{
		L[i].create(3, std::array<int, 3>{ h, w, n }.data());
		minL[i].create(h, w);
		costAggregation((ushort*)C.data, (ushort*)L[i].data, (ushort*)minL[i].data, h, w, n, r[i][0], r[i][1], param_.P1, param_.P2);
		Ldata[i] = (ushort*)L[i].data;
	}

	// winner takes all
	cv::Mat1w S(3, std::array<int, 3>{ h, w, n }.data());
	cv::Mat1w D(h, w);
	winnerTakesAll(Ldata, (ushort*)S.data, (ushort*)D.data, h, w, n, DISP_SCALE);

	return D;
}
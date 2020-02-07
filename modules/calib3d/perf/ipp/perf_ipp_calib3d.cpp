#include "../perf_precomp.hpp"

#ifdef HAVE_IPP
#include "opencv2/ts/ipp_perf.hpp"

namespace opencv_test
{
using namespace perf;

/////////// FilterSpeckles //////////////////////

typedef tuple<IppTestThreads, Size, MatType> FilterSpecklesParamType;
typedef TestBaseWithParam<FilterSpecklesParamType> FilterSpecklesFixture;

PERF_TEST_P(FilterSpecklesFixture, ippFilterSpeckles,
            ::testing::Combine(
                IPP_TEST_STHREAD,
                IPP_TEST_SIZES,
                ::testing::Values(CV_8UC1, CV_16SC1)
            ))
{
    const FilterSpecklesParamType params = GetParam();
    const cv::Size size       = get<1>(params);
    const int      type       = get<2>(params);

    cv::Mat src(size, type);
    cv::Mat dst(size, type);
    cv::Mat dstRef(size, type);

    declare.in(src, WARMUP_RNG).out(dst);
    src.copyTo(dst);
    src.copyTo(dstRef);

    // Conformance check
    RUN_IPP(cv::filterSpeckles(dst, 0, 2, 2));
    RUN_REF(cv::filterSpeckles(dstRef, 0, 2, 2));
    VERIFY_DATA(dst, dstRef, 1, 1e-9);

    IPP_TEST_CYCLE() cv::filterSpeckles(dst, 0, 2, 2);
}

} // namespace
#endif // HAVE_IPP

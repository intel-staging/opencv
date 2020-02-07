#include "../perf_precomp.hpp"

#ifdef HAVE_IPP
#include "opencv2/ts/ipp_perf.hpp"
#include "opencv2/core/hal/hal.hpp"

namespace opencv_test
{
using namespace perf;

///////////// HAL ////////////////////////

typedef tuple<IppTestThreads, Size, MatType> HALParamType;
typedef TestBaseWithParam<HALParamType> HALFixture;

PERF_TEST_P(HALFixture, ippInvSqrt32f,
            ::testing::Combine(
                IPP_TEST_STHREAD,
                IPP_TEST_SIZES,
                ::testing::Values(CV_32FC1)
            ))
{
    const HALParamType params = GetParam();
    const cv::Size size = get<1>(params);
    const int      type = get<2>(params);

    cv::Mat src(size, type);
    cv::Mat dst(size, type);
    cv::Mat dstRef(size, type);
    int     len = (int)((src.step*src.rows)/src.elemSize1());

    declare.in(src, WARMUP_RNG).out(dst);
    cv::randu(src, 1, 1000);

    // Conformance check
    RUN_IPP(cv::hal::invSqrt32f(src.ptr<Ipp32f>(), dst.ptr<Ipp32f>(), len));
    RUN_REF(cv::hal::invSqrt32f(src.ptr<Ipp32f>(), dstRef.ptr<Ipp32f>(), len));
    VERIFY_DATA(dst, dstRef, 0, 4e-7);

    IPP_TEST_CYCLE_MULTIRUN(50) cv::hal::invSqrt32f(src.ptr<Ipp32f>(), dst.ptr<Ipp32f>(), len);
}

PERF_TEST_P(HALFixture, ippInvSqrt64f,
            ::testing::Combine(
                IPP_TEST_STHREAD,
                IPP_TEST_SIZES,
                ::testing::Values(CV_64FC1)
            ))
{
    const HALParamType params = GetParam();
    const cv::Size size = get<1>(params);
    const int      type = get<2>(params);

    cv::Mat src(size, type);
    cv::Mat dst(size, type);
    cv::Mat dstRef(size, type);
    int     len = (int)((src.step*src.rows)/src.elemSize1());

    declare.in(src, WARMUP_RNG).out(dst);
    cv::randu(src, 1, 1000);

    // Conformance check
    RUN_IPP(cv::hal::invSqrt64f(src.ptr<Ipp64f>(), dst.ptr<Ipp64f>(), len));
    RUN_REF(cv::hal::invSqrt64f(src.ptr<Ipp64f>(), dstRef.ptr<Ipp64f>(), len));
    VERIFY_DATA_PRES(dst, dstRef, 0, 0, 1e-15);

    IPP_TEST_CYCLE_MULTIRUN(50) cv::hal::invSqrt64f(src.ptr<Ipp64f>(), dst.ptr<Ipp64f>(), len);
}

} // namespace
#endif // HAVE_IPP

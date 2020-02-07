#include "../perf_precomp.hpp"

#ifdef HAVE_IPP
#include "opencv2/ts/ipp_perf.hpp"

namespace opencv_test
{
using namespace perf;

///////////// Multiply ////////////////////////

typedef tuple<IppTestThreads, Size, MatType> MultiplyParamType;
typedef TestBaseWithParam<MultiplyParamType> MultiplyFixture;

PERF_TEST_P(MultiplyFixture, ippMultiply,
            ::testing::Combine(
                IPP_TEST_STHREAD,
                IPP_TEST_SIZES,
                ::testing::Values(CV_8U, CV_16U, CV_16S, CV_32F)
            ))
{
    const MultiplyParamType params = GetParam();
    const cv::Size size = get<1>(params);
    const int      type = get<2>(params);

    cv::Mat src1(size, type);
    cv::Mat src2(size, type);
    cv::Mat dst(size, type);
    cv::Mat dstRef(size, type);

    declare.in(src1, WARMUP_RNG).out(dst);
    cv::randu(src2, 0, 100);

    // Conformance check
    RUN_IPP(cv::multiply(src1, src2, dst));
    RUN_REF(cv::multiply(src1, src2, dstRef));
    VERIFY_DATA(dst, dstRef, 1, 1e-9);

    IPP_TEST_CYCLE_MULTIRUN(10) cv::multiply(src1, src2, dst);
}

PERF_TEST_P(MultiplyFixture, ippMultiply_Const,
            ::testing::Combine(
                IPP_TEST_STHREAD,
                IPP_TEST_SIZES,
                ::testing::Values(CV_8U, CV_16U, CV_16S, CV_32S, CV_32F, CV_64F)
            ))
{
    const MultiplyParamType params = GetParam();
    const cv::Size size = get<1>(params);
    const int      type = get<2>(params);
    const Scalar   s    = 1.333;

    cv::Mat src(size, type);
    cv::Mat dst(size, type);
    cv::Mat dstRef(size, type);

    declare.in(src, WARMUP_RNG).out(dst);

    // Conformance check
    RUN_IPP(cv::multiply(src, s, dst));
    RUN_REF(cv::multiply(src, s, dstRef));
    VERIFY_DATA(dst, dstRef, 1, 1e-9);

    IPP_TEST_CYCLE_MULTIRUN(10) cv::multiply(src, s, dst);
}

# if 0
///////////// Divide ////////////////////////

PERF_TEST_P(MultiplyFixture, ippDivide,
            ::testing::Combine(
                IPP_TEST_STHREAD,
                IPP_TEST_SIZES,
                ::testing::Values(CV_8U, CV_16U, CV_16S, CV_32F)
            ))
{
    const MultiplyParamType params = GetParam();
    const cv::Size size = get<1>(params);
    const int      type = get<2>(params);

    cv::Mat src1(size, type);
    cv::Mat src2(size, type);
    cv::Mat dst(size, type);
    cv::Mat dstRef(size, type);

    declare.in(src1, WARMUP_RNG).out(dst);
    cv::randu(src2, 1, 100);

    // Conformance check
    RUN_IPP(cv::divide(src1, src2, dst));
    RUN_REF(cv::divide(src1, src2, dstRef));
    VERIFY_DATA(dst, dstRef, 1, 1e-9);

    IPP_TEST_CYCLE_MULTIRUN(10) cv::divide(src1, src2, dst);
}

PERF_TEST_P(MultiplyFixture, ippDivide_Const,
            ::testing::Combine(
                IPP_TEST_STHREAD,
                IPP_TEST_SIZES,
                ::testing::Values(CV_8U, CV_16U, CV_16S, CV_32S, CV_32F, CV_64F)
            ))
{
    const MultiplyParamType params = GetParam();
    const cv::Size size = get<1>(params);
    const int      type = get<2>(params);
    const Scalar   s    = 1.333;

    cv::Mat src(size, type);
    cv::Mat dst(size, type);
    cv::Mat dstRef(size, type);

    declare.in(src, WARMUP_RNG).out(dst);

    // Conformance check
    RUN_IPP(cv::divide(src, s, dst));
    RUN_REF(cv::divide(src, s, dstRef));
    VERIFY_DATA(dst, dstRef, 1, 1e-9);

    IPP_TEST_CYCLE_MULTIRUN(10) cv::divide(src, s, dst);
}

PERF_TEST_P(MultiplyFixture, ippDivide_Scale,
            ::testing::Combine(
                IPP_TEST_STHREAD,
                IPP_TEST_SIZES,
                ::testing::Values(CV_8U, CV_16U, CV_16S)
            ))
{
    const MultiplyParamType params = GetParam();
    const cv::Size size = get<1>(params);
    const int      type = get<2>(params);

    cv::Mat src1(size, type);
    cv::Mat src2(size, CV_16U);
    cv::Mat dst(size, CV_32F);
    cv::Mat dstRef(size, CV_32F);

    declare.in(src1, WARMUP_RNG).out(dst);
    cv::randu(src2, 1, 100);

    // Conformance check
    RUN_IPP(cv::divide(src1, src2, dst, 2, CV_32F));
    RUN_REF(cv::divide(src1, src2, dstRef, 2, CV_32F));
    VERIFY_DATA(dst, dstRef, 1, 1e-9);

    IPP_TEST_CYCLE_MULTIRUN(10) cv::divide(src1, src2, dst, 2, CV_32F);
}
#endif

///////////// Add ////////////////////////

PERF_TEST_P(MultiplyFixture, ippAdd,
            ::testing::Combine(
                IPP_TEST_STHREAD,
                IPP_TEST_SIZES,
                ::testing::Values(CV_8U, CV_16U, CV_16S, CV_32F)
            ))
{
    const MultiplyParamType params = GetParam();
    const cv::Size size = get<1>(params);
    const int      type = get<2>(params);

    cv::Mat src1(size, type);
    cv::Mat src2(size, type);
    cv::Mat dst(size, type);
    cv::Mat dstRef(size, type);

    declare.in(src1, src2, WARMUP_RNG).out(dst);

    // Conformance check
    RUN_IPP(cv::add(src1, src2, dst));
    RUN_REF(cv::add(src1, src2, dstRef));
    VERIFY_DATA(dst, dstRef, 1, 1e-9);

    IPP_TEST_CYCLE_MULTIRUN(10) cv::add(src1, src2, dst);
}

PERF_TEST_P(MultiplyFixture, ippAdd_Const,
            ::testing::Combine(
                IPP_TEST_STHREAD,
                IPP_TEST_SIZES,
                ::testing::Values(CV_8U, CV_16U, CV_16S, CV_32S, CV_32F, CV_64F)
            ))
{
    const MultiplyParamType params = GetParam();
    const cv::Size size = get<1>(params);
    const int      type = get<2>(params);
    const Scalar   s    = 50;

    cv::Mat src(size, type);
    cv::Mat dst(size, type);
    cv::Mat dstRef(size, type);

    declare.in(src, WARMUP_RNG).out(dst);

    // Conformance check
    RUN_IPP(cv::add(src, s, dst));
    RUN_REF(cv::add(src, s, dstRef));
    VERIFY_DATA(dst, dstRef, 1, 1e-9);

    IPP_TEST_CYCLE_MULTIRUN(10) cv::add(src, s, dst);
}

PERF_TEST_P(MultiplyFixture, ippAdd_Scale,
            ::testing::Combine(
                IPP_TEST_STHREAD,
                IPP_TEST_SIZES,
                ::testing::Values(CV_8U, CV_16U, CV_16S)
            ))
{
    const MultiplyParamType params = GetParam();
    const cv::Size size = get<1>(params);
    const int      type = get<2>(params);

    cv::Mat src1(size, type);
    cv::Mat src2(size, CV_16U);
    cv::Mat dst(size, CV_32F);
    cv::Mat dstRef(size, CV_32F);

    declare.in(src1, src2, WARMUP_RNG).out(dst);

    // Conformance check
    RUN_IPP(cv::add(src1, src2, dst, noArray(), CV_32F));
    RUN_REF(cv::add(src1, src2, dstRef, noArray(), CV_32F));
    VERIFY_DATA(dst, dstRef, 1, 1e-9);

    IPP_TEST_CYCLE_MULTIRUN(10) cv::add(src1, src2, dst, noArray(), CV_32F);
}

///////////// Subtract ////////////////////////

PERF_TEST_P(MultiplyFixture, ippSubtract,
            ::testing::Combine(
                IPP_TEST_STHREAD,
                IPP_TEST_SIZES,
                ::testing::Values(CV_8U, CV_16U, CV_16S, CV_32F)
            ))
{
    const MultiplyParamType params = GetParam();
    const cv::Size size = get<1>(params);
    const int      type = get<2>(params);

    cv::Mat src1(size, type);
    cv::Mat src2(size, type);
    cv::Mat dst(size, type);
    cv::Mat dstRef(size, type);

    declare.in(src1, src2, WARMUP_RNG).out(dst);

    // Conformance check
    RUN_IPP(cv::subtract(src1, src2, dst));
    RUN_REF(cv::subtract(src1, src2, dstRef));
    VERIFY_DATA(dst, dstRef, 1, 1e-9);

    IPP_TEST_CYCLE_MULTIRUN(10) cv::subtract(src1, src2, dst);
}

PERF_TEST_P(MultiplyFixture, ippSubtract_Const,
            ::testing::Combine(
                IPP_TEST_STHREAD,
                IPP_TEST_SIZES,
                ::testing::Values(CV_8U, CV_16U, CV_16S, CV_32S, CV_32F, CV_64F)
            ))
{
    const MultiplyParamType params = GetParam();
    const cv::Size size = get<1>(params);
    const int      type = get<2>(params);
    const Scalar   s    = 50;

    cv::Mat src(size, type);
    cv::Mat dst(size, type);
    cv::Mat dstRef(size, type);

    declare.in(src, WARMUP_RNG).out(dst);

    // Conformance check
    RUN_IPP(cv::subtract(src, s, dst));
    RUN_REF(cv::subtract(src, s, dstRef));
    VERIFY_DATA(dst, dstRef, 1, 1e-9);

    IPP_TEST_CYCLE_MULTIRUN(10) cv::subtract(src, s, dst);
}

PERF_TEST_P(MultiplyFixture, ippSubtract_Scale,
            ::testing::Combine(
                IPP_TEST_STHREAD,
                IPP_TEST_SIZES,
                ::testing::Values(CV_8U, CV_16U, CV_16S)
            ))
{
    const MultiplyParamType params = GetParam();
    const cv::Size size = get<1>(params);
    const int      type = get<2>(params);

    cv::Mat src1(size, type);
    cv::Mat src2(size, CV_16U);
    cv::Mat dst(size, CV_32F);
    cv::Mat dstRef(size, CV_32F);

    declare.in(src1, src2, WARMUP_RNG).out(dst);

    // Conformance check
    RUN_IPP(cv::subtract(src1, src2, dst, noArray(), CV_32F));
    RUN_REF(cv::subtract(src1, src2, dstRef, noArray(), CV_32F));
    VERIFY_DATA(dst, dstRef, 1, 1e-9);

    IPP_TEST_CYCLE_MULTIRUN(10) cv::subtract(src1, src2, dst, noArray(), CV_32F);
}

///////////// Absdiff ////////////////////////

PERF_TEST_P(MultiplyFixture, ippAbsdiff,
            ::testing::Combine(
                IPP_TEST_STHREAD,
                IPP_TEST_SIZES,
                ::testing::Values(CV_8U, CV_16U, CV_16S, CV_32F)
            ))
{
    const MultiplyParamType params = GetParam();
    const cv::Size size = get<1>(params);
    const int      type = get<2>(params);

    cv::Mat src1(size, type);
    cv::Mat src2(size, type);
    cv::Mat dst(size, type);
    cv::Mat dstRef(size, type);

    declare.in(src1, src2, WARMUP_RNG).out(dst);

    // Conformance check
    RUN_IPP(cv::absdiff(src1, src2, dst));
    RUN_REF(cv::absdiff(src1, src2, dstRef));
    VERIFY_DATA(dst, dstRef, 1, 1e-9);

    IPP_TEST_CYCLE_MULTIRUN(10) cv::absdiff(src1, src2, dst);
}

PERF_TEST_P(MultiplyFixture, ippAbsdiff_Const,
            ::testing::Combine(
                IPP_TEST_STHREAD,
                IPP_TEST_SIZES,
                ::testing::Values(CV_8U, CV_16U, CV_16S, CV_32S, CV_32F, CV_64F)
            ))
{
    const MultiplyParamType params = GetParam();
    const cv::Size size = get<1>(params);
    const int      type = get<2>(params);
    const Scalar   s    = 50;

    cv::Mat src(size, type);
    cv::Mat dst(size, type);
    cv::Mat dstRef(size, type);

    declare.in(src, WARMUP_RNG).out(dst);

    // Conformance check
    RUN_IPP(cv::absdiff(src, s, dst));
    RUN_REF(cv::absdiff(src, s, dstRef));
    VERIFY_DATA(dst, dstRef, 1, 1e-9);

    IPP_TEST_CYCLE_MULTIRUN(10) cv::absdiff(src, s, dst);
}

///////////// MinMax ////////////////////////

PERF_TEST_P(MultiplyFixture, ippMin,
            ::testing::Combine(
                IPP_TEST_STHREAD,
                IPP_TEST_SIZES,
                ::testing::Values(CV_8U, CV_16U, CV_16S, CV_32F, CV_64F)
            ))
{
    const MultiplyParamType params = GetParam();
    const cv::Size size = get<1>(params);
    const int      type = get<2>(params);

    cv::Mat src1(size, type);
    cv::Mat src2(size, type);
    cv::Mat dst(size, type);
    cv::Mat dstRef(size, type);

    declare.in(src1, src2, WARMUP_RNG).out(dst);

    // Conformance check
    RUN_IPP(cv::min(src1, src2, dst));
    RUN_REF(cv::min(src1, src2, dstRef));
    VERIFY_DATA(dst, dstRef, 0, 0);

    IPP_TEST_CYCLE_MULTIRUN(10) cv::min(src1, src2, dst);
}

PERF_TEST_P(MultiplyFixture, ippMax,
            ::testing::Combine(
                IPP_TEST_STHREAD,
                IPP_TEST_SIZES,
                ::testing::Values(CV_8U, CV_16U, CV_16S, CV_32F, CV_64F)
            ))
{
    const MultiplyParamType params = GetParam();
    const cv::Size size = get<1>(params);
    const int      type = get<2>(params);

    cv::Mat src1(size, type);
    cv::Mat src2(size, type);
    cv::Mat dst(size, type);
    cv::Mat dstRef(size, type);

    declare.in(src1, src2, WARMUP_RNG).out(dst);

    // Conformance check
    RUN_IPP(cv::max(src1, src2, dst));
    RUN_REF(cv::max(src1, src2, dstRef));
    VERIFY_DATA(dst, dstRef, 0, 0);

    IPP_TEST_CYCLE_MULTIRUN(10) cv::max(src1, src2, dst);
}

///////////// Bitwise ////////////////////////

PERF_TEST_P(MultiplyFixture, ippBitwiseAnd,
            ::testing::Combine(
                IPP_TEST_STHREAD,
                IPP_TEST_SIZES,
                ::testing::Values(CV_8U)
            ))
{
    const MultiplyParamType params = GetParam();
    const cv::Size size = get<1>(params);
    const int      type = get<2>(params);

    cv::Mat src1(size, type);
    cv::Mat src2(size, type);
    cv::Mat dst(size, type);
    cv::Mat dstRef(size, type);

    declare.in(src1, src2, WARMUP_RNG).out(dst);

    // Conformance check
    RUN_IPP(cv::bitwise_and(src1, src2, dst));
    RUN_REF(cv::bitwise_and(src1, src2, dstRef));
    VERIFY_DATA(dst, dstRef, 0, 0);

    IPP_TEST_CYCLE() cv::bitwise_and(src1, src2, dst);
}

PERF_TEST_P(MultiplyFixture, ippBitwiseAnd_Const,
            ::testing::Combine(
                IPP_TEST_STHREAD,
                IPP_TEST_SIZES,
                ::testing::Values(CV_8U, CV_16U, CV_32S)
            ))
{
    const MultiplyParamType params = GetParam();
    const cv::Size size = get<1>(params);
    const int      type = get<2>(params);
    const Scalar   s    = 50;

    cv::Mat src(size, type);
    cv::Mat dst(size, type);
    cv::Mat dstRef(size, type);

    declare.in(src, WARMUP_RNG).out(dst);

    // Conformance check
    RUN_IPP(cv::bitwise_and(src, s, dst));
    RUN_REF(cv::bitwise_and(src, s, dstRef));
    VERIFY_DATA(dst, dstRef, 0, 0);

    IPP_TEST_CYCLE() cv::bitwise_and(src, s, dst);
}

PERF_TEST_P(MultiplyFixture, ippBitwiseOr,
            ::testing::Combine(
                IPP_TEST_STHREAD,
                IPP_TEST_SIZES,
                ::testing::Values(CV_8U)
            ))
{
    const MultiplyParamType params = GetParam();
    const cv::Size size = get<1>(params);
    const int      type = get<2>(params);

    cv::Mat src1(size, type);
    cv::Mat src2(size, type);
    cv::Mat dst(size, type);
    cv::Mat dstRef(size, type);

    declare.in(src1, src2, WARMUP_RNG).out(dst);

    // Conformance check
    RUN_IPP(cv::bitwise_or(src1, src2, dst));
    RUN_REF(cv::bitwise_or(src1, src2, dstRef));
    VERIFY_DATA(dst, dstRef, 0, 0);

    IPP_TEST_CYCLE() cv::bitwise_or(src1, src2, dst);
}

PERF_TEST_P(MultiplyFixture, ippBitwiseOr_Const,
            ::testing::Combine(
                IPP_TEST_STHREAD,
                IPP_TEST_SIZES,
                ::testing::Values(CV_8U, CV_16U, CV_32S)
            ))
{
    const MultiplyParamType params = GetParam();
    const cv::Size size = get<1>(params);
    const int      type = get<2>(params);
    const Scalar   s    = 50;

    cv::Mat src(size, type);
    cv::Mat dst(size, type);
    cv::Mat dstRef(size, type);

    declare.in(src, WARMUP_RNG).out(dst);

    // Conformance check
    RUN_IPP(cv::bitwise_or(src, s, dst));
    RUN_REF(cv::bitwise_or(src, s, dstRef));
    VERIFY_DATA(dst, dstRef, 0, 0);

    IPP_TEST_CYCLE() cv::bitwise_or(src, s, dst);
}

PERF_TEST_P(MultiplyFixture, ippBitwiseNot,
            ::testing::Combine(
                IPP_TEST_STHREAD,
                IPP_TEST_SIZES,
                ::testing::Values(CV_8U)
            ))
{
    const MultiplyParamType params = GetParam();
    const cv::Size size = get<1>(params);
    const int      type = get<2>(params);

    cv::Mat src(size, type);
    cv::Mat dst(size, type);
    cv::Mat dstRef(size, type);

    declare.in(src, WARMUP_RNG).out(dst);

    // Conformance check
    RUN_IPP(cv::bitwise_not(src, dst));
    RUN_REF(cv::bitwise_not(src, dstRef));
    VERIFY_DATA(dst, dstRef, 0, 0);

    IPP_TEST_CYCLE() cv::bitwise_not(src, dst);
}

PERF_TEST_P(MultiplyFixture, ippBitwiseXor,
            ::testing::Combine(
                IPP_TEST_STHREAD,
                IPP_TEST_SIZES,
                ::testing::Values(CV_8U)
            ))
{
    const MultiplyParamType params = GetParam();
    const cv::Size size = get<1>(params);
    const int      type = get<2>(params);

    cv::Mat src1(size, type);
    cv::Mat src2(size, type);
    cv::Mat dst(size, type);
    cv::Mat dstRef(size, type);

    declare.in(src1, src2, WARMUP_RNG).out(dst);

    // Conformance check
    RUN_IPP(cv::bitwise_xor(src1, src2, dst));
    RUN_REF(cv::bitwise_xor(src1, src2, dstRef));
    VERIFY_DATA(dst, dstRef, 0, 0);

    IPP_TEST_CYCLE() cv::bitwise_xor(src1, src2, dst);
}

PERF_TEST_P(MultiplyFixture, ippBitwiseXor_Const,
            ::testing::Combine(
                IPP_TEST_STHREAD,
                IPP_TEST_SIZES,
                ::testing::Values(CV_8U, CV_16U, CV_32S)
            ))
{
    const MultiplyParamType params = GetParam();
    const cv::Size size = get<1>(params);
    const int      type = get<2>(params);
    const Scalar   s    = 50;

    cv::Mat src(size, type);
    cv::Mat dst(size, type);
    cv::Mat dstRef(size, type);

    declare.in(src, WARMUP_RNG).out(dst);

    // Conformance check
    RUN_IPP(cv::bitwise_xor(src, s, dst));
    RUN_REF(cv::bitwise_xor(src, s, dstRef));
    VERIFY_DATA(dst, dstRef, 0, 0);

    IPP_TEST_CYCLE() cv::bitwise_xor(src, s, dst);
}

///////////// Compare ////////////////////////

CV_ENUM(CompareType, CMP_EQ, CMP_GT, CMP_GE, CMP_LT, CMP_LE, CMP_NE)

typedef tuple<IppTestThreads, Size, MatType, CompareType> CompareParamType;
typedef TestBaseWithParam<CompareParamType> CompareFixture;

PERF_TEST_P(CompareFixture, ippCompare,
            ::testing::Combine(
                IPP_TEST_STHREAD,
                IPP_TEST_SIZES,
                ::testing::Values(CV_8U, CV_16U, CV_16S, CV_32F),
                CompareType::all()
            ))
{
    const CompareParamType params = GetParam();
    const cv::Size size = get<1>(params);
    const int      type = get<2>(params);
    const int      cmp  = get<3>(params);

    cv::Mat src1(size, type);
    cv::Mat src2(size, type);
    cv::Mat dst(size, type);
    cv::Mat dstRef(size, type);

    declare.in(src1, src2, WARMUP_RNG).out(dst);

    // Conformance check
    RUN_IPP(cv::compare(src1, src2, dst, cmp));
    RUN_REF(cv::compare(src1, src2, dstRef, cmp));
    VERIFY_DATA(dst, dstRef, 0, 0);

    IPP_TEST_CYCLE_MULTIRUN(10) cv::compare(src1, src2, dst, cmp);
}

PERF_TEST_P(CompareFixture, ippCompare_Const,
            ::testing::Combine(
                IPP_TEST_STHREAD,
                IPP_TEST_SIZES,
                ::testing::Values(CV_8U, CV_16U, CV_16S, CV_32F),
                CompareType::all()
            ))
{
    const CompareParamType params = GetParam();
    const cv::Size size = get<1>(params);
    const int      type = get<2>(params);
    const int      cmp  = get<3>(params);
    const Scalar   s    = 50;

    cv::Mat src(size, type);
    cv::Mat dst(size, type);
    cv::Mat dstRef(size, type);

    declare.in(src, WARMUP_RNG).out(dst);

    // Conformance check
    RUN_IPP(cv::compare(src, s, dst, cmp));
    RUN_REF(cv::compare(src, s, dstRef, cmp));
    VERIFY_DATA(dst, dstRef, 0, 0);

    IPP_TEST_CYCLE_MULTIRUN(10) cv::compare(src, s, dst, cmp);
}

///////////// Magnitude ////////////////////////

PERF_TEST_P(MultiplyFixture, ippMagnitude,
            ::testing::Combine(
                IPP_TEST_STHREAD,
                IPP_TEST_SIZES,
                ::testing::Values(CV_32F, CV_64F)
            ))
{
    const MultiplyParamType params = GetParam();
    const cv::Size size = get<1>(params);
    const int      type = get<2>(params);

    cv::Mat src1(size, type);
    cv::Mat src2(size, type);
    cv::Mat dst(size, type);
    cv::Mat dstRef(size, type);

    declare.in(src1, src2, WARMUP_RNG).out(dst);

    // Conformance check
    RUN_IPP(cv::magnitude(src1, src2, dst));
    RUN_REF(cv::magnitude(src1, src2, dstRef));
    VERIFY_DATA_PRES(dst, dstRef, 0, 1e-6, 1e-12);

    IPP_TEST_CYCLE_MULTIRUN(10) cv::magnitude(src1, src2, dst);
}

///////////// Log ////////////////////////

PERF_TEST_P(MultiplyFixture, ippLog,
            ::testing::Combine(
                IPP_TEST_STHREAD,
                IPP_TEST_SIZES,
                ::testing::Values(CV_32F, CV_64F)
            ))
{
    const MultiplyParamType params = GetParam();
    const cv::Size size = get<1>(params);
    const int      type = get<2>(params);

    cv::Mat src(size, type);
    cv::Mat dst(size, type);
    cv::Mat dstRef(size, type);

    declare.in(src).out(dst);
    randu(src, 0.1, 1000);

    // Conformance check
    RUN_IPP(log(src, dst));
    RUN_REF(log(src, dstRef));
    VERIFY_DATA_PRES(dst, dstRef, 0, 1e-7, 1e-12);

    IPP_TEST_CYCLE() log(src, dst);
}

///////////// Exp ////////////////////////

PERF_TEST_P(MultiplyFixture, ippExp,
            ::testing::Combine(
                IPP_TEST_STHREAD,
                IPP_TEST_SIZES,
                ::testing::Values(CV_32F, CV_64F)
            ))
{
    const MultiplyParamType params = GetParam();
    const cv::Size size = get<1>(params);
    const int      type = get<2>(params);

    cv::Mat src(size, type);
    cv::Mat dst(size, type);
    cv::Mat dstRef(size, type);

    declare.in(src).out(dst);
    randu(src, 0.1, 16);

    // Conformance check
    RUN_IPP(cv::exp(src, dst));
    RUN_REF(cv::exp(src, dstRef));
    VERIFY_DATA_PRES(dst, dstRef, 0, 1e-6, 1e-12);

    int runs = (size.width < sz2160p.width) ? 10 : 1;
    IPP_TEST_CYCLE_MULTIRUN(runs) cv::exp(src, dst);
}

///////////// Pow ////////////////////////

typedef tuple<IppTestThreads, Size, MatType, double> PowParamType;
typedef TestBaseWithParam<PowParamType> PowFixture;

PERF_TEST_P(PowFixture, ippPow,
            ::testing::Combine(
                IPP_TEST_STHREAD,
                IPP_TEST_SIZES,
                ::testing::Values(CV_32F, CV_64F),
                ::testing::Values(-6, -5, -4.5, -4, -3, -2, -1, -0.7, -0.5, 0, 0.5, 0.7, 1, 2, 3, 4, 4.5, 5, 6)
            ))
{
    const PowParamType params = GetParam();
    const cv::Size size  = get<1>(params);
    const int      type  = get<2>(params);
    const double   power = get<3>(params);

    cv::Mat src(size, type);
    cv::Mat dst(size, type);
    cv::Mat dstRef(size, type);

    declare.in(src).out(dst);
    randu(src, 0.1, 25);

    // Conformance check
    RUN_IPP(cv::pow(src, power, dst));
    RUN_REF(cv::pow(src, power, dstRef));
    VERIFY_DATA_PRES(dst, dstRef, 0, 4e-6, 1e-12);

    IPP_TEST_CYCLE() cv::pow(src, power, dst);
}

///////////// PolarToCart ////////////////////////

PERF_TEST_P(MultiplyFixture, ippPolarToCart,
            ::testing::Combine(
                IPP_TEST_STHREAD,
                IPP_TEST_SIZES,
                ::testing::Values(CV_32F, CV_64F)
            ))
{
    const MultiplyParamType params = GetParam();
    const cv::Size size = get<1>(params);
    const int      type = get<2>(params);

    cv::Mat srcMag(size, type);
    cv::Mat srcAng(size, type);
    cv::Mat dstX(size, type);
    cv::Mat dstY(size, type);
    cv::Mat dstXRef(size, type);
    cv::Mat dstYRef(size, type);

    declare.in(srcMag, srcAng, WARMUP_RNG).out(dstX, dstY);

    // Conformance check
    RUN_IPP(cv::polarToCart(srcMag, srcAng, dstX, dstY));
    RUN_REF(cv::polarToCart(srcMag, srcAng, dstXRef, dstYRef));
    VERIFY_DATA_PRES(dstX, dstXRef, 0, 5e-7, 1e-3);
    VERIFY_DATA_PRES(dstY, dstYRef, 0, 5e-7, 1e-3);

    IPP_TEST_CYCLE_MULTIRUN(10) cv::polarToCart(srcMag, srcAng, dstX, dstY);
}

typedef tuple<IppTestThreads, Size, MatType> CopyToMaskParamType;
typedef TestBaseWithParam<CopyToMaskParamType> CopyToMaskFixture;

#if 0
///////////// CopyTo ////////////////////////
PERF_TEST_P(CopyToMaskFixture, ippCopyTo,
            ::testing::Combine(
                IPP_TEST_STHREAD,
                IPP_TEST_SIZES,
                ::testing::Values(CV_8UC1)
            ))
{
    const CopyToMaskParamType params = GetParam();
    const cv::Size size = get<1>(params);
    const int      type = get<2>(params);

    cv::Mat src(size, type);
    cv::Mat dst(size, type);
    cv::Mat dstRef(size, type);

    declare.in(src, WARMUP_RNG).out(dst);

    // Conformance check
    RUN_IPP(src.copyTo(dst));
    RUN_REF(src.copyTo(dstRef));
    VERIFY_DATA(dst, dstRef, 0, 0);

    int runs = (size.width <= 320) ? 1000 : ((size.width <= 1920) ? 100 : 1);
    IPP_TEST_CYCLE_MULTIRUN(runs) src.copyTo(dst);
}
#endif

///////////// CopyTo with mask ////////////////////////

PERF_TEST_P(CopyToMaskFixture, ippCopyTo_Mask,
            ::testing::Combine(
                IPP_TEST_STHREAD,
                IPP_TEST_SIZES,
                ::testing::Values(CV_8UC1, CV_8UC2, CV_8UC3, CV_8UC4, CV_16UC1, CV_16UC2, CV_16UC3, CV_16UC4, CV_16SC1, CV_16SC2, CV_16SC3, CV_16SC4, CV_32SC1, CV_32SC2, CV_32SC3, CV_32SC4, CV_32FC1, CV_32FC2, CV_32FC3, CV_32FC4, CV_64FC1, CV_64FC2)
            ))
{
    const CopyToMaskParamType params = GetParam();
    const cv::Size size       = get<1>(params);
    const int      type       = get<2>(params);

    cv::Mat src(size, type);
    cv::Mat dst(size, type);
    cv::Mat dstRef(size, type);
    cv::Mat mask(size, CV_8UC1);

    declare.in(src, WARMUP_RNG).out(dst, dstRef);
    randu(mask, 0, 2);

    // Conformance check
    RUN_IPP(src.copyTo(dst, mask));
    RUN_REF(src.copyTo(dstRef, mask));
    VERIFY_DATA(dst, dstRef, 0, 0);

    int runs = (size.width <= 320) ? 100 : ((size.width <= 800) ? 5 : 1);
    IPP_TEST_CYCLE_MULTIRUN(runs) src.copyTo(dst, mask);
}

#if 0
///////////// SetTo ////////////////////////

PERF_TEST_P(CopyToMaskFixture, ippSetTo,
            ::testing::Combine(
                IPP_TEST_STHREAD,
                IPP_TEST_SIZES,
                ::testing::Values(CV_8UC1, CV_8UC3, CV_8UC4, CV_16UC1, CV_16UC3, CV_16UC4, CV_32SC1, CV_32SC3, CV_32SC4, CV_32FC1, CV_32FC3, CV_32FC4)
            ))
{
    const CopyToMaskParamType params = GetParam();
    const cv::Size size       = get<1>(params);
    const int      type       = get<2>(params);

    cv::Mat dst(size, type);
    cv::Mat dstRef(size, type);

    declare.in(dst, WARMUP_RNG).out(dst);

    // Conformance check
    RUN_IPP(dst.setTo(Scalar::all(17)));
    RUN_REF(dstRef.setTo(Scalar::all(17)));
    VERIFY_DATA(dst, dstRef, 0, 0);

    int runs = (size.width <= 320) ? 100 : ((size.width <= 800) ? 10 : 1);
    IPP_TEST_CYCLE_MULTIRUN(runs) dst.setTo(Scalar::all(17));
}
#endif

///////////// SetTo with mask ////////////////////////

PERF_TEST_P(CopyToMaskFixture, ippSetTo_Mask,
            ::testing::Combine(
                IPP_TEST_STHREAD,
                IPP_TEST_SIZES,
                ::testing::Values(CV_8UC1, CV_8UC3, CV_8UC4, CV_16UC1, CV_16UC3, CV_16UC4, CV_32SC1, CV_32SC3, CV_32SC4, CV_32FC1, CV_32FC3, CV_32FC4)
            ))
{
    const CopyToMaskParamType params = GetParam();
    const cv::Size size       = get<1>(params);
    const int      type       = get<2>(params);

    cv::Mat dst(size, type);
    cv::Mat dstRef(size, type);
    cv::Mat mask(size, CV_8UC1);

    declare.in(dst, WARMUP_RNG).out(dst, dstRef);
    randu(mask, 0, 2);

    // Conformance check
    RUN_IPP(dst.setTo(Scalar::all(17), mask));
    RUN_REF(dstRef.setTo(Scalar::all(17), mask));
    VERIFY_DATA(dst, dstRef, 0, 0);

    int runs = (size.width <= 320) ? 100 : ((size.width <= 800) ? 10 : 1);
    IPP_TEST_CYCLE_MULTIRUN(runs) dst.setTo(Scalar::all(17), mask);
}

#if 0
///////////// SetIdentity ////////////////////////

PERF_TEST_P(CopyToMaskFixture, ippSetIdentity,
            ::testing::Combine(
                IPP_TEST_STHREAD,
                IPP_TEST_SIZES,
                ::testing::Values(CV_8UC1, CV_8UC3, CV_8UC4, CV_16UC1, CV_16UC3, CV_16UC4, CV_32SC1, CV_32SC3, CV_32SC4, CV_32FC1, CV_32FC3, CV_32FC4)
            ))
{
    const CopyToMaskParamType params = GetParam();
    const cv::Size size       = get<1>(params);
    const int      type       = get<2>(params);

    cv::Mat dst(size, type);
    cv::Mat dstRef(size, type);
    declare.in(dst, WARMUP_RNG).out(dst);

    // Conformance check
    RUN_IPP(cv::setIdentity(dst, cv::Scalar::all(181)));
    RUN_REF(cv::setIdentity(dstRef, cv::Scalar::all(181)));
    VERIFY_DATA(dst, dstRef, 0, 0);

    int runs = (size.width <= 320) ? 100 : ((size.width <= 800) ? 10 : 1);
    IPP_TEST_CYCLE_MULTIRUN(runs) cv::setIdentity(dst, cv::Scalar::all(181));
}
#endif

/////////// ConvertTo //////////////////////

typedef tuple<IppTestThreads, Size, MatType, MatType> ConvertToParamType;
typedef TestBaseWithParam<ConvertToParamType> ConvertToFixture;

PERF_TEST_P(ConvertToFixture, ippConvertTo_NoScale,
            ::testing::Combine(
                IPP_TEST_STHREAD,
                IPP_TEST_SIZES,
                ::testing::Values(CV_8U, CV_8S, CV_16U, CV_16S, CV_32S, CV_32F, CV_64F),
                ::testing::Values(CV_8U, CV_8S, CV_16U, CV_16S, CV_32S, CV_32F, CV_64F)
            ))
{
    const ConvertToParamType params = GetParam();
    const cv::Size size       = get<1>(params);
    const int      srcType    = CV_MAKE_TYPE(get<2>(params), 1);
    const int      dstType    = CV_MAKE_TYPE(get<3>(params), 1);

    cv::Mat src(size, srcType);
    cv::Mat dst(size, dstType);
    cv::Mat dstRef(size, dstType);

    declare.in(src, WARMUP_RNG).out(dst);

    // Conformance check
    RUN_IPP(src.convertTo(dst, dstType));
    RUN_REF(src.convertTo(dstRef, dstType));
    VERIFY_DATA_PRES(dst, dstRef, 1, 0, 0); // different rounding

    IPP_TEST_CYCLE_MULTIRUN(10) src.convertTo(dst, dstType);
}

typedef tuple<IppTestThreads, Size, MatType, MatType, double, double> ConvertToSParamType;
typedef TestBaseWithParam<ConvertToSParamType> ConvertToSFixture;

PERF_TEST_P(ConvertToSFixture, ippConvertTo_Scale,
            ::testing::Combine(
                IPP_TEST_STHREAD,
                IPP_TEST_SIZES,
                ::testing::Values(CV_8U, CV_8S, CV_16U, CV_16S, CV_32S, CV_32F, CV_64F),
                ::testing::Values(CV_8U, CV_8S, CV_16U, CV_16S, CV_32S, CV_32F, CV_64F),
                ::testing::Values(1.0, 1./255),
                ::testing::Values(5.)
            ))
{
    const ConvertToSParamType params = GetParam();
    const cv::Size size       = get<1>(params);
    const int      srcType    = CV_MAKE_TYPE(get<2>(params), 1);
    const int      dstType    = CV_MAKE_TYPE(get<3>(params), 1);
    const double   scale      = get<4>(params);
    const double   shift      = get<5>(params);

    cv::Mat src(size, srcType);
    cv::Mat dst(size, dstType);
    cv::Mat dstRef(size, dstType);

    declare.in(src, WARMUP_RNG).out(dst);

    // Conformance check
    RUN_IPP(src.convertTo(dst, dstType, scale, shift));
    RUN_REF(src.convertTo(dstRef, dstType, scale, shift));
    VERIFY_DATA_PRES(dst, dstRef, 1, 4e-7, 1e-15);

    IPP_TEST_CYCLE_MULTIRUN(10) src.convertTo(dst, dstType, scale, shift);
}

typedef tuple<IppTestThreads, Size, MatType, double, double> ConvertToSIParamType;
typedef TestBaseWithParam<ConvertToSIParamType> ConvertToSIFixture;

PERF_TEST_P(ConvertToSIFixture, ippConvertTo_ScaleInplace,
            ::testing::Combine(
                IPP_TEST_STHREAD,
                IPP_TEST_SIZES,
                ::testing::Values(CV_8U, CV_8S, CV_16U, CV_16S, CV_32S, CV_32F, CV_64F),
                ::testing::Values(1.0, 1./255),
                ::testing::Values(5.)
            ))
{
    const ConvertToSIParamType params = GetParam();
    const cv::Size size       = get<1>(params);
    const int      type       = CV_MAKE_TYPE(get<2>(params), 1);
    const double   scale      = get<3>(params);
    const double   shift      = get<4>(params);

    cv::Mat src(size, type);
    cv::Mat dst(size, type);
    cv::Mat dstRef(size, type);

    declare.in(src, WARMUP_RNG).out(dst);
    src.copyTo(dst);
    src.copyTo(dstRef);

    // Conformance check
    RUN_IPP(dst.convertTo(dst, type, scale, shift));
    RUN_REF(dstRef.convertTo(dstRef, type, scale, shift));
    VERIFY_DATA_PRES(dst, dstRef, 1, 1e-7, 1e-15);

    IPP_TEST_CYCLE_MULTIRUN(10) dst.convertTo(dst, type, scale, shift);
}

/////////// Split //////////////////////

typedef tuple<IppTestThreads, Size, MatType> SplitParamType;
typedef TestBaseWithParam<SplitParamType> SplitFixture;

PERF_TEST_P(SplitFixture, ippSplit,
            ::testing::Combine(
                IPP_TEST_STHREAD,
                IPP_TEST_SIZES,
                ::testing::Values(CV_8UC3, CV_8UC4, CV_16UC3, CV_16UC4, CV_32FC3, CV_32FC4)
            ))
{
    const SplitParamType params = GetParam();
    const cv::Size size       = get<1>(params);
    const int      type       = get<2>(params);

    cv::Mat src(size, type);
    vector<Mat> dst(4);
    vector<Mat> dstRef(4);
    dst[0].create(size, CV_MAT_DEPTH(type));
    dst[1].create(size, CV_MAT_DEPTH(type));
    dst[2].create(size, CV_MAT_DEPTH(type));
    dst[3].create(size, CV_MAT_DEPTH(type));

    declare.in(src, WARMUP_RNG).out(dst);

    RUN_IPP(cv::split(src, dst));
    RUN_REF(cv::split(src, dstRef));
    VERIFY_DATA(dst[0], dstRef[0], 0, 0);
    VERIFY_DATA(dst[1], dstRef[1], 0, 0);
    VERIFY_DATA(dst[2], dstRef[2], 0, 0);
    if(CV_MAT_CN(type) == 4)
    {
        VERIFY_DATA(dst[3], dstRef[3], 0, 0);
    }

    int runs = (size.width <= 640) ? 10 : 1;
    IPP_TEST_CYCLE_MULTIRUN(runs) cv::split(src, dst);
}

/////////// Merge //////////////////////

PERF_TEST_P(SplitFixture, ippMerge,
            ::testing::Combine(
                IPP_TEST_STHREAD,
                IPP_TEST_SIZES,
                ::testing::Values(CV_8UC3, CV_8UC4, CV_16UC3, CV_16UC4, CV_32FC3, CV_32FC4)
            ))
{
    const SplitParamType params = GetParam();
    const cv::Size size       = get<1>(params);
    const int      type       = get<2>(params);

    vector<Mat> src(4);
    cv::Mat dst(size, type);
    cv::Mat dstRef(size, type);
    src[0].create(size, CV_MAT_DEPTH(type));
    src[1].create(size, CV_MAT_DEPTH(type));
    src[2].create(size, CV_MAT_DEPTH(type));
    src[3].create(size, CV_MAT_DEPTH(type));

    declare.in(src, WARMUP_RNG).out(dst);

    RUN_IPP(cv::merge(src, dst));
    RUN_REF(cv::merge(src, dstRef));
    VERIFY_DATA(dst, dstRef, 0, 0);

    int runs = (size.width <= 640) ? 10 : 1;
    IPP_TEST_CYCLE_MULTIRUN(runs) cv::merge(src, dst);
}

/////////// ExtractChannel //////////////////////

PERF_TEST_P(SplitFixture, ippExtractChannel,
            ::testing::Combine(
                IPP_TEST_STHREAD,
                IPP_TEST_SIZES,
                ::testing::Values(CV_8UC3, CV_8UC4, CV_16UC3, CV_16UC4, CV_32FC3, CV_32FC4)
            ))
{
    const SplitParamType params = GetParam();
    const cv::Size size       = get<1>(params);
    const int      type       = get<2>(params);

    cv::Mat src(size, type);
    cv::Mat dst(size, CV_MAT_DEPTH(type));
    cv::Mat dstRef(size, CV_MAT_DEPTH(type));

    declare.in(src, WARMUP_RNG).out(dst);

    RUN_IPP(cv::extractChannel(src, dst, 2));
    RUN_REF(cv::extractChannel(src, dstRef, 2));
    VERIFY_DATA(dst, dstRef, 0, 0);

    IPP_TEST_CYCLE_MULTIRUN(10) cv::extractChannel(src, dst, 2);
}

/////////// InsertChannel //////////////////////

PERF_TEST_P(SplitFixture, ippInsertChannel,
            ::testing::Combine(
                IPP_TEST_STHREAD,
                IPP_TEST_SIZES,
                ::testing::Values(CV_8UC3, CV_8UC4, CV_16UC3, CV_16UC4, CV_32FC3, CV_32FC4)
            ))
{
    const SplitParamType params = GetParam();
    const cv::Size size       = get<1>(params);
    const int      type       = get<2>(params);

    cv::Mat src(size, CV_MAT_DEPTH(type));
    cv::Mat dst(size, type);
    cv::Mat dstRef(size, type);

    declare.in(src, WARMUP_RNG).out(dst, dstRef);

    RUN_IPP(cv::insertChannel(src, dst, 2));
    RUN_REF(cv::insertChannel(src, dstRef, 2));
    VERIFY_DATA(dst, dstRef, 0, 0);

    IPP_TEST_CYCLE_MULTIRUN(10) cv::insertChannel(src, dst, 2);
}

///////////// Flip ////////////////////////

enum
{
    FLIP_BOTH,
    FLIP_ROWS,
    FLIP_COLS
};

CV_ENUM(FlipType, FLIP_BOTH, FLIP_ROWS, FLIP_COLS)

typedef tuple<IppTestThreads, Size, MatType, FlipType> FlipParamType;
typedef TestBaseWithParam<FlipParamType> FlipFixture;

PERF_TEST_P(FlipFixture, ippFlip,
            ::testing::Combine(
                IPP_TEST_STHREAD,
                IPP_TEST_SIZES,
                ::testing::Values(CV_8UC1, CV_8UC3, CV_8UC4, CV_16UC1, CV_16UC3, CV_16UC4, CV_32FC1, CV_32FC3, CV_32FC4),
                FlipType::all()
            ))
{
    const FlipParamType params = GetParam();
    const cv::Size size       = get<1>(params);
    const int      type       = get<2>(params);
          int      flipType   = get<3>(params);

    if(flipType == FLIP_BOTH)
        flipType = -1;
    else if(flipType == FLIP_ROWS)
        flipType = 0;
    else if(flipType == FLIP_COLS)
        flipType = 1;

    cv::Mat src(size, type);
    cv::Mat dst(size, type);
    cv::Mat dstRef(size, type);

    declare.in(src, WARMUP_RNG).out(dst);

    // Conformance check
    RUN_IPP(cv::flip(src, dst, flipType));
    RUN_REF(cv::flip(src, dstRef, flipType));
    VERIFY_DATA(dst, dstRef, 0, 0);

    IPP_TEST_CYCLE() cv::flip(src, dst, flipType);
}

PERF_TEST_P(FlipFixture, ippFlip_Inplace,
            ::testing::Combine(
                IPP_TEST_STHREAD,
                IPP_TEST_SIZES,
                ::testing::Values(CV_8UC1, CV_8UC3, CV_8UC4, CV_16UC1, CV_16UC3, CV_16UC4, CV_32FC1, CV_32FC3, CV_32FC4),
                FlipType::all()
            ))
{
    const FlipParamType params = GetParam();
    const cv::Size size       = get<1>(params);
    const int      type       = get<2>(params);
          int      flipType   = get<3>(params);

    if(flipType == FLIP_BOTH)
        flipType = -1;
    else if(flipType == FLIP_ROWS)
        flipType = 0;
    else if(flipType == FLIP_COLS)
        flipType = 1;

    cv::Mat src(size, type);
    cv::Mat dst(size, type);
    cv::Mat dstRef(size, type);

    declare.in(src, WARMUP_RNG).out(dst);
    src.copyTo(dst);
    src.copyTo(dstRef);

    // Conformance check
    RUN_IPP(cv::flip(dst, dst, flipType));
    RUN_REF(cv::flip(dstRef, dstRef, flipType));
    VERIFY_DATA(dst, dstRef, 0, 0);

    IPP_TEST_CYCLE_MULTIRUN(10) cv::flip(dst, dst, flipType);
}

///////////// LUT ////////////////////////
#if !IPP_DISABLE_PERF_LUT
typedef tuple<IppTestThreads, Size, MatPairs> LUTParamType;
typedef TestBaseWithParam<LUTParamType> LUTFixture;

PERF_TEST_P(LUTFixture, ippLUT,
            ::testing::Combine(
                IppTestThreads::all(),
                IPP_TEST_SIZES,
                ::testing::Values(CV_8U_8U_C1, CV_8U_8U_C3, CV_8U_8U_C4, CV_8U_32S_C1)
            ))
{
    const LUTParamType params = GetParam();
    const cv::Size size     = get<1>(params);
    const int      srcType  = CV_MAT_PAIR_GET1(get<2>(params));
    const int      dstType  = CV_MAT_PAIR_GET2(get<2>(params));

    // Init IPP extensions
    {
        TS_SET_THREADS(get<0>(params));
    }

    cv::Mat src(size, srcType);
    cv::Mat lut(1, 256, dstType);
    cv::Mat dst(size, dstType);
    cv::Mat dstRef(size, dstType);

    declare.in(src, lut, WARMUP_RNG).out(dst);
    randu(lut, 0, src.cols*src.rows);

    // Conformance check
    RUN_IPP(cv::LUT(src, lut, dst));
    RUN_REF(cv::LUT(src, lut, dstRef));
    VERIFY_DATA(dst, dstRef, 0, 0);

    IPP_TEST_CYCLE_MULTIRUN(10) cv::LUT(src, lut, dst);
}
#endif

///////////// MinMaxIdx ////////////////////////

enum
{
    MIN         = 0x1,
    MAX         = 0x2,
    IDX         = 0x4,
    MIN_IDX     = MIN|IDX,
    MAX_IDX     = MAX|IDX,
    MIN_MAX     = MIN|MAX,
    MIN_MAX_IDX = MIN_IDX|MAX_IDX,
};

CV_ENUM(MinMaxType, MIN_MAX_IDX, MIN_MAX, MIN_IDX, MAX_IDX)


typedef tuple<IppTestThreads, Size, MatType, MinMaxType> MinMaxIdxParamType;
typedef TestBaseWithParam<MinMaxIdxParamType> MinMaxIdxFixture;

PERF_TEST_P(MinMaxIdxFixture, ippMinMaxIdx,
            ::testing::Combine(
                IPP_TEST_STHREAD,
                IPP_TEST_SIZES,
                ::testing::Values(CV_8UC1, CV_16UC1, CV_16SC1, CV_32FC1),
                MinMaxType::all()
            ))
{
    const MinMaxIdxParamType params = GetParam();
    const cv::Size size     = get<1>(params);
    const int      type     = get<2>(params);
    const int      minMaxT  = get<3>(params);

    cv::Mat src(size, type);
    double  minVal = 0;
    double  maxVal = 0;
    int     minIdx[2] = {0};
    int     maxIdx[2] = {0};
    double  minValRef = 0;
    double  maxValRef = 0;
    int     minIdxRef[2] = {0};
    int     maxIdxRef[2] = {0};

    double  *pMinVal    = (minMaxT&MIN)?&minVal:NULL;
    double  *pMaxVal    = (minMaxT&MAX)?&maxVal:NULL;
    int     *pMinIdx    = (minMaxT&MIN && minMaxT&IDX)?minIdx:NULL;
    int     *pMaxIdx    = (minMaxT&MAX && minMaxT&IDX)?maxIdx:NULL;
    double  *pMinValRef = (minMaxT&MIN)?&minValRef:NULL;
    double  *pMaxValRef = (minMaxT&MAX)?&maxValRef:NULL;
    int     *pMinIdxRef = (minMaxT&MIN && minMaxT&IDX)?minIdxRef:NULL;
    int     *pMaxIdxRef = (minMaxT&MAX && minMaxT&IDX)?maxIdxRef:NULL;

    declare.in(src, WARMUP_RNG);
    randu(src, 1, 155); // use middle of the range to avoid short-cut optimizations triggering

    // Conformance check
    RUN_IPP(cv::minMaxIdx(src, pMinVal, pMaxVal, pMinIdx, pMaxIdx));
    RUN_REF(cv::minMaxIdx(src, pMinValRef, pMaxValRef, pMinIdxRef, pMaxIdxRef));
    {
        if(minVal != minValRef || maxVal != maxValRef
            || minIdx[0] != minIdxRef[0] || minIdx[1] != minIdxRef[1]
            || maxIdx[0] != maxIdxRef[0] || maxIdx[1] != maxIdxRef[1])
        {
            ADD_FAILURE() << "MinMax doesn't match\nEXPECTED:" << "\nMIN=" << minValRef << "\nMAX=" << maxValRef << "\nMIN_IDX=" << minIdxRef[0] << ", " << minIdxRef[1] << "\nMAX_IDX=" << maxIdxRef[0] << ", "  << maxIdxRef[1]
                << "\n\nACTUAL:" << "\nMIN=" << minVal << "\nMAX=" << maxVal << "\nMIN_IDX=" << minIdx[0] << ", "  << minIdx[1] << "\nMAX_IDX=" << maxIdx[0] << ", " << maxIdx[1] << "\n";
            this->setVerified();
            return;
        }
        this->setVerified();
    }

    IPP_TEST_CYCLE() cv::minMaxIdx(src, pMinVal, pMaxVal, pMinIdx, pMaxIdx);
}

PERF_TEST_P(MinMaxIdxFixture, ippMinMaxIdx_Mask,
            ::testing::Combine(
                IPP_TEST_STHREAD,
                IPP_TEST_SIZES,
                ::testing::Values(CV_8UC1, CV_16UC1, CV_16SC1, CV_32FC1),
                MinMaxType::all()
            ))
{
    const MinMaxIdxParamType params = GetParam();
    const cv::Size size     = get<1>(params);
    const int      type     = get<2>(params);
    const int      minMaxT  = get<3>(params);

    cv::Mat src(size, type);
    cv::Mat mask(size, CV_8UC1);
    double  minVal = 0;
    double  maxVal = 0;
    int     minIdx[2] = {0};
    int     maxIdx[2] = {0};
    double  minValRef = 0;
    double  maxValRef = 0;
    int     minIdxRef[2] = {0};
    int     maxIdxRef[2] = {0};

    double  *pMinVal    = (minMaxT&MIN)?&minVal:NULL;
    double  *pMaxVal    = (minMaxT&MAX)?&maxVal:NULL;
    int     *pMinIdx    = (minMaxT&MIN && minMaxT&IDX)?minIdx:NULL;
    int     *pMaxIdx    = (minMaxT&MAX && minMaxT&IDX)?maxIdx:NULL;
    double  *pMinValRef = (minMaxT&MIN)?&minValRef:NULL;
    double  *pMaxValRef = (minMaxT&MAX)?&maxValRef:NULL;
    int     *pMinIdxRef = (minMaxT&MIN && minMaxT&IDX)?minIdxRef:NULL;
    int     *pMaxIdxRef = (minMaxT&MAX && minMaxT&IDX)?maxIdxRef:NULL;

    declare.in(src, WARMUP_RNG);
    randu(src, 1, 155); // use middle of the range to avoid short-cut optimizations triggering
    randu(mask, 0, 2);

    // Conformance check
    RUN_IPP(cv::minMaxIdx(src, pMinVal, pMaxVal, pMinIdx, pMaxIdx, mask));
    RUN_REF(cv::minMaxIdx(src, pMinValRef, pMaxValRef, pMinIdxRef, pMaxIdxRef, mask));
    {
        if(minVal != minValRef || maxVal != maxValRef
            || minIdx[0] != minIdxRef[0] || minIdx[1] != minIdxRef[1]
            || maxIdx[0] != maxIdxRef[0] || maxIdx[1] != maxIdxRef[1])
        {
            ADD_FAILURE() << "MinMax doesn't match\nEXPECTED:" << "\nMIN=" << minValRef << "\nMAX=" << maxValRef << "\nMIN_IDX=" << minIdxRef[0] << ", " << minIdxRef[1] << "\nMAX_IDX=" << maxIdxRef[0] << ", "  << maxIdxRef[1]
                << "\n\nACTUAL:" << "\nMIN=" << minVal << "\nMAX=" << maxVal << "\nMIN_IDX=" << minIdx[0] << ", "  << minIdx[1] << "\nMAX_IDX=" << maxIdx[0] << ", " << maxIdx[1] << "\n";
            this->setVerified();
            return;
        }
        this->setVerified();
    }

    IPP_TEST_CYCLE() cv::minMaxIdx(src, pMinVal, pMaxVal, pMinIdx, pMaxIdx, mask);
}

///////////// Mat::dot ////////////////////////

typedef tuple<IppTestThreads, Size, MatType> DotParamType;
typedef TestBaseWithParam<DotParamType> DotFixture;

PERF_TEST_P(DotFixture, ippDot,
            ::testing::Combine(
                IPP_TEST_STHREAD,
                IPP_TEST_SIZES,
                ::testing::Values(CV_8UC1, CV_16UC1, CV_16SC1, CV_32SC1, CV_32FC1, CV_64FC1)
            ))
{
    const DotParamType params = GetParam();
    const cv::Size size       = get<1>(params);
    const int      type       = get<2>(params);

    cv::Mat src1(size, type);
    cv::Mat src2(size, type);
    double dotProd;
    double dotProdRef;

    declare.in(src1, src2, WARMUP_RNG);

    // Conformance check
    RUN_IPP(dotProd = src1.dot(src2));
    RUN_REF(dotProdRef = src1.dot(src2));
    {
        Mat dst(Size(1,1), CV_64F, &dotProd, 8);
        Mat dstRef(Size(1,1), CV_64F, &dotProdRef, 8);
        VERIFY_DATA(dst, dstRef, 0, 2e-6);
    }

    IPP_TEST_CYCLE() dotProd = src1.dot(src2);
}

///////////// Sort ////////////////////////

enum
{
    ROWS,
    COLUMNS
};

enum
{
    ASCEND,
    DESCEND
};

CV_ENUM(SortType, ROWS, COLUMNS)
CV_ENUM(SortDir, ASCEND, DESCEND)

typedef tuple<IppTestThreads, Size, MatType, SortType, SortDir> SortParamType;
typedef TestBaseWithParam<SortParamType> SortFixture;

PERF_TEST_P(SortFixture, ippSort,
            ::testing::Combine(
                IPP_TEST_STHREAD,
                IPP_TEST_SIZES,
                ::testing::Values(CV_8UC1, CV_16UC1, CV_16SC1, CV_32SC1, CV_32FC1, CV_64FC1),
                SortType::all(),
                SortDir::all()
            ))
{
    const SortParamType params = GetParam();
    const cv::Size size       = get<1>(params);
    const int      type       = get<2>(params);
    const int      flags      = ((get<3>(params)==ROWS)?SORT_EVERY_ROW:SORT_EVERY_COLUMN)|((get<4>(params)==ASCEND)?SORT_ASCENDING:SORT_DESCENDING);

    cv::Mat src(size, type);
    cv::Mat dst(size, type);
    cv::Mat dstRef(size, type);

    declare.in(src, WARMUP_RNG).out(dst);

    // Conformance check
    RUN_IPP(cv::sort(src, dst, flags));
    RUN_REF(cv::sort(src, dstRef, flags));
    VERIFY_DATA(dst, dstRef, 0, 0);

    IPP_TEST_CYCLE() cv::sort(src, dst, flags);
}

PERF_TEST_P(SortFixture, ippSort_Inplace,
            ::testing::Combine(
                IPP_TEST_STHREAD,
                IPP_TEST_SIZES,
                ::testing::Values(CV_8UC1, CV_16UC1, CV_16SC1, CV_32SC1, CV_32FC1, CV_64FC1),
                SortType::all(),
                SortDir::all()
            ))
{
    const SortParamType params = GetParam();
    const cv::Size size       = get<1>(params);
    const int      type       = get<2>(params);
    const int      flags      = ((get<3>(params)==ROWS)?SORT_EVERY_ROW:SORT_EVERY_COLUMN)|((get<4>(params)==ASCEND)?SORT_ASCENDING:SORT_DESCENDING);

    cv::Mat src(size, type);
    cv::Mat dst(size, type);
    cv::Mat dstRef(size, type);

    declare.in(src, WARMUP_RNG).out(dst);
    src.copyTo(dst);
    src.copyTo(dstRef);

    // Conformance check
    RUN_IPP(cv::sort(dst, dst, flags));
    RUN_REF(cv::sort(dstRef, dstRef, flags));
    VERIFY_DATA(dst, dstRef, 0, 0);

    //IPP_TEST_CYCLE()
    RUN_IPP(;)
    for(; next();)
    {
        src.copyTo(dst);
        startTimer();
        cv::sort(dst, dst, flags);
        stopTimer();
    }
}

template<typename T>
static void reorder_t(Mat &src, Mat &dst, Mat &index, int columnBased)
{
    if(columnBased)
    {
        for(int row = 0; row < dst.rows; row++)
        {
            for(int col = 0; col < dst.cols; col++)
            {
                *dst.ptr<T>(row, col) = *src.ptr<T>(*index.ptr<int>(row, col), col);
            }
        }
    }
    else
    {
        for(int row = 0; row < dst.rows; row++)
        {
            for(int col = 0; col < dst.cols; col++)
            {
                *dst.ptr<T>(row, col) = *src.ptr<T>(row, *index.ptr<int>(row, col));
            }
        }
    }
}

static void reorder(Mat &src, Mat &dst, Mat &index, int flags)
{
    int depth   = dst.depth();
    int columns = (flags&SORT_EVERY_COLUMN);

    switch(depth)
    {
    case CV_8U: return reorder_t<uchar>(src, dst, index, columns);
    case CV_8S: return reorder_t<char>(src, dst, index, columns);
    case CV_16U: return reorder_t<ushort>(src, dst, index, columns);
    case CV_16S: return reorder_t<short>(src, dst, index, columns);
    case CV_32S: return reorder_t<uint>(src, dst, index, columns);
    case CV_32F: return reorder_t<float>(src, dst, index, columns);
    case CV_64F: return reorder_t<double>(src, dst, index, columns);
    default:    return;
    }
}

PERF_TEST_P(SortFixture, ippSortIdx,
            ::testing::Combine(
                IPP_TEST_STHREAD,
                IPP_TEST_SIZES,
                ::testing::Values(CV_8UC1, CV_16UC1, CV_16SC1, CV_32SC1, CV_32FC1),
                SortType::all(),
                SortDir::all()
            ))
{
    const SortParamType params = GetParam();
    const cv::Size size       = get<1>(params);
    const int      type       = get<2>(params);
    const int      flags      = ((get<3>(params)==ROWS)?SORT_EVERY_ROW:SORT_EVERY_COLUMN)|((get<4>(params)==ASCEND)?SORT_ASCENDING:SORT_DESCENDING);

    cv::Mat src(size, type);
    cv::Mat dst(size, CV_32SC1);
    cv::Mat dstRef(size, CV_32SC1);
    cv::Mat dstSorted(size, type);
    cv::Mat dstSortedRef(size, type);

    declare.in(src, WARMUP_RNG).out(dst);

    // Conformance check
    RUN_IPP(cv::sortIdx(src, dst, flags));
    RUN_REF(cv::sortIdx(src, dstRef, flags));
    reorder(src, dstSorted, dst, flags);
    reorder(src, dstSortedRef, dstRef, flags);
    VERIFY_DATA(dstSorted, dstSortedRef, 0, 0);

    IPP_TEST_CYCLE() cv::sortIdx(src, dst, flags);
}

///////////// DFT ////////////////////////

enum
{
    FORWARD = 0,
    INVERSE = DFT_INVERSE
};

enum
{
    FULL = 0,
    BY_ROWS = DFT_ROWS
};

enum
{
    R2R = 0,
    C2R = 1,
    R2C = 2,
    C2C = 3
};

CV_ENUM(DftDirType, FORWARD, INVERSE)
CV_ENUM(DftDataType, R2R, R2C, C2R, C2C)
CV_ENUM(DftSliceType, FULL, BY_ROWS)

typedef tuple<IppTestThreads, Size, MatType, DftDataType, DftSliceType, DftDirType> DftParamType;
typedef TestBaseWithParam<DftParamType> DftFixture;

PERF_TEST_P(DftFixture, ippDft,
            ::testing::Combine(
                IppTestThreads::all(),
                IPP_TEST_SIZES,
                ::testing::Values(CV_32F, CV_64F),
                DftDataType::all(),
                DftSliceType::all(),
                DftDirType::all()
            ))
{
    const DftParamType params = GetParam();
    const cv::Size size       = get<1>(params);
    const int      type       = get<2>(params);
    const int      dataType   = get<3>(params);
          int      flags      = get<4>(params)|get<5>(params);

    // Init IPP extensions
    {
        TS_SET_THREADS(get<0>(params));
    }

    int in_cn = 0, out_cn = 0;
    switch (dataType)
    {
    case R2R: flags |= cv::DFT_REAL_OUTPUT; in_cn = 1; out_cn = 1; break;
    case C2R: flags |= cv::DFT_REAL_OUTPUT; in_cn = 2; out_cn = 2; break;
    case R2C: flags |= cv::DFT_COMPLEX_OUTPUT; in_cn = 1; out_cn = 2; break;
    case C2C: flags |= cv::DFT_COMPLEX_OUTPUT; in_cn = 2; out_cn = 2; break;
    }

    cv::Mat src(size, CV_MAKE_TYPE(type, in_cn));
    cv::Mat dst(size, CV_MAKE_TYPE(type, out_cn));
    cv::Mat dstRef(size, CV_MAKE_TYPE(type, out_cn));

    declare.in(src, WARMUP_RNG).out(dst);

    // Conformance check
    RUN_IPP(cv::dft(src, dst, flags));
    RUN_REF(cv::dft(src, dstRef, flags));
    VERIFY_DATA(dst, dstRef, 1, 1e-6);

    IPP_TEST_CYCLE() cv::dft(src, dst, flags);
}

///////////// DCT ////////////////////////

typedef tuple<IppTestThreads, Size, MatType, DftSliceType, DftDirType> DctParamType;
typedef TestBaseWithParam<DctParamType> DctFixture;

PERF_TEST_P(DctFixture, ippDct,
            ::testing::Combine(
                IppTestThreads::all(),
                IPP_TEST_SIZES,
                ::testing::Values(CV_32F, CV_64F),
                DftSliceType::all(),
                DftDirType::all()
            ))
{
    const DctParamType params = GetParam();
    const cv::Size size       = get<1>(params);
    const int      type       = get<2>(params);
          int      flags      = get<3>(params)|get<4>(params);

    // Init IPP extensions
    {
        TS_SET_THREADS(get<0>(params));
    }

    cv::Mat src(size, type);
    cv::Mat dst(size, type);
    cv::Mat dstRef(size, type);

    declare.in(src, WARMUP_RNG).out(dst);

    // Conformance check
    RUN_IPP(cv::dct(src, dst, flags));
    RUN_REF(cv::dct(src, dstRef, flags));
    VERIFY_DATA(dst, dstRef, 1, 1e-6);

    IPP_TEST_CYCLE() cv::dct(src, dst, flags);
}

///////////// Norm ////////////////////////

CV_ENUM(NormTypeType, NORM_L1, NORM_L2, NORM_L2SQR, NORM_INF)

typedef tuple<IppTestThreads, Size, MatType, NormTypeType> NormParamType;
typedef TestBaseWithParam<NormParamType> NormFixture;

PERF_TEST_P(NormFixture, ippNorm,
            ::testing::Combine(
                IPP_TEST_STHREAD,
                IPP_TEST_SIZES,
                ::testing::Values(CV_8UC1, CV_16UC1, CV_16SC1, CV_32FC1),
                NormTypeType::all()
            ))
{
    const NormParamType params = GetParam();
    const cv::Size size       = get<1>(params);
    const int      type       = get<2>(params);
    const int      normType   = get<3>(params);

    cv::Mat src(size, type);
    double dst;
    double dstRef;

    declare.in(src, WARMUP_RNG);

    // Conformance check
    RUN_IPP(dst = cv::norm(src, normType));
    RUN_REF(dstRef = cv::norm(src, normType));
    {
        Mat _dst(Size(1,1), CV_64F, &dst, 8);
        Mat _dstRef(Size(1,1), CV_64F, &dstRef, 8);
        VERIFY_DATA(_dst, _dstRef, 0, 1e-7);
    }

    IPP_TEST_CYCLE() cv::norm(src, normType);
}

PERF_TEST_P(NormFixture, ippNorm_Mask,
            ::testing::Combine(
                IPP_TEST_STHREAD,
                IPP_TEST_SIZES,
                ::testing::Values(CV_8UC1, CV_8UC3, CV_16UC1, CV_16UC3, CV_32FC1, CV_32FC3),
                NormTypeType::all()
            ))
{
    const NormParamType params = GetParam();
    const cv::Size size       = get<1>(params);
    const int      type       = get<2>(params);
    const int      normType   = get<3>(params);

    cv::Mat src(size, type);
    cv::Mat mask(size, CV_8U);
    double dst;
    double dstRef;

    declare.in(src, WARMUP_RNG);
    randu(mask, 0, 2);

    // Conformance check
    RUN_IPP(dst = cv::norm(src, normType, mask));
    RUN_REF(dstRef = cv::norm(src, normType, mask));
    {
        Mat _dst(Size(1,1), CV_64F, &dst, 8);
        Mat _dstRef(Size(1,1), CV_64F, &dstRef, 8);
        VERIFY_DATA(_dst, _dstRef, 0, 1e-7);
    }

    IPP_TEST_CYCLE() cv::norm(src, normType, mask);
}

enum NORM_REL
{
    ABS = 0,
    REL = NORM_RELATIVE
};

CV_ENUM(NormTypeRel, ABS, REL)

typedef tuple<IppTestThreads, Size, MatType, NormTypeType, NormTypeRel> Norm2ParamType;
typedef TestBaseWithParam<Norm2ParamType> Norm2Fixture;

PERF_TEST_P(Norm2Fixture, ippNorm2,
            ::testing::Combine(
                IPP_TEST_STHREAD,
                IPP_TEST_SIZES,
                ::testing::Values(CV_8UC1, CV_16UC1, CV_16SC1, CV_32FC1),
                NormTypeType::all(),
                NormTypeRel::all()
            ))
{
    const Norm2ParamType params = GetParam();
    const cv::Size size       = get<1>(params);
    const int      type       = get<2>(params);
    const int      normType   = get<3>(params)|get<4>(params);

    cv::Mat src1(size, type);
    cv::Mat src2(size, type);
    double dst;
    double dstRef;

    declare.in(src1, src2, WARMUP_RNG);

    // Conformance check
    RUN_IPP(dst = cv::norm(src1, src2, normType));
    RUN_REF(dstRef = cv::norm(src1, src2, normType));
    {
        Mat _dst(Size(1,1), CV_64F, &dst, 8);
        Mat _dstRef(Size(1,1), CV_64F, &dstRef, 8);
        VERIFY_DATA(_dst, _dstRef, 0, 1e-7);
    }

    IPP_TEST_CYCLE() cv::norm(src1, src2, normType);
}

PERF_TEST_P(Norm2Fixture, ippNorm2_Mask,
            ::testing::Combine(
                IPP_TEST_STHREAD,
                IPP_TEST_SIZES,
                ::testing::Values(CV_8UC1, CV_8UC3, CV_16UC1, CV_16UC3, CV_32FC1, CV_32FC3),
                NormTypeType::all(),
                NormTypeRel::all()
            ))
{
    const Norm2ParamType params = GetParam();
    const cv::Size size       = get<1>(params);
    const int      type       = get<2>(params);
    const int      normType   = get<3>(params)|get<4>(params);

    cv::Mat src1(size, type);
    cv::Mat src2(size, type);
    cv::Mat mask(size, CV_8U);
    double dst;
    double dstRef;

    declare.in(src1, src2, WARMUP_RNG);
    randu(mask, 0, 2);

    // Conformance check
    RUN_IPP(dst = cv::norm(src1, src2, normType, mask));
    RUN_REF(dstRef = cv::norm(src1, src2, normType, mask));
    {
        Mat _dst(Size(1,1), CV_64F, &dst, 8);
        Mat _dstRef(Size(1,1), CV_64F, &dstRef, 8);
        VERIFY_DATA(_dst, _dstRef, 0, 1e-7);
    }

    IPP_TEST_CYCLE() cv::norm(src1, src2, normType, mask);
}

///////////// Reduce ////////////////////////

enum REDUCE
{
    ROW = 0,
    COL = 1
};

CV_ENUM(ReduceDim, ROW, COL)
CV_ENUM(ReduceSameType, REDUCE_MAX, REDUCE_MIN)

typedef tuple<IppTestThreads, Size, MatType, ReduceSameType, ReduceDim> ReduceSameParamType;
typedef TestBaseWithParam<ReduceSameParamType> ReduceSameFixture;

PERF_TEST_P(ReduceSameFixture, ippReduce_Same,
            ::testing::Combine(
                IPP_TEST_STHREAD,
                IPP_TEST_SIZES,
                ::testing::Values(CV_8UC1, CV_8UC3, CV_8UC4, CV_16UC1, CV_16UC3, CV_16UC4, CV_16SC1, CV_16SC3, CV_16SC4, CV_32FC1, CV_32FC3, CV_32FC4),
                ReduceSameType::all(),
                ReduceDim::all()
            ))
{
    const ReduceSameParamType params = GetParam();
    const Size     size       = get<1>(params);
    const int      type       = get<2>(params);
    const int      reduceType = get<3>(params);
    const int      reduceDim  = get<4>(params);
          Size     dstSize    = Size(reduceDim==ROW?size.width:1, reduceDim==ROW?1:size.height);

    Mat src(size, type);
    Mat dst(dstSize, type);
    Mat dstRef(dstSize, type);

    declare.in(src, WARMUP_RNG).out(dst);

    // Conformance check
    RUN_IPP(cv::reduce(src, dst, reduceDim, reduceType));
    RUN_REF(cv::reduce(src, dstRef, reduceDim, reduceType));
    VERIFY_DATA(dst, dstRef, 0, 0);

    IPP_TEST_CYCLE_MULTIRUN(10) cv::reduce(src, dst, reduceDim, reduceType);
}

CV_ENUM(ReduceDiffType, REDUCE_SUM, REDUCE_AVG)

typedef tuple<IppTestThreads, Size, MatPairs, ReduceDiffType, ReduceDim> ReduceDiffParamType;
typedef TestBaseWithParam<ReduceDiffParamType> ReduceDiffFixture;

PERF_TEST_P(ReduceDiffFixture, ippReduce_Diff,
            ::testing::Combine(
                IPP_TEST_STHREAD,
                IPP_TEST_SIZES,
                ::testing::Values(CV_8U_32S_C1, CV_8U_32S_C3, CV_8U_32S_C4, CV_8U_32F_C1, CV_8U_32F_C3, CV_8U_32F_C4, CV_8U_64F_C1, CV_8U_64F_C3, CV_8U_64F_C4,
                                  CV_16U_32F_C1, CV_16U_32F_C3, CV_16U_32F_C4, CV_16U_64F_C1, CV_16U_64F_C3, CV_16U_64F_C4,
                                  CV_16S_32F_C1, CV_16S_32F_C3, CV_16S_32F_C4, CV_16S_64F_C1, CV_16S_64F_C3, CV_16S_64F_C4,
                                  CV_32F_32F_C1, CV_32F_32F_C3, CV_32F_32F_C4, CV_32F_64F_C1, CV_32F_64F_C3, CV_32F_64F_C4,
                                  CV_64F_64F_C1, CV_64F_64F_C3, CV_64F_64F_C4),
                ReduceDiffType::all(),
                ReduceDim::all()
            ))
{
    const ReduceDiffParamType params = GetParam();
    const Size     size       = get<1>(params);
    const int      srcType    = CV_MAT_PAIR_GET1(get<2>(params));
    const int      dstType    = CV_MAT_PAIR_GET2(get<2>(params));
    const int      reduceType = get<3>(params);
    const int      reduceDim  = get<4>(params);
          Size     dstSize    = Size(reduceDim==ROW?size.width:1, reduceDim==ROW?1:size.height);

    Mat src(size, srcType);
    Mat dst(dstSize, dstType);
    Mat dstRef(dstSize, dstType);

    declare.in(src, WARMUP_RNG).out(dst);

    // Conformance check
    RUN_IPP(cv::reduce(src, dst, reduceDim, reduceType, dstType));
    RUN_REF(cv::reduce(src, dstRef, reduceDim, reduceType, dstType));
    VERIFY_DATA(dst, dstRef, 0, 0);

    IPP_TEST_CYCLE_MULTIRUN(10) cv::reduce(src, dst, reduceDim, reduceType, dstType);
}

///////////// Transpose ////////////////////////

typedef tuple<IppTestThreads, Size, MatType> TransposeParamType;
typedef TestBaseWithParam<TransposeParamType> TransposeFixture;

PERF_TEST_P(TransposeFixture, ippTranspose,
            ::testing::Combine(
                IPP_TEST_STHREAD,
                IPP_TEST_SIZES,
                ::testing::Values(CV_8UC1, CV_8UC3, CV_8UC4, CV_16UC1, CV_16UC3, CV_16UC4, CV_32FC1, CV_32FC3, CV_32FC4)
            ))
{
    const TransposeParamType params = GetParam();
    const Size     size       = get<1>(params);
    const int      type       = get<2>(params);

    Mat src(size, type);
    Mat dst(size, type);
    Mat dstRef(size, type);

    declare.in(src, WARMUP_RNG).out(dst);

    // Conformance check
    RUN_IPP(cv::transpose(src, dst));
    RUN_REF(cv::transpose(src, dstRef));
    VERIFY_DATA(dst, dstRef, 0, 0);

    IPP_TEST_CYCLE_MULTIRUN(10) cv::transpose(src, dst);
}

PERF_TEST_P(TransposeFixture, ippTranspose_Inplace,
            ::testing::Combine(
                IPP_TEST_STHREAD,
                IPP_TEST_RECT_SIZES,
                ::testing::Values(CV_8UC1, CV_8UC3, CV_8UC4, CV_16UC1, CV_16UC3, CV_16UC4, CV_32FC1, CV_32FC3, CV_32FC4)
            ))
{
    const TransposeParamType params = GetParam();
    const Size     size       = get<1>(params);
    const int      type       = get<2>(params);

    Mat src(size, type);
    Mat dst(size, type);
    Mat dstRef(size, type);

    declare.in(src, WARMUP_RNG).out(dst);
    src.copyTo(dst);
    src.copyTo(dstRef);

    // Conformance check
    RUN_IPP(cv::transpose(dst, dst));
    RUN_REF(cv::transpose(dstRef, dstRef));
    VERIFY_DATA(dst, dstRef, 0, 0);

    IPP_TEST_CYCLE_MULTIRUN(10) cv::transpose(dst, dst);
}

///////////// cv::countNonZero ////////////////////////

typedef tuple<IppTestThreads, Size, MatType> CountNonZeroParamType;
typedef TestBaseWithParam<CountNonZeroParamType> CountNonZeroFixture;

PERF_TEST_P(CountNonZeroFixture, ippCountNonZero,
            ::testing::Combine(
                IPP_TEST_STHREAD,
                IPP_TEST_SIZES,
                ::testing::Values(CV_8UC1, CV_32FC1)
            ))
{
    const CountNonZeroParamType params = GetParam();
    const Size     size       = get<1>(params);
    const int      type       = get<2>(params);

    Mat src(size, type);
    int count    = 0;
    int countRef = 0;

    declare.in(src, WARMUP_RNG);

    // Conformance check
    RUN_IPP(count = cv::countNonZero(src));
    RUN_REF(countRef = cv::countNonZero(src));
    {
        if(count != countRef)
        {
            ADD_FAILURE() << "Counter mismatch\nEXPECTED: " << countRef << "\nACTUAL: " << count << "\n";
            this->setVerified();
            return;
        }
        this->setVerified();
    }

    IPP_TEST_CYCLE() cv::countNonZero(src);
}

///////////// Mean ////////////////////////

typedef tuple<IppTestThreads, Size, MatType> MeanParamType;
typedef TestBaseWithParam<MeanParamType> MeanFixture;

PERF_TEST_P(MeanFixture, ippMean,
            ::testing::Combine(
                IPP_TEST_STHREAD,
                IPP_TEST_SIZES,
                ::testing::Values(CV_8UC1, CV_8UC3, CV_8UC4, CV_16UC1, CV_16UC3, CV_16UC4, CV_16SC1, CV_16SC3, CV_16SC4, CV_32FC1, CV_32FC3, CV_32FC4)
            ))
{
    const MeanParamType params = GetParam();
    const Size     size       = get<1>(params);
    const int      type       = get<2>(params);

    Mat src(size, type);
    Scalar dst    = 0;
    Scalar dstRef = 0;

    declare.in(src, WARMUP_RNG);

    // Conformance check
    RUN_IPP(dst = cv::mean(src));
    RUN_REF(dstRef = cv::mean(src));
    {
        Mat _dst(Size(1,1), CV_64FC4, &dst, 32);
        Mat _dstRef(Size(1,1), CV_64FC4, &dstRef, 32);
        VERIFY_DATA(_dst, _dstRef, 0, 3e-6);
    }

    IPP_TEST_CYCLE_MULTIRUN(10) cv::mean(src);
}

PERF_TEST_P(MeanFixture, ippMean_Mask,
            ::testing::Combine(
                IPP_TEST_STHREAD,
                IPP_TEST_SIZES,
                ::testing::Values(CV_8UC1, CV_8UC3, CV_8UC4, CV_16UC1, CV_16UC3, CV_16UC4, CV_32FC1, CV_32FC3, CV_32FC4)
            ))
{
    const MeanParamType params = GetParam();
    const Size     size       = get<1>(params);
    const int      type       = get<2>(params);

    Mat src(size, type);
    Mat mask(size, CV_8U);
    Scalar dst    = 0;
    Scalar dstRef = 0;

    declare.in(src, WARMUP_RNG);
    randu(mask, 0, 2);

    // Conformance check
    RUN_IPP(dst = cv::mean(src, mask));
    RUN_REF(dstRef = cv::mean(src, mask));
    {
        Mat _dst(Size(1,1), CV_64FC4, &dst, 32);
        Mat _dstRef(Size(1,1), CV_64FC4, &dstRef, 32);
        VERIFY_DATA(_dst, _dstRef, 0, 1e-5);
    }

    IPP_TEST_CYCLE_MULTIRUN(10) cv::mean(src, mask);
}

///////////// MeanStdDev ////////////////////////

PERF_TEST_P(MeanFixture, ippMeanStdDev,
            ::testing::Combine(
                IPP_TEST_STHREAD,
                IPP_TEST_SIZES,
                ::testing::Values(CV_8UC1, CV_8UC3, CV_8UC4, CV_16UC1, CV_16UC3, CV_16UC4, CV_32FC1, CV_32FC3, CV_32FC4)
            ))
{
    const MeanParamType params = GetParam();
    const Size     size       = get<1>(params);
    const int      type       = get<2>(params);

    Mat src(size, type);
    Scalar dst     = 0;
    Scalar dstm    = 0;
    Scalar dstRef  = 0;
    Scalar dstmRef = 0;

    declare.in(src, WARMUP_RNG);

    // Conformance check
    RUN_IPP(cv::meanStdDev(src, dstm, dst));
    RUN_REF(cv::meanStdDev(src, dstmRef, dstRef));
    {
        Mat _dst(Size(1,1), CV_64FC4, &dst, 32);
        Mat _dstm(Size(1,1), CV_64FC4, &dstm, 32);
        Mat _dstRef(Size(1,1), CV_64FC4, &dstRef, 32);
        Mat _dstmRef(Size(1,1), CV_64FC4, &dstmRef, 32);
        VERIFY_DATA(_dst, _dstRef, 0, 1e-7);
        VERIFY_DATA(_dstm, _dstmRef, 0, 1e-5);
    }

    IPP_TEST_CYCLE_MULTIRUN(10) cv::meanStdDev(src, dstm, dst);
}

PERF_TEST_P(MeanFixture, ippMeanStdDev_Mask,
            ::testing::Combine(
                IPP_TEST_STHREAD,
                IPP_TEST_SIZES,
                ::testing::Values(CV_8UC1, CV_8UC3, CV_8UC4, CV_16UC1, CV_16UC3, CV_16UC4, CV_32FC1, CV_32FC3, CV_32FC4)
            ))
{
    const MeanParamType params = GetParam();
    const Size     size       = get<1>(params);
    const int      type       = get<2>(params);

    Mat src(size, type);
    Mat mask(size, CV_8U);
    Scalar dst     = 0;
    Scalar dstm    = 0;
    Scalar dstRef  = 0;
    Scalar dstmRef = 0;

    declare.in(src, WARMUP_RNG);
    randu(mask, 0, 2);

    // Conformance check
    RUN_IPP(cv::meanStdDev(src, dstm, dst, mask));
    RUN_REF(cv::meanStdDev(src, dstmRef, dstRef, mask));
    {
        Mat _dst(Size(1,1), CV_64FC4, &dst, 32);
        Mat _dstm(Size(1,1), CV_64FC4, &dstm, 32);
        Mat _dstRef(Size(1,1), CV_64FC4, &dstRef, 32);
        Mat _dstmRef(Size(1,1), CV_64FC4, &dstmRef, 32);
        VERIFY_DATA(_dst, _dstRef, 0, 1e-7);
        VERIFY_DATA(_dstm, _dstmRef, 0, 1e-5);
    }

    IPP_TEST_CYCLE_MULTIRUN(10) cv::meanStdDev(src, dstm, dst, mask);
}

///////////// Sum ////////////////////////

PERF_TEST_P(MeanFixture, ippSum,
            ::testing::Combine(
                IPP_TEST_STHREAD,
                IPP_TEST_SIZES,
                ::testing::Values(CV_8UC1, CV_8UC3, CV_8UC4, CV_16UC1, CV_16UC3, CV_16UC4, CV_16SC1, CV_16SC3, CV_16SC4, CV_32FC1, CV_32FC3, CV_32FC4)
            ))
{
    const MeanParamType params = GetParam();
    const Size     size       = get<1>(params);
    const int      type       = get<2>(params);

    Mat src(size, type);
    Scalar dst    = 0;
    Scalar dstRef = 0;

    declare.in(src, WARMUP_RNG);

    // Conformance check
    RUN_IPP(dst = cv::sum(src));
    RUN_REF(dstRef = cv::sum(src));
    {
        Mat _dst(Size(1,1), CV_64FC4, &dst, 32);
        Mat _dstRef(Size(1,1), CV_64FC4, &dstRef, 32);
        VERIFY_DATA(_dst, _dstRef, 0, 3e-6);
    }

    IPP_TEST_CYCLE_MULTIRUN(10) cv::sum(src);
}

} // namespace
#endif // HAVE_IPP

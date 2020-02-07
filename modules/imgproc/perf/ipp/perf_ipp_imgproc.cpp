#include "../perf_precomp.hpp"

#ifdef HAVE_IPP
#include "opencv2/ts/ipp_perf.hpp"

namespace opencv_test
{
using namespace perf;

enum MaskEnum
{
    MASK_TRUE  = 0,
    MASK_3x3   = 3,
    MASK_5x5   = 5
};

/////////// cv::copyMakeBorder //////////////////////
#if !IPP_DISABLE_PERF_COPYMAKE
CV_ENUM(IppBorderCopy, BORDER_CONSTANT, BORDER_REPLICATE, BORDER_REFLECT_101)

typedef tuple<IppTestThreads, Size, MatType, IppBorderCopy, int> CopyMakeBorderParamType;
typedef TestBaseWithParam<CopyMakeBorderParamType> CopyMakeBorderFixture;

PERF_TEST_P(CopyMakeBorderFixture, ippCopyMakeBorder,
            ::testing::Combine(
                IPP_TEST_STHREAD,
                IPP_TEST_SIZES,
                ::testing::Values(CV_8UC1, CV_8UC3, CV_8UC4, CV_16UC1, CV_16UC3, CV_16UC4, CV_16SC1, CV_16SC3, CV_16SC4, CV_32SC1, CV_32SC3, CV_32SC4, CV_32FC1, CV_32FC3, CV_32FC4),
                IppBorderCopy::all(),
                ::testing::Values(1, 2, 6)
            ))
{
    const CopyMakeBorderParamType params = GetParam();
    const cv::Size size       = get<1>(params);
    const int      type       = get<2>(params);
    const int      borderType = get<3>(params);
    const int      borderSize = get<4>(params);

    cv::Mat dst(size + cv::Size(borderSize*2, borderSize*2), type);
    cv::Mat dstRef(size + cv::Size(borderSize*2, borderSize*2), type);
    cv::Mat src(size, type);

    declare.in(src, WARMUP_RNG).out(dst);

    // Conformance check
    RUN_IPP(cv::copyMakeBorder(src, dst, borderSize, borderSize, borderSize, borderSize, borderType|cv::BORDER_ISOLATED, cv::Scalar(-50)));
    RUN_REF(cv::copyMakeBorder(src, dstRef, borderSize, borderSize, borderSize, borderSize, borderType|cv::BORDER_ISOLATED, cv::Scalar(-50)));
    VERIFY_DATA(dst, dstRef, 0, 0);

    IPP_TEST_CYCLE_MULTIRUN(10) cv::copyMakeBorder(src, dst, borderSize, borderSize, borderSize, borderSize, borderType|cv::BORDER_ISOLATED, cv::Scalar(-50));
}

PERF_TEST_P(CopyMakeBorderFixture, ippCopyMakeBorder_Inplace,
            ::testing::Combine(
                IPP_TEST_STHREAD,
                IPP_TEST_SIZES,
                ::testing::Values(CV_8UC1, CV_8UC3, CV_8UC4, CV_16UC1, CV_16UC3, CV_16UC4, CV_16SC1, CV_16SC3, CV_16SC4, CV_32SC1, CV_32SC3, CV_32SC4, CV_32FC1, CV_32FC3, CV_32FC4),
                IppBorderCopy::all(),
                ::testing::Values(1, 2, 6)
            ))
{
    const CopyMakeBorderParamType params = GetParam();
    const cv::Size size       = get<1>(params);
    const int      type       = get<2>(params);
    const int      borderType = get<3>(params);
    const int      borderSize = get<4>(params);

    cv::Mat dst(size + cv::Size(borderSize*2, borderSize*2), type);
    cv::Mat dstRef(size + cv::Size(borderSize*2, borderSize*2), type);
    cv::Mat src    = cv::Mat(dst, cv::Rect(borderSize, borderSize, dst.cols - borderSize*2, dst.rows - borderSize*2));
    cv::Mat srcRef = cv::Mat(dstRef, cv::Rect(borderSize, borderSize, dst.cols - borderSize*2, dst.rows - borderSize*2));

    declare.in(src, WARMUP_RNG);
    src.copyTo(srcRef);

    // Conformance check
    RUN_IPP(cv::copyMakeBorder(src, dst, borderSize, borderSize, borderSize, borderSize, borderType|cv::BORDER_ISOLATED, cv::Scalar(-50)));
    RUN_REF(cv::copyMakeBorder(srcRef, dstRef, borderSize, borderSize, borderSize, borderSize, borderType|cv::BORDER_ISOLATED, cv::Scalar(-50)));
    VERIFY_DATA(dst, dstRef, 0, 0);

    IPP_TEST_CYCLE_MULTIRUN(500) cv::copyMakeBorder(src, dst, borderSize, borderSize, borderSize, borderSize, borderType|cv::BORDER_ISOLATED, cv::Scalar(-50));
}
#endif

/////////// cv::resize //////////////////////

CV_ENUM(IppResizeInter, INTER_NEAREST, INTER_LINEAR, INTER_CUBIC, INTER_LANCZOS4, INTER_AREA)

typedef tuple<IppTestThreads, Size, MatType, double, IppResizeInter> ResizeParamType;
typedef TestBaseWithParam<ResizeParamType> ResizeFixture;

PERF_TEST_P(ResizeFixture, ippResize_General,
            ::testing::Combine(
                IppTestThreads::all(),
                IPP_TEST_RESIZE_SIZES,
                ::testing::Values(CV_8UC1, CV_8UC3, CV_8UC4, CV_16UC1, CV_16UC3, CV_16UC4, CV_16SC1, CV_16SC3, CV_16SC4, CV_32FC1, CV_32FC3, CV_32FC4, CV_64FC1, CV_64FC3, CV_64FC4),
                ::testing::Values(0.777, 1.333, 2, 0.5, 0.25),
                ::testing::Values(INTER_LINEAR, INTER_CUBIC, INTER_LANCZOS4)
                )
            )
{
    const ResizeParamType params = GetParam();
    const Size     srcSize    = get<1>(params);
    const int      type       = get<2>(params);
    const double   scale      = get<3>(params);
    const Size     dstSize    = Size(saturate_cast<int>(srcSize.width*scale), saturate_cast<int>(srcSize.height*scale));
    const int      inter      = get<4>(params);

    // Init IPP extensions
    {
        if(CV_MAT_DEPTH(type) == CV_64F && inter != INTER_LINEAR)
            TS_SKIP_TEST;

        if(CV_MAT_DEPTH(type) == CV_8U && inter == INTER_LINEAR)
            TS_SET_NOT_EXACT;

        TS_SET_THREADS(get<0>(params));
    }

    Mat src = tsLoadImage(getDataPath("ipp/boat2.jpg"), type, srcSize);
    Mat dst(dstSize, type);
    Mat dstRef(dstSize, type);
    declare.in(src, WARMUP_READ).out(dst);

    // Conformance check
    RUN_IPP(cv::resize(src, dst, dstSize, 0, 0, inter));
    RUN_REF(cv::resize(src, dstRef, dstSize, 0, 0, inter));
    VERIFY_DATA(dst, dstRef, 4, 6e-5);

    // Performance loop
    IPP_TEST_CYCLE() cv::resize(src, dst, dstSize, 0, 0, inter);
}

PERF_TEST_P(ResizeFixture, ippResize_Affine,
            ::testing::Combine(
                IppTestThreads::all(),
                IPP_TEST_RESIZE_SIZES,
                ::testing::Values(CV_8UC1, CV_8UC3, CV_8UC4, CV_16UC1, CV_16UC3, CV_16UC4, CV_16SC1, CV_16SC3, CV_16SC4, CV_32FC1, CV_32FC3, CV_32FC4),
                ::testing::Values(0.777, 1.333, 2, 0.5, 0.25),
                ::testing::Values(INTER_LINEAR, INTER_CUBIC)
                )
            )
{
    const ResizeParamType params = GetParam();
    const Size     srcSize    = get<1>(params);
    const int      type       = get<2>(params);
    const double   scale      = get<3>(params);
    const Size     dstSize    = Size(saturate_cast<int>(srcSize.width*scale), saturate_cast<int>(srcSize.height*scale));
    const int      inter      = get<4>(params);

    // Init IPP extensions
    {
        if(CV_MAT_DEPTH(type) == CV_8U && inter == INTER_LINEAR)
            TS_SET_NOT_EXACT;

        TS_SET_THREADS(get<0>(params));
    }

    Mat src = tsLoadImage(getDataPath("ipp/boat2.jpg"), type, srcSize);
    Mat dst(dstSize, type);
    Mat dstRef(dstSize, type);
    declare.in(src, WARMUP_READ).out(dst);

    // Conformance check
    RUN_IPP(cv::resize(src, dst, Size(), scale, scale, inter));
    RUN_REF(cv::resize(src, dstRef, Size(), scale, scale, inter));
    VERIFY_DATA(dst, dstRef, 4, 6e-5);

    // Performance loop
    IPP_TEST_CYCLE() cv::resize(src, dst, Size(), scale, scale, inter);
}

/////////// cv::moments //////////////////////

typedef tuple<IppTestThreads, Size, MatType> MomentsParamType;
typedef TestBaseWithParam<MomentsParamType> MomentsFixture;

PERF_TEST_P(MomentsFixture, ippMoments,
            ::testing::Combine(
                IPP_TEST_STHREAD,
                IPP_TEST_SIZES,
                ::testing::Values(CV_8UC1, CV_16UC1, CV_32FC1)
                )
            )
{
    const MomentsParamType params = GetParam();
    const Size     size       = get<1>(params);
    const int      type       = get<2>(params);

    cv::Moments m;
    cv::Mat mMat(1, (int)sizeof(cv::Moments) / sizeof(double), CV_64F, (void*)&m);
    cv::Moments mRef;
    cv::Mat mMatRef(1, (int)sizeof(cv::Moments) / sizeof(double), CV_64F, (void*)&mRef);

    Mat src = tsLoadImage(getDataPath("ipp/boat2.jpg"), type, size);
    declare.in(src, WARMUP_READ);

    // Conformance check
    RUN_IPP(m = cv::moments(src));
    RUN_REF(mRef = cv::moments(src));
    VERIFY_DATA_PRES(mMat, mMatRef, 0, 0, 1e-14);

    IPP_TEST_CYCLE() cv::moments(src);
}

/////////// cv::Canny //////////////////////

enum
{
    CANNY_NORM_L1,
    CANNY_NORM_L2
};

CV_ENUM(IppCannyNormType, CANNY_NORM_L1, CANNY_NORM_L2)

typedef tuple<IppTestThreads, Size, MatType, int, IppCannyNormType> CannyDirectParamType;
typedef TestBaseWithParam<CannyDirectParamType> CannyDirectFixture;

PERF_TEST_P(CannyDirectFixture, ippCannyDirect,
            ::testing::Combine(
                IppTestThreads::all(),
                IPP_TEST_SIZES,
                ::testing::Values(CV_8UC1),
                ::testing::Values(3, 5),
                IppCannyNormType::all()
                )
            )
{
    const CannyDirectParamType params = GetParam();
    const Size     size       = get<1>(params);
    const int      type       = get<2>(params);
    const int      kernelSize = get<3>(params);
    const bool     l2Norm     = (get<4>(params)==CANNY_NORM_L2)?1:0;

    double highTres = tsTypeGetMax(type)*0.8;
    double lowTres  = tsTypeGetMax(type)*0.5;

    // Init IPP extensions
    {
        TS_SET_THREADS(get<0>(params));
    }

    Mat src = tsLoadImage(getDataPath("ipp/boat2.jpg"), type, size);
    Mat dst(size, type);
    Mat dstRef(size, type);
    declare.in(src, WARMUP_READ).out(dst);

    // Conformance check
    RUN_IPP(cv::Canny(src, dst, lowTres, highTres, kernelSize, l2Norm));
    RUN_REF(cv::Canny(src, dstRef, lowTres, highTres, kernelSize, l2Norm));
    VERIFY_DATA_MATCHES(dst, dstRef, 2);

    IPP_TEST_CYCLE() cv::Canny(src, dst, lowTres, highTres, kernelSize, l2Norm);
}

typedef tuple<IppTestThreads, Size, MatPairs, IppCannyNormType> CannyDerivParamType;
typedef TestBaseWithParam<CannyDerivParamType> CannyDerivFixture;

PERF_TEST_P(CannyDerivFixture, ippCannyDeriv,
            ::testing::Combine(
                IppTestThreads::all(),
                IPP_TEST_SIZES,
                ::testing::Values(CV_16S_8U_C1),
                IppCannyNormType::all()
                )
            )
{
    const CannyDerivParamType params = GetParam();
    const Size     size       = get<1>(params);
    const int      srcType    = CV_MAT_PAIR_GET1(get<2>(params));
    const int      dstType    = CV_MAT_PAIR_GET2(get<2>(params));
    const bool     l2Norm     = (get<3>(params)==CANNY_NORM_L2)?1:0;

    double highTres = tsTypeGetMax(dstType)*0.8;
    double lowTres  = tsTypeGetMax(dstType)*0.5;

    // Init IPP extensions
    {
        TS_SET_THREADS(get<0>(params));
    }

    Mat orig = tsLoadImage(getDataPath("ipp/boat2.jpg"), srcType, size);
    Mat srcX;
    Mat srcY;
    Sobel(orig, srcX, srcType, 1, 0);
    Sobel(orig, srcY, srcType, 0, 1);

    Mat dst(size, dstType);
    Mat dstRef(size, dstType);
    declare.in(srcX, srcY, WARMUP_READ).out(dst);

    // Conformance check
    RUN_IPP(cv::Canny(srcX, srcY, dst, lowTres, highTres, l2Norm));
    RUN_REF(cv::Canny(srcX, srcY, dst, lowTres, highTres, l2Norm));
    VERIFY_DATA_MATCHES(dst, dstRef, 2);

    IPP_TEST_CYCLE() cv::Canny(srcX, srcY, dst, lowTres, highTres, l2Norm);
}

/////////// cv::getRectSubPix //////////////////////

typedef tuple<IppTestThreads, Size, MatType, MatType> GetRectSubPixParamType;
typedef TestBaseWithParam<GetRectSubPixParamType> GetRectSubPixFixture;

PERF_TEST_P(GetRectSubPixFixture, ippGetRectSubPix,
            ::testing::Combine(
                IPP_TEST_STHREAD,
                IPP_TEST_SIZES,
                ::testing::Values(CV_8UC1, CV_32FC1),
                ::testing::Values(CV_8UC1, CV_32FC1)
                )
            )
{
    const GetRectSubPixParamType params = GetParam();
    const Size     size       = get<1>(params);
    const int      srcType    = get<2>(params);
    const int      dstType    = get<3>(params);
    const Size     rectSize(800, 600);
    const Point2f  rectCenter((float)100.33, (float)100.77);

    if(CV_MAT_DEPTH(dstType) < CV_MAT_DEPTH(srcType))
        throw ::perf::TestBase::PerfSkipTestException();

    Mat src(size, srcType);
    Mat dst(rectSize, dstType);
    Mat dstRef(rectSize, dstType);
    declare.in(src, WARMUP_RNG).out(dst);

    // Conformance check
    RUN_IPP(cv::getRectSubPix(src, rectSize, rectCenter, dst, dstType));
    RUN_REF(cv::getRectSubPix(src, rectSize, rectCenter, dstRef, dstType));
    VERIFY_DATA(dst, dstRef, 1, 3e-7);

    IPP_TEST_CYCLE() cv::getRectSubPix(src, rectSize, rectCenter, dst, dstType);
}

/////////// cv::accumulate //////////////////////

typedef tuple<IppTestThreads, Size, MatPairs> AccumulateParamType;
typedef TestBaseWithParam<AccumulateParamType> AccumulateFixture;

PERF_TEST_P(AccumulateFixture, ippAccumulate,
            ::testing::Combine(
                IPP_TEST_STHREAD,
                IPP_TEST_SIZES,
                ::testing::Values(CV_8U_32F_C1, CV_8U_32F_C3, CV_16U_32F_C1, CV_16U_32F_C3, CV_32F_32F_C1, CV_32F_32F_C3,
                    CV_8U_64F_C1, CV_8U_64F_C3, CV_16U_64F_C1, CV_16U_64F_C3, CV_32F_64F_C1, CV_32F_64F_C3)
                )
            )
{
    const AccumulateParamType params = GetParam();
    const Size     size       = get<1>(params);
    const int      srcType    = CV_MAT_PAIR_GET1(get<2>(params));
    const int      dstType    = CV_MAT_PAIR_GET2(get<2>(params));

    Mat src(size, srcType);
    Mat dst(size, dstType);
    Mat dstRef(size, dstType);
    declare.in(src, WARMUP_RNG).out(dst, dstRef);

    // Conformance check
    for(int i = 0; i < 10; i++)
    {
        RUN_IPP(cv::accumulate(src, dst));
        RUN_REF(cv::accumulate(src, dstRef));
    }
    VERIFY_DATA(dst, dstRef, 0, 4e-7);

    IPP_TEST_CYCLE() cv::accumulate(src, dst);
}

PERF_TEST_P(AccumulateFixture, ippAccumulate_Mask,
            ::testing::Combine(
                IPP_TEST_STHREAD,
                IPP_TEST_SIZES,
                ::testing::Values(CV_8U_32F_C1, CV_8U_32F_C3, CV_16U_32F_C1, CV_16U_32F_C3, CV_32F_32F_C1, CV_32F_32F_C3,
                    CV_8U_64F_C1, CV_8U_64F_C3, CV_16U_64F_C1, CV_16U_64F_C3, CV_32F_64F_C1, CV_32F_64F_C3)
                )
            )
{
    const AccumulateParamType params = GetParam();
    const Size     size       = get<1>(params);
    const int      srcType    = CV_MAT_PAIR_GET1(get<2>(params));
    const int      dstType    = CV_MAT_PAIR_GET2(get<2>(params));

    Mat src(size, srcType);
    Mat mask(size, CV_8U);
    Mat dst(size, dstType);
    Mat dstRef(size, dstType);
    declare.in(src, WARMUP_RNG).out(dst, dstRef);
    randu(mask, 0, 2);

    // Conformance check
    for(int i = 0; i < 10; i++)
    {
        RUN_IPP(cv::accumulate(src, dst, mask));
        RUN_REF(cv::accumulate(src, dstRef, mask));
    }
    VERIFY_DATA(dst, dstRef, 0, 4e-7);

    IPP_TEST_CYCLE() cv::accumulate(src, dst, mask);
}

/////////// cv::accumulateProduct //////////////////////

PERF_TEST_P(AccumulateFixture, ippAccumulateProduct,
            ::testing::Combine(
                IPP_TEST_STHREAD,
                IPP_TEST_SIZES,
                ::testing::Values(CV_8U_32F_C1, CV_8U_32F_C3, CV_16U_32F_C1, CV_16U_32F_C3, CV_32F_32F_C1, CV_32F_32F_C3,
                    CV_8U_64F_C1, CV_8U_64F_C3, CV_16U_64F_C1, CV_16U_64F_C3, CV_32F_64F_C1, CV_32F_64F_C3)
                )
            )
{
    const AccumulateParamType params = GetParam();
    const Size     size       = get<1>(params);
    const int      srcType    = CV_MAT_PAIR_GET1(get<2>(params));
    const int      dstType    = CV_MAT_PAIR_GET2(get<2>(params));

    Mat src1(size, srcType);
    Mat src2(size, srcType);
    Mat dst(size, dstType);
    Mat dstRef(size, dstType);
    declare.in(src1, src2, WARMUP_RNG).out(dst, dstRef);

    // Conformance check
    for(int i = 0; i < 10; i++)
    {
        RUN_IPP(cv::accumulateProduct(src1, src2, dst));
        RUN_REF(cv::accumulateProduct(src1, src2, dstRef));
    }
    VERIFY_DATA(dst, dstRef, 0, 4e-7);

    IPP_TEST_CYCLE() cv::accumulateProduct(src1, src2, dst);
}

PERF_TEST_P(AccumulateFixture, ippAccumulateProduct_Mask,
            ::testing::Combine(
                IPP_TEST_STHREAD,
                IPP_TEST_SIZES,
                ::testing::Values(CV_8U_32F_C1, CV_8U_32F_C3, CV_16U_32F_C1, CV_16U_32F_C3, CV_32F_32F_C1, CV_32F_32F_C3,
                    CV_8U_64F_C1, CV_8U_64F_C3, CV_16U_64F_C1, CV_16U_64F_C3, CV_32F_64F_C1, CV_32F_64F_C3)
                )
            )
{
    const AccumulateParamType params = GetParam();
    const Size     size       = get<1>(params);
    const int      srcType    = CV_MAT_PAIR_GET1(get<2>(params));
    const int      dstType    = CV_MAT_PAIR_GET2(get<2>(params));

    Mat src1(size, srcType);
    Mat src2(size, srcType);
    Mat mask(size, CV_8U);
    Mat dst(size, dstType);
    Mat dstRef(size, dstType);
    declare.in(src1, src2, WARMUP_RNG).out(dst, dstRef);
    randu(mask, 0, 2);

    // Conformance check
    for(int i = 0; i < 10; i++)
    {
        RUN_IPP(cv::accumulateProduct(src1, src2, dst, mask));
        RUN_REF(cv::accumulateProduct(src1, src2, dstRef, mask));
    }
    VERIFY_DATA(dst, dstRef, 0, 4e-7);

    IPP_TEST_CYCLE() cv::accumulateProduct(src1, src2, dst, mask);
}

/////////// cv::accumulateSquare //////////////////////

PERF_TEST_P(AccumulateFixture, ippAccumulateSquare,
            ::testing::Combine(
                IPP_TEST_STHREAD,
                IPP_TEST_SIZES,
                ::testing::Values(CV_8U_32F_C1, CV_8U_32F_C3, CV_16U_32F_C1, CV_16U_32F_C3, CV_32F_32F_C1, CV_32F_32F_C3,
                    CV_8U_64F_C1, CV_8U_64F_C3, CV_16U_64F_C1, CV_16U_64F_C3, CV_32F_64F_C1, CV_32F_64F_C3)
                )
            )
{
    const AccumulateParamType params = GetParam();
    const Size     size       = get<1>(params);
    const int      srcType    = CV_MAT_PAIR_GET1(get<2>(params));
    const int      dstType    = CV_MAT_PAIR_GET2(get<2>(params));

    Mat src(size, srcType);
    Mat dst(size, dstType);
    Mat dstRef(size, dstType);
    declare.in(src, WARMUP_RNG).out(dst, dstRef);

    // Conformance check
    for(int i = 0; i < 10; i++)
    {
        RUN_IPP(cv::accumulateSquare(src, dst));
        RUN_REF(cv::accumulateSquare(src, dstRef));
    }
    VERIFY_DATA(dst, dstRef, 0, 4e-7);

    IPP_TEST_CYCLE() cv::accumulateSquare(src, dst);
}

PERF_TEST_P(AccumulateFixture, ippAccumulateSquare_Mask,
            ::testing::Combine(
                IPP_TEST_STHREAD,
                IPP_TEST_SIZES,
                ::testing::Values(CV_8U_32F_C1, CV_8U_32F_C3, CV_16U_32F_C1, CV_16U_32F_C3, CV_32F_32F_C1, CV_32F_32F_C3,
                    CV_8U_64F_C1, CV_8U_64F_C3, CV_16U_64F_C1, CV_16U_64F_C3, CV_32F_64F_C1, CV_32F_64F_C3)
                )
            )
{
    const AccumulateParamType params = GetParam();
    const Size     size       = get<1>(params);
    const int      srcType    = CV_MAT_PAIR_GET1(get<2>(params));
    const int      dstType    = CV_MAT_PAIR_GET2(get<2>(params));

    Mat src(size, srcType);
    Mat mask(size, CV_8U);
    Mat dst(size, dstType);
    Mat dstRef(size, dstType);
    declare.in(src, WARMUP_RNG).out(dst, dstRef);
    randu(mask, 0, 2);

    // Conformance check
    for(int i = 0; i < 10; i++)
    {
        RUN_IPP(cv::accumulateSquare(src, dst, mask));
        RUN_REF(cv::accumulateSquare(src, dstRef, mask));
    }
    VERIFY_DATA(dst, dstRef, 0, 4e-7);

    IPP_TEST_CYCLE() cv::accumulateSquare(src, dst, mask);
}

/////////// cv::accumulateWeighted //////////////////////

PERF_TEST_P(AccumulateFixture, ippAccumulateWeighted,
            ::testing::Combine(
                IPP_TEST_STHREAD,
                IPP_TEST_SIZES,
                ::testing::Values(CV_8U_32F_C1, CV_8U_32F_C3, CV_16U_32F_C1, CV_16U_32F_C3, CV_32F_32F_C1, CV_32F_32F_C3,
                    CV_8U_64F_C1, CV_8U_64F_C3, CV_16U_64F_C1, CV_16U_64F_C3, CV_32F_64F_C1, CV_32F_64F_C3)
                )
            )
{
    const AccumulateParamType params = GetParam();
    const Size     size       = get<1>(params);
    const int      srcType    = CV_MAT_PAIR_GET1(get<2>(params));
    const int      dstType    = CV_MAT_PAIR_GET2(get<2>(params));

    Scalar weight = 1;
    Mat src(size, srcType);
    Mat dst(size, dstType);
    Mat dstRef(size, dstType);
    declare.in(src, WARMUP_RNG).out(dst, dstRef);
    randu(weight, 0.5, 2);

    // Conformance check
    for(int i = 0; i < 10; i++)
    {
        randu(weight, 0.5, 2);
        RUN_IPP(cv::accumulateWeighted(src, dst, weight[0]));
        RUN_REF(cv::accumulateWeighted(src, dstRef, weight[0]));
    }
    VERIFY_DATA(dst, dstRef, 0, 4e-7);

    IPP_TEST_CYCLE_MULTIRUN(5) cv::accumulateWeighted(src, dst, weight[0]);
}

PERF_TEST_P(AccumulateFixture, ippAccumulateWeighted_Mask,
            ::testing::Combine(
                IPP_TEST_STHREAD,
                IPP_TEST_SIZES,
                ::testing::Values(CV_8U_32F_C1, CV_8U_32F_C3, CV_16U_32F_C1, CV_16U_32F_C3, CV_32F_32F_C1, CV_32F_32F_C3,
                    CV_8U_64F_C1, CV_8U_64F_C3, CV_16U_64F_C1, CV_16U_64F_C3, CV_32F_64F_C1, CV_32F_64F_C3)
                )
            )
{
    const AccumulateParamType params = GetParam();
    const Size     size       = get<1>(params);
    const int      srcType    = CV_MAT_PAIR_GET1(get<2>(params));
    const int      dstType    = CV_MAT_PAIR_GET2(get<2>(params));

    Scalar weight = 1;
    Mat src(size, srcType);
    Mat mask(size, CV_8U);
    Mat dst(size, dstType);
    Mat dstRef(size, dstType);
    declare.in(src, WARMUP_RNG).out(dst, dstRef);
    randu(mask, 0, 2);
    randu(weight, 0.5, 2);

    // Conformance check
    for(int i = 0; i < 10; i++)
    {
        randu(weight, 0.5, 2);
        RUN_IPP(cv::accumulateWeighted(src, dst, weight[0], mask));
        RUN_REF(cv::accumulateWeighted(src, dstRef, weight[0], mask));
    }
    VERIFY_DATA(dst, dstRef, 0, 4e-7);

    IPP_TEST_CYCLE_MULTIRUN(5) cv::accumulateWeighted(src, dst, weight[0], mask);
}

/////////// cv::distanceTransform //////////////////////

CV_ENUM(DistanceTransformMaskType, MASK_TRUE, MASK_3x3, MASK_5x5)

typedef tuple<IppTestThreads, Size, MatType, MatType, DistanceTransformMaskType> DistanceTransformParamType;
typedef TestBaseWithParam<DistanceTransformParamType> DistanceTransformFixture;

PERF_TEST_P(DistanceTransformFixture, ippDistanceTransform,
            ::testing::Combine(
                IppTestThreads::all(),
                IPP_TEST_SIZES,
                ::testing::Values(CV_8UC1),
                ::testing::Values(CV_8UC1, CV_32FC1),
                DistanceTransformMaskType::all()
                )
            )
{
    const DistanceTransformParamType params = GetParam();
    const Size     size       = get<1>(params);
    const int      srcType    = get<2>(params);
    const int      dstType    = get<3>(params);
    const int      maskSize   = get<4>(params);
          int      distType   = (dstType == CV_8UC1)?DIST_L1:DIST_L2;

    if(dstType == CV_8UC1 && maskSize != MASK_3x3)
        throw ::perf::TestBase::PerfSkipTestException();

    // Init IPP extensions
    {
        TS_SET_THREADS(get<0>(params));
    }

    Mat src(size, srcType);
    Mat dst(size, dstType);
    Mat dstRef(size, dstType);

    declare.in(src, WARMUP_RNG).out(dst, dstRef);

    // Conformance check
    RUN_IPP(cv::distanceTransform(src, dst, distType, maskSize, dstType));
    RUN_REF(cv::distanceTransform(src, dstRef, distType, maskSize, dstType));
    VERIFY_DATA(dst, dstRef, 0, 1e-5);

    IPP_TEST_CYCLE() cv::distanceTransform(src, dst, distType, maskSize, dstType);
}

/////////// cv::calcHist //////////////////////

enum HistAccumEnum
{
    NO_ACCUM = 0,
    ACCUM    = 1
};

enum HistUnifEnum
{
    NON_UNIFORM = 0,
    UNIFORM      = 1
};

CV_ENUM(HistAccumType, NO_ACCUM, ACCUM)
CV_ENUM(HistUnifType, NON_UNIFORM, UNIFORM)

typedef tuple<IppTestThreads, Size, MatType, HistUnifType, HistAccumType> CalcHistParamType;
typedef TestBaseWithParam<CalcHistParamType> CalcHistFixture;

PERF_TEST_P(CalcHistFixture, ippCalcHist,
            ::testing::Combine(
                IppTestThreads::all(),
                IPP_TEST_SIZES,
                ::testing::Values(CV_8UC1, CV_16UC1, CV_32FC1),
                HistUnifType::all(),
                HistAccumType::all()
                )
            )
{
    const CalcHistParamType params = GetParam();
    const Size     size          = get<1>(params);
    const int      type          = get<2>(params);
    const bool     uniform       = (get<3>(params) == UNIFORM);
    const bool     accumulate    = (get<4>(params) == ACCUM);
    const int      bins          = 256;
    const int      channels      = 0;
    const int      dims          = 1;
          float    rangeLower    = 0;
          float    rangeUpper    = 0;
          float    rangeSize     = 0;
          float    range[bins+1] = {0};
    const float*   ranges[]      = {range};

    // Init IPP extensions
    {
        TS_SET_THREADS(get<0>(params));
    }

    Mat src(size, type);
    Mat dst(1, &bins, CV_32FC1);
    Mat dstRef(1, &bins, CV_32FC1);

    if(tsTypeIsFloat(type))
    {
        rangeLower = -1;
        rangeUpper = 1;
        rangeSize  = 2;
    }
    else
    {
        rangeLower = (float)tsTypeGetMin(type);
        rangeUpper = (float)tsTypeGetMax(type)+1;
        rangeSize  = (float)tsTypeGetRange(type)+1;
    }

    if(uniform)
    {
        range[0] = rangeLower;
        range[1] = rangeUpper;
    }
    else
    {
        range[0] = rangeLower;
        for(int i = 1; i < bins+1; i++)
            range[i] = range[i-1]+rangeSize/bins;
    }

    declare.in(src, WARMUP_READ).out(dst, dstRef);
    randu(src, rangeLower, rangeUpper);

    // Conformance check
    RUN_IPP(cv::calcHist(&src, 1, &channels, noArray(), dst, dims, &bins, ranges, uniform, accumulate));
    RUN_REF(cv::calcHist(&src, 1, &channels, noArray(), dstRef, dims, &bins, ranges, uniform, accumulate));
    if(accumulate)
    {
        for(int i = 0; i < 5; i++)
        {
            RUN_IPP(cv::calcHist(&src, 1, &channels, noArray(), dst, dims, &bins, ranges, uniform, accumulate));
            RUN_REF(cv::calcHist(&src, 1, &channels, noArray(), dstRef, dims, &bins, ranges, uniform, accumulate));
        }
    }
    VERIFY_DATA(dst, dstRef, 0, 0);

    IPP_TEST_CYCLE() cv::calcHist(&src, 1, &channels, noArray(), dst, dims, &bins, ranges, uniform, accumulate);
}

/////////// cv::medianBlur //////////////////////

typedef tuple<IppTestThreads, Size, MatType, int> MedianBlurParamType;
typedef TestBaseWithParam<MedianBlurParamType> MedianBlurFixture;

PERF_TEST_P(MedianBlurFixture, ippMedianBlur,
            ::testing::Combine(
                IPP_TEST_STHREAD,
                IPP_TEST_SIZES,
                ::testing::Values(CV_8UC1, CV_8UC3, CV_8UC4, CV_16UC1, CV_16UC3, CV_16UC4, CV_16SC1, CV_16SC3, CV_16SC4, CV_32FC1),
                IPP_TEST_KERNELS
                )
            )
{
    const MedianBlurParamType params = GetParam();
    const Size     size          = get<1>(params);
    const int      type          = get<2>(params);
    const int      kerSize       = get<3>(params);

    Mat src(size, type);
    Mat dst(size, type);
    Mat dstRef(size, type);

    declare.in(src, WARMUP_READ).out(dst, dstRef);
    tsGenerateDefault(src);

    // Conformance check
    RUN_REF(cv::medianBlur(src, dstRef, kerSize));
    RUN_IPP(cv::medianBlur(src, dst, kerSize));
    VERIFY_DATA(dst, dstRef, 1, 1e-6);

    IPP_TEST_CYCLE() cv::medianBlur(src, dst, kerSize);
}

PERF_TEST_P(MedianBlurFixture, ippMedianBlur_Inplace,
            ::testing::Combine(
                IPP_TEST_STHREAD,
                IPP_TEST_SIZES,
                ::testing::Values(CV_8UC1, CV_8UC3, CV_8UC4, CV_16UC1, CV_16UC3, CV_16UC4, CV_16SC1, CV_16SC3, CV_16SC4, CV_32FC1),
                IPP_TEST_KERNELS
                )
            )
{
    const MedianBlurParamType params = GetParam();
    const Size     size          = get<1>(params);
    const int      type          = get<2>(params);
    const int      kerSize       = get<3>(params);

    Mat src(size, type);
    Mat dst(size, type);
    Mat dstRef(size, type);

    declare.in(src, WARMUP_READ).out(dst, dstRef);
    tsGenerateDefault(src);
    src.copyTo(dst);
    src.copyTo(dstRef);

    // Conformance check
    RUN_REF(cv::medianBlur(dstRef, dstRef, kerSize));
    RUN_IPP(cv::medianBlur(dst, dst, kerSize));
    VERIFY_DATA(dst, dstRef, 1, 1e-6);

    IPP_TEST_CYCLE() cv::medianBlur(dst, dst, kerSize);
}

/////////// cv::integral //////////////////////

enum IntegralTiltEnum
{
    NON_TILTED = 0,
    TILTED     = 1
};

CV_ENUM(IntegralTiltType, NON_TILTED, TILTED)

typedef tuple<IppTestThreads, Size, MatPairs, IntegralTiltType> IntegralParamType;
typedef TestBaseWithParam<IntegralParamType> IntegralFixture;

PERF_TEST_P(IntegralFixture, ippIntegral_Sum,
            ::testing::Combine(
                IPP_TEST_STHREAD,
                IPP_TEST_SIZES,
                ::testing::Values(CV_8U_32S_C1, CV_8U_32F_C1, CV_32F_32F_C1),
                IntegralTiltType::all()
                )
            )
{
    const IntegralParamType params = GetParam();
    const Size     srcSize       = get<1>(params);
    const Size     dstSize       = Size(srcSize.width+1, srcSize.height+1);
    const int      typePair      = get<2>(params);
          int      srcType       = CV_MAT_PAIR_GET1(typePair);
          int      sumType       = CV_MAT_PAIR_GET2(typePair);
    const bool     tilted        = (get<3>(params)==TILTED);

    Mat src(srcSize, srcType);
    Mat sum(dstSize, sumType);
    Mat sumRef(dstSize, sumType);
    Mat sumTilt(dstSize, sumType);
    Mat sumTiltRef(dstSize, sumType);

    declare.in(src, WARMUP_READ).out(sum, sumRef);
    tsGenerateDefault(src);

    // Conformance check
    RUN_IPP(cv::integral(src, sum, noArray(), (tilted)?sumTilt:noArray(), sum.depth()));
    RUN_REF(cv::integral(src, sumRef, noArray(), (tilted)?sumTiltRef:noArray(), sum.depth()));
    VERIFY_DATA(sum, sumRef, 0, 4e-6);
    if(tilted)
    {
        VERIFY_DATA(sumTilt, sumTiltRef, 0, 4e-6);
    }

    IPP_TEST_CYCLE_MULTIRUN(10) cv::integral(src, sum, noArray(), (tilted)?sumTilt:noArray(), sum.depth());
}

PERF_TEST_P(IntegralFixture, ippIntegral_SqSum,
            ::testing::Combine(
                IPP_TEST_STHREAD,
                IPP_TEST_SIZES,
                ::testing::Values(CV_8U_32S_32S_C1, CV_8U_32S_64F_C1, CV_8U_32F_64F_C1),
                IntegralTiltType::all()
                )
            )
{
    const IntegralParamType params = GetParam();
    const Size     srcSize       = get<1>(params);
    const Size     dstSize       = Size(srcSize.width+1, srcSize.height+1);
    const int      typePair      = get<2>(params);
          int      srcType       = CV_MAT_PAIR_GET1(typePair);
          int      sumType       = CV_MAT_PAIR_GET2(typePair);
          int      sqSumType     = CV_MAT_PAIR_GET3(typePair);
    const bool     tilted        = (get<3>(params)==TILTED);

    Mat src(srcSize, srcType);
    Mat sum(dstSize, sumType);
    Mat sqSum(dstSize, sqSumType);
    Mat sumRef(dstSize, sumType);
    Mat sqSumRef(dstSize, sqSumType);
    Mat sumTilt(dstSize, sumType);
    Mat sumTiltRef(dstSize, sumType);

    declare.in(src, WARMUP_READ).out(sum, sumRef, sqSum, sqSumRef);
    tsGenerateDefault(src);

    // Conformance check
    RUN_IPP(cv::integral(src, sum, sqSum, (tilted)?sumTilt:noArray(), sum.depth(), sqSum.depth()));
    RUN_REF(cv::integral(src, sumRef, sqSumRef, (tilted)?sumTiltRef:noArray(), sum.depth(), sqSum.depth()));
    VERIFY_DATA(sum, sumRef, 0, 2e-6);
    if(tilted)
    {
        VERIFY_DATA(sumTilt, sumTiltRef, 0, 2e-6);
    }
    VERIFY_DATA(sqSum, sqSumRef, 0, 2e-6);

    IPP_TEST_CYCLE_MULTIRUN(10) cv::integral(src, sum, sqSum, (tilted)?sumTilt:noArray(), sum.depth(), sqSum.depth());
}

/////////// cv::matchTemplate //////////////////////

CV_ENUM(MatchTemplateType, TM_SQDIFF, TM_SQDIFF_NORMED, TM_CCORR, TM_CCORR_NORMED, TM_CCOEFF, TM_CCOEFF_NORMED)

typedef tuple<IppTestThreads, Size, MatPairs, MatchTemplateType> MatchTemplateParamType;
typedef TestBaseWithParam<MatchTemplateParamType> MatchTemplateFixture;

PERF_TEST_P(MatchTemplateFixture, ippMatchTemplate,
            ::testing::Combine(
                IPP_TEST_STHREAD,
                IPP_TEST_SIZES,
                ::testing::Values(CV_8U_32F_C1, CV_32F_32F_C1),
                MatchTemplateType::all()
                )
            )
{
    const MatchTemplateParamType params = GetParam();
    const Size     size          = get<1>(params);
    const Size     templSize     = Size((int)(size.width*0.2), (int)(size.height*0.2));
    const Size     dstSize       = Size(size.width-templSize.width+1, size.height-templSize.height+1);
    const int      typePair      = get<2>(params);
          int      srcType       = CV_MAT_PAIR_GET1(typePair);
          int      dstType       = CV_MAT_PAIR_GET2(typePair);
    const int      matchType     = get<3>(params);

    Mat src = tsLoadImage(getDataPath("ipp/boat2.jpg"), srcType, size);
    Mat templOrig;
    src.copyTo(templOrig);
    Mat templ = Mat(templOrig, Rect(32, 32, templSize.width, templSize.height));
    Mat dst(dstSize, dstType);
    Mat dstRef(dstSize, dstType);

    declare.in(src, WARMUP_READ).out(dst, dstRef);

    // Conformance check
    RUN_IPP(cv::matchTemplate(src, templ, dst, matchType));
    RUN_REF(cv::matchTemplate(src, templ, dstRef, matchType));
    VERIFY_DATA(dst, dstRef, 0, 5e-5);

    IPP_TEST_CYCLE() cv::matchTemplate(src, templ, dst, matchType);
}

/////////// cv::threshold //////////////////////

enum ThresholdTypeEnum
{
    GT      = THRESH_TRUNC,
    LTVAL   = THRESH_TOZERO,
    GTVAL   = THRESH_TOZERO_INV,
    GT_OTSU = THRESH_TRUNC|THRESH_OTSU
};

CV_ENUM(ThresholdType, GT, LTVAL, GTVAL, GT_OTSU)

typedef tuple<IppTestThreads, Size, MatType, ThresholdType> ThresholdParamType;
typedef TestBaseWithParam<ThresholdParamType> ThresholdFixture;

PERF_TEST_P(ThresholdFixture, ippThreshold,
            ::testing::Combine(
                IppTestThreads::all(),
                IPP_TEST_SIZES,
                ::testing::Values(CV_8UC1, CV_16SC1, CV_32FC1),
                ThresholdType::all()
                )
            )
{
    const ThresholdParamType params = GetParam();
    const Size     size          = get<1>(params);
    const int      type          = get<2>(params);
    const int      thresType     = get<3>(params);
    const double   thres         = tsTypeGetRange(type)*0.5 + tsTypeGetMin(type);

    if(thresType&THRESH_OTSU && type != CV_8UC1)
        throw ::perf::TestBase::PerfSkipTestException();

    // Init IPP extensions
    {
        TS_SET_THREADS(get<0>(params));
    }

    Mat src(size, type);
    Mat dst(size, type);
    Mat dstRef(size, type);

    declare.in(src, WARMUP_READ).out(dst, dstRef);
    tsGenerateDefault(src);

    // Conformance check
    RUN_IPP(cv::threshold(src, dst, thres, 1, thresType));
    RUN_REF(cv::threshold(src, dstRef, thres, 1, thresType));
    VERIFY_DATA(dst, dstRef, 0, 1e-8);

    IPP_TEST_CYCLE_MULTIRUN(10) cv::threshold(src, dst, thres, 1, thresType);
}

PERF_TEST_P(ThresholdFixture, ippThreshold_Inplace,
            ::testing::Combine(
                IppTestThreads::all(),
                IPP_TEST_SIZES,
                ::testing::Values(CV_8UC1, CV_16SC1, CV_32FC1),
                ThresholdType::all()
                )
            )
{
    const ThresholdParamType params = GetParam();
    const Size     size          = get<1>(params);
    const int      type          = get<2>(params);
    const int      thresType     = get<3>(params);
    const double   thres         = tsTypeGetRange(type)*0.5 + tsTypeGetMin(type);

    if(thresType&THRESH_OTSU && type != CV_8UC1)
        throw ::perf::TestBase::PerfSkipTestException();

    // Init IPP extensions
    {
        TS_SET_THREADS(get<0>(params));
    }

    Mat src(size, type);
    Mat dst(size, type);
    Mat dstRef(size, type);

    declare.in(src, WARMUP_READ).out(dst, dstRef);
    tsGenerateDefault(src);
    src.copyTo(dst);
    src.copyTo(dstRef);

    // Conformance check
    RUN_IPP(cv::threshold(dst, dst, thres, 1, thresType));
    RUN_REF(cv::threshold(dstRef, dstRef, thres, 1, thresType));
    VERIFY_DATA(dst, dstRef, 0, 1e-8);

    IPP_TEST_CYCLE_MULTIRUN(10) cv::threshold(dst, dst, thres, 1, thresType);
}

} // namespace
#endif // HAVE_IPP

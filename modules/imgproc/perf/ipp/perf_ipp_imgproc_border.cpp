#include "../perf_precomp.hpp"

#ifdef HAVE_IPP
#include "opencv2/ts/ipp_perf.hpp"

namespace opencv_test
{
using namespace perf;

/////////// cv::GaussianBlur //////////////////////

typedef tuple<IppTestThreads, Size, MatType, int, IppBorder> GaussianBlurParamType;
typedef TestBaseWithParam<GaussianBlurParamType> GaussianBlurFixture;

PERF_TEST_P(GaussianBlurFixture, ippGaussianBlur,
            ::testing::Combine(
                IppTestThreads::all(),
                IPP_TEST_SIZES,
                ::testing::Values(CV_8UC1, CV_8UC3, CV_16UC1, CV_16UC3, CV_16SC1, CV_16SC3, CV_32FC1, CV_32FC3),
                IPP_TEST_KERNELS,
                IppBorder::all()
                )
            )
{
    const GaussianBlurParamType params = GetParam();
    const Size     size       = get<1>(params);
    const int      type       = get<2>(params);
    const int      maskSize   = get<3>(params);
          int      border     = get<4>(params);

    // Init IPP extensions
    {
        TS_SET_THREADS(get<0>(params));
    }

    Mat src(size, type);
    Mat dst(size, type);
    Mat dstRef(size, type);

    declare.in(src, WARMUP_RNG).out(dst);
    tsGenerateDefault(src);
    src = tsSrcSetBorder(src, border);

    // Conformance check
    RUN_IPP(cv::GaussianBlur(src, dst, Size(maskSize, maskSize), 1, 1 DEFAULT_BORDER(border)));
    RUN_REF(cv::GaussianBlur(src, dstRef, Size(maskSize, maskSize), 1, 1 DEFAULT_BORDER(border)));
    VERIFY_DATA(dst, dstRef, 2, 4e-7);

    IPP_TEST_CYCLE() cv::GaussianBlur(src, dst, Size(maskSize, maskSize), 1, 1 DEFAULT_BORDER(border));
}

/////////// cv::bilateralFilter //////////////////////

typedef tuple<IppTestThreads, Size, MatType, int, IppBorder> BilateralFilterParamType;
typedef TestBaseWithParam<BilateralFilterParamType> BilateralFilterFixture;

PERF_TEST_P(BilateralFilterFixture, ippBilateralFilter,
            ::testing::Combine(
                IppTestThreads::all(),
                IPP_TEST_SIZES,
                ::testing::Values(CV_8UC1, CV_8UC3, CV_32FC1, CV_32FC3),
                IPP_TEST_KERNELS,
                IppBorder::all()
                )
            )
{
    const BilateralFilterParamType params = GetParam();
    const Size     size       = get<1>(params);
    const int      type       = get<2>(params);
    const int      diam       = get<3>(params);
          int      border     = get<4>(params);

    // Init IPP extensions
    {
        TS_SET_THREADS(get<0>(params));
    }

    Mat src(size, type);
    Mat dst(size, type);
    Mat dstRef(size, type);
    declare.in(src, WARMUP_READ).out(dst);
    tsGenerateDefault(src);
    src = tsSrcSetBorder(src, border);

    // Conformance check
    RUN_IPP(cv::bilateralFilter(src, dst, diam, 3, 3 DEFAULT_BORDER(border)));
    RUN_REF(cv::bilateralFilter(src, dstRef, diam, 3, 3 DEFAULT_BORDER(border)));
    VERIFY_DATA(dst, dstRef, 1, 4e-5);

    IPP_TEST_CYCLE() cv::bilateralFilter(src, dst, diam, 3, 3 DEFAULT_BORDER(border));
}

/////////// cv::Sobel //////////////////////

enum DerivEnum
{
    VER_FIRST,
    NVER_FIRST,
    VER_SECOND,
    HOR_FIRST,
    HOR_SECOND
};

CV_ENUM(SobelDerivType, NVER_FIRST, VER_SECOND, HOR_FIRST, HOR_SECOND)

typedef tuple<IppTestThreads, Size, MatType, MatType, SobelDerivType, int, int, IppBorder> SobelParamType;
typedef TestBaseWithParam<SobelParamType> SobelFixture;

PERF_TEST_P(SobelFixture, ippSobel,
            ::testing::Combine(
                IPP_TEST_STHREAD,
                IPP_TEST_SIZES,
                ::testing::Values(CV_8UC1, CV_16SC1, CV_32FC1),
                ::testing::Values(CV_8UC1, CV_16SC1, CV_32FC1),
                SobelDerivType::all(),
                ::testing::Values(3, 5),
                ::testing::Values(0, 1),
                IppBorder::all()
                )
            )
{
    const SobelParamType params = GetParam();
    const Size     size       = get<1>(params);
    const int      srcType    = get<2>(params);
    const int      dstType    = get<3>(params);
    const int      deriv      = get<4>(params);
    const int      kerSize    = get<5>(params);
    const int      scaled     = get<6>(params);
          int      border     = get<7>(params);

    int    srcDepth = CV_MAT_DEPTH(srcType);
    int    dstDepth = CV_MAT_DEPTH(dstType);
    double fMul = (scaled)?1.333:1.;
    double fAdd = (scaled)?2:0;

    int dx = (deriv == NVER_FIRST)?1:(deriv == VER_SECOND)?2:0;
    int dy = (deriv == HOR_FIRST)?1:(deriv == HOR_SECOND)?2:0;

    if(dstDepth < srcDepth)
        throw ::perf::TestBase::PerfSkipTestException();

    Mat src(size, srcType);
    Mat dst(size, dstType);
    Mat dstRef(size, dstType);
    declare.in(src, WARMUP_RNG).out(dst);
    src = tsSrcSetBorder(src, border);

    // Conformance check
    RUN_IPP(cv::Sobel(src, dst, dstDepth, dx, dy, kerSize, fMul, fAdd DEFAULT_BORDER(border)));
    RUN_REF(cv::Sobel(src, dstRef, dstDepth, dx, dy, kerSize, fMul, fAdd DEFAULT_BORDER(border)));
    VERIFY_DATA(dst, dstRef, 1, 3e-5);

    IPP_TEST_CYCLE() cv::Sobel(src, dst, dstDepth, dx, dy, kerSize, fMul, fAdd DEFAULT_BORDER(border));
}

/////////// cv::Scharr //////////////////////

CV_ENUM(ScharrDerivType, VER_FIRST, HOR_FIRST)

typedef tuple<IppTestThreads, Size, MatType, MatType, ScharrDerivType, int, int, IppBorder> ScharrParamType;
typedef TestBaseWithParam<ScharrParamType> ScharrFixture;

PERF_TEST_P(ScharrFixture, ippScharr,
            ::testing::Combine(
                IPP_TEST_STHREAD,
                IPP_TEST_SIZES,
                ::testing::Values(CV_8UC1, CV_16SC1, CV_32FC1),
                ::testing::Values(CV_8UC1, CV_16SC1, CV_32FC1),
                ScharrDerivType::all(),
                ::testing::Values(3),
                ::testing::Values(0, 1),
                IppBorder::all()
                )
            )
{
    const ScharrParamType params = GetParam();
    const Size     size       = get<1>(params);
    const int      srcType    = get<2>(params);
    const int      dstType    = get<3>(params);
    const int      deriv      = get<4>(params);
    const int      scaled     = get<6>(params);
          int      border     = get<7>(params);

    int    srcDepth = CV_MAT_DEPTH(srcType);
    int    dstDepth = CV_MAT_DEPTH(dstType);
    double fMul = (scaled)?1.333:1.;
    double fAdd = (scaled)?2:0;

    int dx = (deriv == VER_FIRST)?1:0;
    int dy = (deriv == HOR_FIRST)?1:0;

    if(dstDepth < srcDepth)
        throw ::perf::TestBase::PerfSkipTestException();

    Mat src(size, srcType);
    Mat dst(size, dstType);
    Mat dstRef(size, dstType);
    declare.in(src, WARMUP_RNG).out(dst);
    src = tsSrcSetBorder(src, border);

    // Conformance check
    RUN_IPP(cv::Scharr(src, dst, dstDepth, dx, dy, fMul, fAdd DEFAULT_BORDER(border)));
    RUN_REF(cv::Scharr(src, dstRef, dstDepth, dx, dy, fMul, fAdd DEFAULT_BORDER(border)));
    VERIFY_DATA(dst, dstRef, 1, 3e-5);

    IPP_TEST_CYCLE() cv::Scharr(src, dst, dstDepth, dx, dy, fMul, fAdd DEFAULT_BORDER(border));
}

/////////// cv::Laplacian //////////////////////

typedef tuple<IppTestThreads, Size, MatType, MatType, int, int, IppBorder> LaplacianParamType;
typedef TestBaseWithParam<LaplacianParamType> LaplacianFixture;

PERF_TEST_P(LaplacianFixture, ippLaplacian,
            ::testing::Combine(
                IPP_TEST_STHREAD,
                IPP_TEST_SIZES,
                ::testing::Values(CV_8UC1, CV_32FC1),
                ::testing::Values(CV_8UC1, CV_16SC1, CV_32FC1),
                ::testing::Values(3, 5),
                ::testing::Values(0, 1),
                IppBorder::all()
                )
            )
{
    const LaplacianParamType params = GetParam();
    const Size     size       = get<1>(params);
    const int      srcType    = get<2>(params);
    const int      dstType    = get<3>(params);
    const int      kerSize    = get<4>(params);
    const int      scaled     = get<5>(params);
          int      border     = get<6>(params);

    int    srcDepth = CV_MAT_DEPTH(srcType);
    int    dstDepth = CV_MAT_DEPTH(dstType);
    double fMul = (scaled)?1.333:1.;
    double fAdd = (scaled)?2:0;

    if(dstDepth < srcDepth)
        throw ::perf::TestBase::PerfSkipTestException();

    Mat src(size, srcType);
    Mat dst(size, dstType);
    Mat dstRef(size, dstType);
    declare.in(src, WARMUP_RNG).out(dst);
    src = tsSrcSetBorder(src, border);

    // Conformance check
    RUN_IPP(cv::Laplacian(src, dst, dstDepth, kerSize, fMul, fAdd DEFAULT_BORDER(border)));
    RUN_REF(cv::Laplacian(src, dstRef, dstDepth, kerSize, fMul, fAdd DEFAULT_BORDER(border)));
    VERIFY_DATA(dst, dstRef, 1, 3e-5);

    IPP_TEST_CYCLE() cv::Laplacian(src, dst, dstDepth, kerSize, fMul, fAdd DEFAULT_BORDER(border));
}

/////////// cv::erode //////////////////////

enum
{
    MASK_RECT,
    MASK_RAND
};

const uchar morphMask_21x21[] = {
    0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0,
    1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0,
    0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0,
    1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1,
    0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0,
    1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1,
    0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0,
    1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0,
    1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1,
    0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1,
    0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1,
    0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1,
    1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0,
    0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1,
    1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1,
    0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1,
    1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0,
    0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1,
    1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1,
    1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1,
    0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1
};

CV_ENUM(IppMorphMask, MASK_RECT, MASK_RAND)

typedef tuple<IppTestThreads, Size, MatType, int, IppMorphMask, int, IppBorder> MorphologySimpleParamType;
typedef TestBaseWithParam<MorphologySimpleParamType> MorphologySimpleFixture;

PERF_TEST_P(MorphologySimpleFixture, ippMorphErode,
            ::testing::Combine(
                IPP_TEST_STHREAD,
                IPP_TEST_SIZES,
                ::testing::Values(CV_8UC1, CV_8UC3, CV_8UC4, CV_16UC1, CV_16SC1, CV_32FC1, CV_32FC3, CV_32FC4),
                IPP_TEST_KERNELS,
                IppMorphMask::all(),
                ::testing::Values(1, 3),
                IppBorder::all()
                )
            )
{
    const MorphologySimpleParamType params = GetParam();
    const Size     size       = get<1>(params);
    const int      type       = get<2>(params);
    const int      maskSize   = get<3>(params);
    const int      maskType   = get<4>(params);
    const int      iterations = get<5>(params);
          int      border     = get<6>(params);

    Mat kernel(Size(maskSize, maskSize), CV_8U, (uchar*)morphMask_21x21, maskSize);

    Mat src = tsLoadImage(getDataPath("ipp/boat2.jpg"), type, size);
    Mat dst(size, type);
    Mat dstRef(size, type);
    declare.in(src, WARMUP_READ).out(dst);
    src = tsSrcSetBorder(src, border);
    if(maskType == MASK_RECT)
        kernel = getStructuringElement(MORPH_RECT, Size(maskSize, maskSize));

    // Conformance check
    RUN_IPP(cv::erode(src, dst, kernel, Point(-1,-1), iterations DEFAULT_BORDER(border)));
    RUN_REF(cv::erode(src, dstRef, kernel, Point(-1,-1), iterations DEFAULT_BORDER(border)));
    VERIFY_DATA(dst, dstRef, 0, 0);

    IPP_TEST_CYCLE() cv::erode(src, dst, kernel, Point(-1,-1), iterations DEFAULT_BORDER(border));
}

/////////// cv::dilate //////////////////////

PERF_TEST_P(MorphologySimpleFixture, ippMorphDilate,
            ::testing::Combine(
                IPP_TEST_STHREAD,
                IPP_TEST_SIZES,
                ::testing::Values(CV_8UC1, CV_8UC3, CV_8UC4, CV_16UC1, CV_16SC1, CV_32FC1, CV_32FC3, CV_32FC4),
                IPP_TEST_KERNELS,
                IppMorphMask::all(),
                ::testing::Values(1, 3),
                IppBorder::all()
                )
            )
{
    const MorphologySimpleParamType params = GetParam();
    const Size     size       = get<1>(params);
    const int      type       = get<2>(params);
    const int      maskSize   = get<3>(params);
    const int      maskType   = get<4>(params);
    const int      iterations = get<5>(params);
          int      border     = get<6>(params);

    Mat kernel(Size(maskSize, maskSize), CV_8U, (uchar*)morphMask_21x21, maskSize);

    Mat src = tsLoadImage(getDataPath("ipp/boat2.jpg"), type, size);
    Mat dst(size, type);
    Mat dstRef(size, type);
    declare.in(src, WARMUP_READ).out(dst);
    src = tsSrcSetBorder(src, border);
    if(maskType == MASK_RECT)
        kernel = getStructuringElement(MORPH_RECT, Size(maskSize, maskSize));

    // Conformance check
    RUN_IPP(cv::dilate(src, dst, kernel, Point(-1,-1), iterations DEFAULT_BORDER(border)));
    RUN_REF(cv::dilate(src, dstRef, kernel, Point(-1,-1), iterations DEFAULT_BORDER(border)));
    VERIFY_DATA(dst, dstRef, 0, 0);

    IPP_TEST_CYCLE() cv::dilate(src, dst, kernel, Point(-1,-1), iterations DEFAULT_BORDER(border));
}

/////////// cv::morphologyEx //////////////////////

CV_ENUM(IppMorphType, MORPH_OPEN, MORPH_CLOSE, MORPH_BLACKHAT, MORPH_TOPHAT, MORPH_GRADIENT)

typedef tuple<IppTestThreads, Size, MatType, IppMorphType, int, IppMorphMask, int, IppBorder> MorphologyExParamType;
typedef TestBaseWithParam<MorphologyExParamType> MorphologyExFixture;

PERF_TEST_P(MorphologyExFixture, ippMorphEx,
            ::testing::Combine(
                IPP_TEST_STHREAD,
                IPP_TEST_SIZES,
                ::testing::Values(CV_8UC1, CV_8UC3, CV_8UC4, CV_16UC1, CV_16SC1, CV_32FC1, CV_32FC3, CV_32FC4),
                IppMorphType::all(),
                IPP_TEST_KERNELS,
                IppMorphMask::all(),
                ::testing::Values(1, 3),
                IppBorder::all()
                )
            )
{
    const MorphologyExParamType params = GetParam();
    const Size     size       = get<1>(params);
    const int      type       = get<2>(params);
    const int      morphType  = get<3>(params);
    const int      maskSize   = get<4>(params);
    const int      maskType   = get<5>(params);
    const int      iterations = get<6>(params);
          int      border     = get<7>(params);

    Mat kernel(Size(maskSize, maskSize), CV_8U, (uchar*)morphMask_21x21, maskSize);

    Mat src = tsLoadImage(getDataPath("ipp/boat2.jpg"), type, size);
    Mat dst(size, type);
    Mat dstRef(size, type);
    declare.in(src, WARMUP_READ).out(dst);
    src = tsSrcSetBorder(src, border);
    if(maskType == MASK_RECT)
        kernel = getStructuringElement(MORPH_RECT, Size(maskSize, maskSize));

    // Conformance check
    RUN_IPP(cv::morphologyEx(src, dst, morphType, kernel, Point(-1,-1), iterations DEFAULT_BORDER(border)));
    RUN_REF(cv::morphologyEx(src, dstRef, morphType, kernel, Point(-1,-1), iterations DEFAULT_BORDER(border)));
    VERIFY_DATA(dst, dstRef, 0, 2e-7);

    IPP_TEST_CYCLE() cv::morphologyEx(src, dst, morphType, kernel, Point(-1,-1), iterations DEFAULT_BORDER(border));
}

/////////// cv::cornerHarris //////////////////////

enum DerivMaskTypeEnum
{
    SOBEL_3x3  = 3,
    SOBEL_5x5  = 5,
    SCHARR_3x3 = -1
};

CV_ENUM(DerivMaskType, SOBEL_3x3, SOBEL_5x5, SCHARR_3x3)

typedef tuple<IppTestThreads, Size, MatType, MatType, DerivMaskType, int, double, IppBorder> CornerHarrisParamType;
typedef TestBaseWithParam<CornerHarrisParamType> CornerHarrisFixture;

PERF_TEST_P(CornerHarrisFixture, ippCornerHarris,
            ::testing::Combine(
                IPP_TEST_STHREAD,
                IPP_TEST_SIZES,
                ::testing::Values(CV_8UC1, CV_32FC1),
                ::testing::Values(CV_32FC1),
                DerivMaskType::all(),
                testing::Values(3, 5),
                testing::Values(0.04, 0.1),
                IppBorder::all()
                )
            )
{
    const CornerHarrisParamType params = GetParam();
    const Size     size       = get<1>(params);
    const int      srcType    = get<2>(params);
    const int      dstType    = get<3>(params);
    const int      maskSize   = get<4>(params);
    const int      blockSize  = get<5>(params);
    const double   k          = get<6>(params);
          int      border     = get<7>(params);

    Mat src = tsLoadImage(getDataPath("ipp/boat2.jpg"), srcType, size);
    Mat dst(size, dstType);
    Mat dstRef(size, dstType);

    declare.in(src, WARMUP_READ).out(dst);
    src = tsSrcSetBorder(src, border);

    // Conformance check
    RUN_IPP(cv::cornerHarris(src, dst, blockSize, maskSize, k DEFAULT_BORDER(border)));
    RUN_REF(cv::cornerHarris(src, dstRef, blockSize, maskSize, k DEFAULT_BORDER(border)));
    VERIFY_DATA(dst, dstRef, 0, 5e-6);

    IPP_TEST_CYCLE() cv::cornerHarris(src, dst, blockSize, maskSize, k DEFAULT_BORDER(border));
}

/////////// cv::cornerMinEigenVal //////////////////////

typedef tuple<IppTestThreads, Size, MatType, MatType, DerivMaskType, int, IppBorder> CornerMinEigenValParamType;
typedef TestBaseWithParam<CornerMinEigenValParamType> CornerMinEigenValFixture;

PERF_TEST_P(CornerMinEigenValFixture, ippCornerMinEigenVal,
            ::testing::Combine(
                IPP_TEST_STHREAD,
                IPP_TEST_SIZES,
                ::testing::Values(CV_8UC1, CV_32FC1),
                ::testing::Values(CV_32FC1),
                DerivMaskType::all(),
                testing::Values(3, 5),
                IppBorder::all()
                )
            )
{
    const CornerMinEigenValParamType params = GetParam();
    const Size     size       = get<1>(params);
    const int      srcType    = get<2>(params);
    const int      dstType    = get<3>(params);
    const int      maskSize   = get<4>(params);
    const int      blockSize  = get<5>(params);
          int      border     = get<6>(params);

    Mat src = tsLoadImage(getDataPath("ipp/boat2.jpg"), srcType, size);
    Mat dst(size, dstType);
    Mat dstRef(size, dstType);

    declare.in(src, WARMUP_READ).out(dst);
    src = tsSrcSetBorder(src, border);

    // Conformance check
    RUN_IPP(cv::cornerMinEigenVal(src, dst, blockSize, maskSize DEFAULT_BORDER(border)));
    RUN_REF(cv::cornerMinEigenVal(src, dstRef, blockSize, maskSize DEFAULT_BORDER(border)));
    VERIFY_DATA(dst, dstRef, 0, 5e-6);

    IPP_TEST_CYCLE() cv::cornerMinEigenVal(src, dst, blockSize, maskSize DEFAULT_BORDER(border));
}

/////////// cv::filter2D //////////////////////

typedef tuple<IppTestThreads, Size, MatType, MatType, int, IppBorder> Filter2DParamType;
typedef TestBaseWithParam<Filter2DParamType> Filter2DFixture;

PERF_TEST_P(Filter2DFixture, ippFilter2D,
            ::testing::Combine(
                IPP_TEST_STHREAD,
                IPP_TEST_SIZES,
                ::testing::Values(CV_8UC1, CV_8UC3, CV_8UC4, CV_16UC1, CV_16UC3, CV_16UC4, CV_16SC1, CV_16SC3, CV_16SC4, CV_32FC1, CV_32FC3, CV_32FC4),
                ::testing::Values(CV_16SC1, CV_32FC1),
                IPP_TEST_KERNELS,
                IppBorder::all()
                )
            )
{
    const Filter2DParamType params = GetParam();
    const Size     size          = get<1>(params);
    const int      type          = get<2>(params);
    const int      kerType       = get<3>(params);
    const int      kerSize       = get<4>(params);
          int      border        = get<5>(params);

    Mat src(size, type);
    Mat dst(size, type);
    Mat dstRef(size, type);
    Mat ker(Size(kerSize, kerSize), kerType);

    declare.in(src, WARMUP_READ).out(dst, dstRef);
    tsGenerateDefault(src);
    src = tsSrcSetBorder(src, border);
    randu(ker, -6, 6);

    // Conformance check
    RUN_IPP(cv::filter2D(src, dst, dst.depth(), ker, Point(-1, -1), 0 DEFAULT_BORDER(border)));
    RUN_REF(cv::filter2D(src, dstRef, dst.depth(), ker, Point(-1, -1), 0 DEFAULT_BORDER(border)));
    VERIFY_DATA(dst, dstRef, 1, 1e-6);

    IPP_TEST_CYCLE() cv::filter2D(src, dst, dst.depth(), ker, Point(-1, -1), 0 DEFAULT_BORDER(border));
}

/////////// cv::sepFilter2D //////////////////////

typedef tuple<IppTestThreads, Size, MatType, int, IppBorder> SepFilter2DParamType;
typedef TestBaseWithParam<SepFilter2DParamType> SepFilter2DFixture;

PERF_TEST_P(SepFilter2DFixture, ippSepFilter2D,
            ::testing::Combine(
                IPP_TEST_STHREAD,
                IPP_TEST_SIZES,
                ::testing::Values(CV_8UC1, CV_8UC3, CV_8UC4, CV_16UC1, CV_16UC3, CV_16UC4, CV_16SC1, CV_16SC3, CV_16SC4, CV_32FC1, CV_32FC3, CV_32FC4),
                IPP_TEST_KERNELS,
                IppBorder::all()
                )
            )
{
    const SepFilter2DParamType params = GetParam();
    const Size     size          = get<1>(params);
    const int      type          = get<2>(params);
    const int      kerSize       = get<3>(params);
          int      border        = get<4>(params);

    Mat src(size, type);
    Mat dst(size, type);
    Mat dstRef(size, type);
    Mat kerX(1, kerSize, CV_32F);
    Mat kerY(1, kerSize, CV_32F);

    declare.in(src, WARMUP_READ).out(dst, dstRef);
    tsGenerateDefault(src);
    src = tsSrcSetBorder(src, border);
    randu(kerX, -6, 6);
    randu(kerY, -6, 6);

    // Conformance check
    RUN_IPP(cv::sepFilter2D(src, dst, dst.depth(), kerX, kerY, Point(-1, -1), 0 DEFAULT_BORDER(border)));
    RUN_REF(cv::sepFilter2D(src, dstRef, dst.depth(), kerX, kerY, Point(-1, -1), 0 DEFAULT_BORDER(border)));
    VERIFY_DATA(dst, dstRef, 1, 1e-6);

    IPP_TEST_CYCLE() cv::sepFilter2D(src, dst, dst.depth(), kerX, kerY, Point(-1, -1), 0 DEFAULT_BORDER(border));
}

/////////// cv::boxFilter //////////////////////

typedef tuple<IppTestThreads, Size, MatType, int, IppBorder> BoxFilterParamType;
typedef TestBaseWithParam<BoxFilterParamType> BoxFilterFixture;

PERF_TEST_P(BoxFilterFixture, ippBoxFilter,
            ::testing::Combine(
                IPP_TEST_STHREAD,
                IPP_TEST_SIZES,
                ::testing::Values(CV_8UC1, CV_8UC3, CV_8UC4, CV_16UC1, CV_16UC3, CV_16UC4, CV_16SC1, CV_16SC3, CV_16SC4, CV_32FC1, CV_32FC3, CV_32FC4),
                IPP_TEST_KERNELS,
                IppBorder::all()
                )
            )
{
    const BoxFilterParamType params = GetParam();
    const Size     size          = get<1>(params);
    const int      type          = get<2>(params);
    const int      kerSize       = get<3>(params);
    const bool     normalize     = true;
          int      border        = get<4>(params);

    Mat src(size, type);
    Mat dst(size, type);
    Mat dstRef(size, type);

    declare.in(src, WARMUP_READ).out(dst, dstRef);
    tsGenerateDefault(src);
    src = tsSrcSetBorder(src, border);

    // Conformance check
    RUN_IPP(cv::boxFilter(src, dst, dst.depth(), Size(kerSize, kerSize), Point(-1,-1), normalize DEFAULT_BORDER(border)));
    RUN_REF(cv::boxFilter(src, dstRef, dst.depth(), Size(kerSize, kerSize), Point(-1,-1), normalize DEFAULT_BORDER(border)));
    VERIFY_DATA(dst, dstRef, 1, 4e-6);

    IPP_TEST_CYCLE() cv::boxFilter(src, dst, dst.depth(), Size(kerSize, kerSize), Point(-1,-1), normalize DEFAULT_BORDER(border));
}

} // namespace
#endif // HAVE_IPP

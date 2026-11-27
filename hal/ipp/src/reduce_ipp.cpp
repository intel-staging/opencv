// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

#include "ipp_hal_core.hpp"

#include <opencv2/core.hpp>
#include <opencv2/core/base.hpp>

#include <atomic>
#include <climits>

#if IPP_VERSION_X100 >= 700

// Reduces one row: src points to the row, dst to its (cn) destination elements
typedef IppStatus (*IppReduceRowFunc)(const void* src, int srcStep, IppiSize roi, void* dst);

template<typename T, IppStatus (CV_STDCALL *fn)(const T*, int, IppiSize, Ipp64f*)>
static IppStatus ippReduceSumRow(const void* src, int srcStep, IppiSize roi, void* dst)
{
    return CV_INSTRUMENT_FUN_IPP(fn, (const T*)src, srcStep, roi, (Ipp64f*)dst);
}

template<IppStatus (CV_STDCALL *fn)(const Ipp32f*, int, IppiSize, Ipp64f*, IppHintAlgorithm)>
static IppStatus ippReduceSumRow32f(const void* src, int srcStep, IppiSize roi, void* dst)
{
    return CV_INSTRUMENT_FUN_IPP(fn, (const Ipp32f*)src, srcStep, roi, (Ipp64f*)dst, ippAlgHintAccurate);
}

template<typename T, IppStatus (CV_STDCALL *fn)(const T*, int, IppiSize, T*)>
static IppStatus ippReduceMinMaxRow(const void* src, int srcStep, IppiSize roi, void* dst)
{
    return CV_INSTRUMENT_FUN_IPP(fn, (const T*)src, srcStep, roi, (T*)dst);
}

// dim = 1, REDUCE_SUM (and REDUCE_AVG) with a CV_64F destination
static IppReduceRowFunc getReduceSumRowFunc(int src_type)
{
    switch(src_type)
    {
    case CV_8UC1:  return ippReduceSumRow<Ipp8u,  ippiSum_8u_C1R>;
    case CV_8UC3:  return ippReduceSumRow<Ipp8u,  ippiSum_8u_C3R>;
    case CV_8UC4:  return ippReduceSumRow<Ipp8u,  ippiSum_8u_C4R>;
    case CV_16UC1: return ippReduceSumRow<Ipp16u, ippiSum_16u_C1R>;
    case CV_16UC3: return ippReduceSumRow<Ipp16u, ippiSum_16u_C3R>;
    case CV_16UC4: return ippReduceSumRow<Ipp16u, ippiSum_16u_C4R>;
    case CV_16SC1: return ippReduceSumRow<Ipp16s, ippiSum_16s_C1R>;
    case CV_16SC3: return ippReduceSumRow<Ipp16s, ippiSum_16s_C3R>;
    case CV_16SC4: return ippReduceSumRow<Ipp16s, ippiSum_16s_C4R>;
    case CV_32FC1: return ippReduceSumRow32f<ippiSum_32f_C1R>;
    case CV_32FC3: return ippReduceSumRow32f<ippiSum_32f_C3R>;
    case CV_32FC4: return ippReduceSumRow32f<ippiSum_32f_C4R>;
    default:       return 0;
    }
}

// dim = 1, REDUCE_MAX / REDUCE_MIN, the destination has the source type
static IppReduceRowFunc getReduceMinMaxRowFunc(int src_type, bool isMax)
{
    switch(src_type)
    {
    case CV_8UC1:  return isMax ? ippReduceMinMaxRow<Ipp8u,  ippiMax_8u_C1R>  : ippReduceMinMaxRow<Ipp8u,  ippiMin_8u_C1R>;
    case CV_16UC1: return isMax ? ippReduceMinMaxRow<Ipp16u, ippiMax_16u_C1R> : ippReduceMinMaxRow<Ipp16u, ippiMin_16u_C1R>;
    case CV_16SC1: return isMax ? ippReduceMinMaxRow<Ipp16s, ippiMax_16s_C1R> : ippReduceMinMaxRow<Ipp16s, ippiMin_16s_C1R>;
    case CV_32FC1: return isMax ? ippReduceMinMaxRow<Ipp32f, ippiMax_32f_C1R> : ippReduceMinMaxRow<Ipp32f, ippiMin_32f_C1R>;
    default:       return 0;
    }
}

int ipp_hal_reduce(const uchar* src_data, size_t src_step, int src_type, int width, int height,
                   uchar* dst_data, size_t dst_step, int dst_type, int dim, int op)
{
    CV_HAL_CHECK_USE_IPP();

    // Only column reduction (one IPP call per row) for now
    if(dim != 1)
        return CV_HAL_ERROR_NOT_IMPLEMENTED;

    const int depth  = CV_MAT_DEPTH(src_type);
    const int cn     = CV_MAT_CN(src_type);
    const int ddepth = CV_MAT_DEPTH(dst_type);
    if(depth > CV_32F || cn < 1 || cn > 4 || width <= 0 || height <= 0)
        return CV_HAL_ERROR_NOT_IMPLEMENTED;

    // Supported (depth x channels) combinations, checked against ICV 2026.0.0 / 2026.1.0.
    // SUM: ippiSum_{8u,16u,16s,32f}_C{1,3,4}R, 64F destination only.
    // MIN/MAX: ippiMax/Min_{8u,16u,16s,32f}_C1R, the destination has the source depth.
#if defined(IPP_CALLS_ENFORCED)
    static const char implSum[CV_32F + 1][4] = { /* C1 C2 C3 C4 */
        /* 8U  */ {1, 0, 1, 1},
        /* 8S  */ {0, 0, 0, 0},
        /* 16U */ {1, 0, 1, 1},
        /* 16S */ {1, 0, 1, 1},
        /* 32S */ {0, 0, 0, 0},
        /* 32F */ {1, 0, 1, 1}};
    static const char implMinMax[CV_32F + 1] = {
        /* 8U  */ 1,
        /* 8S  */ 0,
        /* 16U */ 1,
        /* 16S */ 1,
        /* 32S */ 0,
        /* 32F */ 1};
#else // IPP_CALLS_ENFORCED is not defined, results are strictly aligned to OpenCV implementation
    static const char implSum[CV_32F + 1][4] = { /* C1 C2 C3 C4 */
        /* 8U  */ {1, 0, 1, 1},
        /* 8S  */ {0, 0, 0, 0},
        /* 16U */ {1, 0, 1, 1},
        /* 16S */ {1, 0, 1, 1},
        /* 32S */ {0, 0, 0, 0},
        /* 32F */ {0, 0, 0, 0}};  // not bit-exact (different summation order), competes with SIMD reduceColSum_32f64f
    static const char implMinMax[CV_32F + 1] = {
        /* 8U  */ 1,
        /* 8S  */ 0,
        /* 16U */ 1,
        /* 16S */ 1,
        /* 32S */ 0,
        /* 32F */ 0};  // NaN: IPP ignores a leading NaN and returns NaN for a trailing one, OpenCV does the opposite
#endif // IPP_CALLS_ENFORCED

    IppReduceRowFunc reduceRow = 0;
    if(op == cv::REDUCE_SUM)
    {
        if(ddepth == CV_64F && implSum[depth][cn - 1])
            reduceRow = getReduceSumRowFunc(src_type);
    }
    else if(op == cv::REDUCE_MAX || op == cv::REDUCE_MIN)
    {
        if(ddepth == depth && cn == 1 && implMinMax[depth])
            reduceRow = getReduceMinMaxRowFunc(src_type, op == cv::REDUCE_MAX);
    }
    if(!reduceRow)
        return CV_HAL_ERROR_NOT_IMPLEMENTED;

    // The ROI is a single row, IPP still checks the step against the row size.
    // A 1D source (height 1) can have the element size as step.
    const size_t rowSize = (size_t)width * CV_ELEM_SIZE(src_type);
    if(height == 1)
        src_step = rowSize;
    if(src_step > (size_t)INT_MAX || src_step < rowSize)
        return CV_HAL_ERROR_NOT_IMPLEMENTED;

    const IppiSize roi = { width, 1 };
    const int      srcStep = (int)src_step;

    // Same threading as the OpenCV fallback (reduceC_): every row is independent.
    // IPP errors depend only on the arguments, which are the same for all rows:
    // a failing call fails on every row before any result is written.
    std::atomic_bool ok{true};
    cv::parallel_for_(cv::Range(0, height), [&](const cv::Range& range)
    {
        for(int y = range.start; y < range.end; y++)
        {
            if(!ok.load(std::memory_order_relaxed))
                return;
            if(reduceRow(src_data + (size_t)y * src_step, srcStep, roi, dst_data + (size_t)y * dst_step) < 0)
                ok.store(false, std::memory_order_relaxed);
        }
    });
    if(!ok)
        return CV_HAL_ERROR_NOT_IMPLEMENTED;

    CV_IMPL_ADD(CV_IMPL_IPP|CV_IMPL_MT);
    return CV_HAL_ERROR_OK;
}

#endif // IPP_VERSION_X100 >= 700

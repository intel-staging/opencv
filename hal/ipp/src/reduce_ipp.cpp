// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

#include "ipp_hal_core.hpp"

#include <opencv2/core.hpp>
#include <opencv2/core/base.hpp>

#include <algorithm>
#include <atomic>
#include <climits>
#include <cstring>
#include <type_traits>

#if IPP_VERSION_X100 >= 700

// Functions marked "full IPP" below are not in ICV: in ICV builds the getters return 0 for them,
// even if the tables in ipp_hal_reduce() enable the combination.

/////////////////////////////// dim = 1: reduce each source row ///////////////////////////////

// src: one row of 'width' pixels, dst: the 'cn' results for this row
typedef IppStatus (*IppReduceRowFunc)(const void* src, int width, void* dst);

// ippiSum_*: the sum is computed in Ipp64f and converted to the destination depth DT
template<typename T, int cn, typename DT, IppStatus (CV_STDCALL *fn)(const T*, int, IppiSize, Ipp64f*)>
static IppStatus ippReduceRowSum(const void* src, int width, void* dst)
{
    Ipp64f sum[4];
    const T* row = (const T*)src;
    // ippiSum_16s: long rows are not exact for some extreme values, sum them in blocks.
    const int blockWidth = std::is_same<T, Ipp16s>::value ? std::min(width, 1 << 16) : width;
    IppiSize roi = { blockWidth, 1 };
    IppStatus status = CV_INSTRUMENT_FUN_IPP(fn, row, blockWidth * cn * (int)sizeof(T), roi, sum);
    if(status < 0)
        return status;
    for(int x = blockWidth; x < width; x += roi.width)
    {
        roi.width = std::min(width - x, blockWidth);
        Ipp64f partial[4];
        status = CV_INSTRUMENT_FUN_IPP(fn, row + (size_t)x * cn, roi.width * cn * (int)sizeof(T), roi, partial);
        if(status < 0)
            return status;
        for(int c = 0; c < cn; c++)
            sum[c] += partial[c];
    }
    for(int c = 0; c < cn; c++)
        ((DT*)dst)[c] = cv::saturate_cast<DT>(sum[c]);
    return status;
}

template<int cn, typename DT, IppStatus (CV_STDCALL *fn)(const Ipp32f*, int, IppiSize, Ipp64f*, IppHintAlgorithm)>
static IppStatus ippReduceRowSum32f(const void* src, int width, void* dst)
{
    Ipp64f sum[4];
    const IppiSize roi = { width, 1 };
    const IppStatus status = CV_INSTRUMENT_FUN_IPP(fn, (const Ipp32f*)src, width * cn * (int)sizeof(Ipp32f), roi, sum, ippAlgHintAccurate);
    if(status < 0)
        return status;
    for(int c = 0; c < cn; c++)
        ((DT*)dst)[c] = cv::saturate_cast<DT>(sum[c]);
    return status;
}

// ippiMax_*, ippiMin_*
template<typename T, int cn, IppStatus (CV_STDCALL *fn)(const T*, int, IppiSize, T*)>
static IppStatus ippReduceRowMinMax(const void* src, int width, void* dst)
{
    const IppiSize roi = { width, 1 };
    return CV_INSTRUMENT_FUN_IPP(fn, (const T*)src, width * cn * (int)sizeof(T), roi, (T*)dst);
}

// ippiDotProd_* of the row with itself: the sum of squares is computed in Ipp64f and converted to the destination depth DT
template<typename T, int cn, typename DT, IppStatus (CV_STDCALL *fn)(const T*, int, const T*, int, IppiSize, Ipp64f*)>
static IppStatus ippReduceRowSumSqr(const void* src, int width, void* dst)
{
    Ipp64f sum[4];
    const IppiSize roi = { width, 1 };
    const int step = width * cn * (int)sizeof(T);
    const IppStatus status = CV_INSTRUMENT_FUN_IPP(fn, (const T*)src, step, (const T*)src, step, roi, sum);
    if(status < 0)
        return status;
    for(int c = 0; c < cn; c++)
        ((DT*)dst)[c] = cv::saturate_cast<DT>(sum[c]);
    return status;
}

template<int cn, typename DT, IppStatus (CV_STDCALL *fn)(const Ipp32f*, int, const Ipp32f*, int, IppiSize, Ipp64f*, IppHintAlgorithm)>
static IppStatus ippReduceRowSumSqr32f(const void* src, int width, void* dst)
{
    Ipp64f sum[4];
    const IppiSize roi = { width, 1 };
    const int step = width * cn * (int)sizeof(Ipp32f);
    const IppStatus status = CV_INSTRUMENT_FUN_IPP(fn, (const Ipp32f*)src, step, (const Ipp32f*)src, step, roi, sum, ippAlgHintAccurate);
    if(status < 0)
        return status;
    for(int c = 0; c < cn; c++)
        ((DT*)dst)[c] = cv::saturate_cast<DT>(sum[c]);
    return status;
}

static IppStatus ippReduceRowSumSqr64f(const void* src, int width, void* dst)
{
    return CV_INSTRUMENT_FUN_IPP(ippsDotProd_64f, (const Ipp64f*)src, (const Ipp64f*)src, width, (Ipp64f*)dst);
}

#if !defined(HAVE_IPP_ICV)
// ipps functions for single channel rows: ippsSum_64f, ippsMax_*, ippsMin_*
template<typename T, IppStatus (CV_STDCALL *fn)(const T*, int, T*)>
static IppStatus ippsReduceRow(const void* src, int width, void* dst)
{
    return CV_INSTRUMENT_FUN_IPP(fn, (const T*)src, width, (T*)dst);
}
#endif

#define IPP_REDUCE_ROW_SUM(T, cn, fn) \
    (ddepth == CV_32S ? (IppReduceRowFunc)ippReduceRowSum<T, cn, int, fn>   : \
     ddepth == CV_32F ? (IppReduceRowFunc)ippReduceRowSum<T, cn, float, fn> : \
     ddepth == CV_64F ? (IppReduceRowFunc)ippReduceRowSum<T, cn, double, fn> : (IppReduceRowFunc)0)

#define IPP_REDUCE_ROW_SUM_32F(cn, fn) \
    (ddepth == CV_32F ? (IppReduceRowFunc)ippReduceRowSum32f<cn, float, fn>  : \
     ddepth == CV_64F ? (IppReduceRowFunc)ippReduceRowSum32f<cn, double, fn> : (IppReduceRowFunc)0)

#define IPP_REDUCE_ROW_SUMSQR(T, cn, fn) \
    (ddepth == CV_32S ? (IppReduceRowFunc)ippReduceRowSumSqr<T, cn, int, fn>   : \
     ddepth == CV_32F ? (IppReduceRowFunc)ippReduceRowSumSqr<T, cn, float, fn> : \
     ddepth == CV_64F ? (IppReduceRowFunc)ippReduceRowSumSqr<T, cn, double, fn> : (IppReduceRowFunc)0)

#define IPP_REDUCE_ROW_SUMSQR_32F(cn, fn) \
    (ddepth == CV_32F ? (IppReduceRowFunc)ippReduceRowSumSqr32f<cn, float, fn>  : \
     ddepth == CV_64F ? (IppReduceRowFunc)ippReduceRowSumSqr32f<cn, double, fn> : (IppReduceRowFunc)0)

#define IPP_REDUCE_ROW_MINMAX(T, cn, fnMax, fnMin) \
    (isMax ? (IppReduceRowFunc)ippReduceRowMinMax<T, cn, fnMax> : (IppReduceRowFunc)ippReduceRowMinMax<T, cn, fnMin>)

static IppReduceRowFunc getReduceRowFunc(int op, int src_type, int ddepth)
{
    const bool isMax = op == cv::REDUCE_MAX;
    if(op == cv::REDUCE_SUM)
    {
        switch(src_type)
        {
        case CV_8UC1:  return IPP_REDUCE_ROW_SUM(Ipp8u,  1, ippiSum_8u_C1R);
        case CV_8UC3:  return IPP_REDUCE_ROW_SUM(Ipp8u,  3, ippiSum_8u_C3R);
        case CV_8UC4:  return IPP_REDUCE_ROW_SUM(Ipp8u,  4, ippiSum_8u_C4R);
        case CV_16UC1: return IPP_REDUCE_ROW_SUM(Ipp16u, 1, ippiSum_16u_C1R);
        case CV_16UC3: return IPP_REDUCE_ROW_SUM(Ipp16u, 3, ippiSum_16u_C3R);
        case CV_16UC4: return IPP_REDUCE_ROW_SUM(Ipp16u, 4, ippiSum_16u_C4R);
        case CV_16SC1: return IPP_REDUCE_ROW_SUM(Ipp16s, 1, ippiSum_16s_C1R);
        case CV_16SC3: return IPP_REDUCE_ROW_SUM(Ipp16s, 3, ippiSum_16s_C3R);
        case CV_16SC4: return IPP_REDUCE_ROW_SUM(Ipp16s, 4, ippiSum_16s_C4R);
        case CV_32FC1: return IPP_REDUCE_ROW_SUM_32F(1, ippiSum_32f_C1R);
        case CV_32FC3: return IPP_REDUCE_ROW_SUM_32F(3, ippiSum_32f_C3R);
        case CV_32FC4: return IPP_REDUCE_ROW_SUM_32F(4, ippiSum_32f_C4R);
#if !defined(HAVE_IPP_ICV)
        case CV_64FC1: return ddepth == CV_64F ? ippsReduceRow<Ipp64f, ippsSum_64f> : 0;
#endif
        default:       return 0;
        }
    }
    if(op == cv::REDUCE_MAX || op == cv::REDUCE_MIN)
    {
        switch(src_type)
        {
        case CV_8UC1:  return IPP_REDUCE_ROW_MINMAX(Ipp8u,  1, ippiMax_8u_C1R,  ippiMin_8u_C1R);
        case CV_16UC1: return IPP_REDUCE_ROW_MINMAX(Ipp16u, 1, ippiMax_16u_C1R, ippiMin_16u_C1R);
        case CV_16SC1: return IPP_REDUCE_ROW_MINMAX(Ipp16s, 1, ippiMax_16s_C1R, ippiMin_16s_C1R);
        case CV_32FC1: return IPP_REDUCE_ROW_MINMAX(Ipp32f, 1, ippiMax_32f_C1R, ippiMin_32f_C1R);
#if !defined(HAVE_IPP_ICV)
        case CV_8UC3:  return IPP_REDUCE_ROW_MINMAX(Ipp8u,  3, ippiMax_8u_C3R,  ippiMin_8u_C3R);
        case CV_8UC4:  return IPP_REDUCE_ROW_MINMAX(Ipp8u,  4, ippiMax_8u_C4R,  ippiMin_8u_C4R);
        case CV_16UC3: return IPP_REDUCE_ROW_MINMAX(Ipp16u, 3, ippiMax_16u_C3R, ippiMin_16u_C3R);
        case CV_16UC4: return IPP_REDUCE_ROW_MINMAX(Ipp16u, 4, ippiMax_16u_C4R, ippiMin_16u_C4R);
        case CV_16SC3: return IPP_REDUCE_ROW_MINMAX(Ipp16s, 3, ippiMax_16s_C3R, ippiMin_16s_C3R);
        case CV_16SC4: return IPP_REDUCE_ROW_MINMAX(Ipp16s, 4, ippiMax_16s_C4R, ippiMin_16s_C4R);
        case CV_32FC3: return IPP_REDUCE_ROW_MINMAX(Ipp32f, 3, ippiMax_32f_C3R, ippiMin_32f_C3R);
        case CV_32FC4: return IPP_REDUCE_ROW_MINMAX(Ipp32f, 4, ippiMax_32f_C4R, ippiMin_32f_C4R);
        case CV_32SC1: return isMax ? ippsReduceRow<Ipp32s, ippsMax_32s> : ippsReduceRow<Ipp32s, ippsMin_32s>;
        case CV_64FC1: return isMax ? ippsReduceRow<Ipp64f, ippsMax_64f> : ippsReduceRow<Ipp64f, ippsMin_64f>;
#endif
        default:       return 0;
        }
    }
    if(op == cv::REDUCE_SUM2)
    {
        switch(src_type)
        {
        case CV_8UC1:  return IPP_REDUCE_ROW_SUMSQR(Ipp8u,  1, ippiDotProd_8u64f_C1R);
        case CV_16UC1: return IPP_REDUCE_ROW_SUMSQR(Ipp16u, 1, ippiDotProd_16u64f_C1R);
        case CV_16SC1: return IPP_REDUCE_ROW_SUMSQR(Ipp16s, 1, ippiDotProd_16s64f_C1R);
        case CV_32SC1: return IPP_REDUCE_ROW_SUMSQR(Ipp32s, 1, ippiDotProd_32s64f_C1R);
        case CV_32FC1: return IPP_REDUCE_ROW_SUMSQR_32F(1, ippiDotProd_32f64f_C1R);
        case CV_64FC1: return ddepth == CV_64F ? ippReduceRowSumSqr64f : 0;
#if !defined(HAVE_IPP_ICV)
        case CV_8UC3:  return IPP_REDUCE_ROW_SUMSQR(Ipp8u,  3, ippiDotProd_8u64f_C3R);
        case CV_8UC4:  return IPP_REDUCE_ROW_SUMSQR(Ipp8u,  4, ippiDotProd_8u64f_C4R);
        case CV_16UC3: return IPP_REDUCE_ROW_SUMSQR(Ipp16u, 3, ippiDotProd_16u64f_C3R);
        case CV_16UC4: return IPP_REDUCE_ROW_SUMSQR(Ipp16u, 4, ippiDotProd_16u64f_C4R);
        case CV_16SC3: return IPP_REDUCE_ROW_SUMSQR(Ipp16s, 3, ippiDotProd_16s64f_C3R);
        case CV_16SC4: return IPP_REDUCE_ROW_SUMSQR(Ipp16s, 4, ippiDotProd_16s64f_C4R);
        case CV_32SC3: return IPP_REDUCE_ROW_SUMSQR(Ipp32s, 3, ippiDotProd_32s64f_C3R);
        case CV_32SC4: return IPP_REDUCE_ROW_SUMSQR(Ipp32s, 4, ippiDotProd_32s64f_C4R);
        case CV_32FC3: return IPP_REDUCE_ROW_SUMSQR_32F(3, ippiDotProd_32f64f_C3R);
        case CV_32FC4: return IPP_REDUCE_ROW_SUMSQR_32F(4, ippiDotProd_32f64f_C4R);
#endif
        default:       return 0;
        }
    }
    return 0;
}

#undef IPP_REDUCE_ROW_SUM
#undef IPP_REDUCE_ROW_SUM_32F
#undef IPP_REDUCE_ROW_MINMAX
#undef IPP_REDUCE_ROW_SUMSQR
#undef IPP_REDUCE_ROW_SUMSQR_32F

/////////////////////// dim = 0: every source row is accumulated into the destination row ///////////////////////

// Element-wise, so any number of channels is processed as 'len' single channel elements
typedef IppStatus (*IppReduceAccFunc)(const void* src, void* acc, int len);

// ippiAdd_*_C1IR, ippiAddSquare_*_C1IR, ippiMaxEvery/MinEvery_*_C1IR
template<typename T, typename AT, IppStatus (CV_STDCALL *fn)(const T*, int, AT*, int, IppiSize)>
static IppStatus ippReduceAcc(const void* src, void* acc, int len)
{
    const IppiSize roi = { len, 1 };
    return CV_INSTRUMENT_FUN_IPP(fn, (const T*)src, len * (int)sizeof(T), (AT*)acc, len * (int)sizeof(AT), roi);
}

// ippsMaxEvery/MinEvery_* with separate source and destination pointers (acc = op(acc, src))
template<typename T, IppStatus (CV_STDCALL *fn)(const T*, const T*, T*, Ipp32u)>
static IppStatus ippsReduceAcc(const void* src, void* acc, int len)
{
    return CV_INSTRUMENT_FUN_IPP(fn, (const T*)acc, (const T*)src, (T*)acc, (Ipp32u)len);
}

#if !defined(HAVE_IPP_ICV)
// in-place ipps functions: ippsAdd_64f_I, ippsMaxEvery/MinEvery_*_I
template<typename T, IppStatus (CV_STDCALL *fn)(const T*, T*, int)>
static IppStatus ippsReduceAccI(const void* src, void* acc, int len)
{
    return CV_INSTRUMENT_FUN_IPP(fn, (const T*)src, (T*)acc, len);
}
#endif

static IppReduceAccFunc getReduceAccFunc(int op, int depth, int ddepth)
{
    const bool isMax = op == cv::REDUCE_MAX;
    if(op == cv::REDUCE_SUM)
    {
        if(depth == CV_8U  && ddepth == CV_32F) return ippReduceAcc<Ipp8u,  Ipp32f, ippiAdd_8u32f_C1IR>;
        if(depth == CV_16U && ddepth == CV_32F) return ippReduceAcc<Ipp16u, Ipp32f, ippiAdd_16u32f_C1IR>;
        if(depth == CV_32F && ddepth == CV_32F) return ippReduceAcc<Ipp32f, Ipp32f, ippiAdd_32f_C1IR>;
#if !defined(HAVE_IPP_ICV)
        if(depth == CV_64F && ddepth == CV_64F) return ippsReduceAccI<Ipp64f, ippsAdd_64f_I>;
#endif
        return 0;
    }
    if(op == cv::REDUCE_SUM2)
    {
        if(depth == CV_8U  && ddepth == CV_32F) return ippReduceAcc<Ipp8u,  Ipp32f, ippiAddSquare_8u32f_C1IR>;
        if(depth == CV_16U && ddepth == CV_32F) return ippReduceAcc<Ipp16u, Ipp32f, ippiAddSquare_16u32f_C1IR>;
        if(depth == CV_32F && ddepth == CV_32F) return ippReduceAcc<Ipp32f, Ipp32f, ippiAddSquare_32f_C1IR>;
        return 0;
    }
    if((op == cv::REDUCE_MAX || op == cv::REDUCE_MIN) && ddepth == depth)
    {
        switch(depth)
        {
        case CV_8U:  return isMax ? ippsReduceAcc<Ipp8u,  ippsMaxEvery_8u>  : ippsReduceAcc<Ipp8u,  ippsMinEvery_8u>;
        case CV_16U: return isMax ? ippsReduceAcc<Ipp16u, ippsMaxEvery_16u> : ippsReduceAcc<Ipp16u, ippsMinEvery_16u>;
        case CV_32F: return isMax ? ippsReduceAcc<Ipp32f, ippsMaxEvery_32f> : ippsReduceAcc<Ipp32f, ippsMinEvery_32f>;
        case CV_64F: return isMax ? ippsReduceAcc<Ipp64f, ippsMaxEvery_64f> : ippsReduceAcc<Ipp64f, ippsMinEvery_64f>;
#if !defined(HAVE_IPP_ICV)
        case CV_16S: return isMax ? ippReduceAcc<Ipp16s, Ipp16s, ippiMaxEvery_16s_C1IR> : ippReduceAcc<Ipp16s, Ipp16s, ippiMinEvery_16s_C1IR>;
        case CV_32S: return isMax ? ippsReduceAccI<Ipp32s, ippsMaxEvery_32s_I> : ippsReduceAccI<Ipp32s, ippsMinEvery_32s_I>;
#endif
        default:     return 0;
        }
    }
    return 0;
}

int ipp_hal_reduce(const uchar* src_data, size_t src_step, int src_type, int width, int height,
                   uchar* dst_data, size_t dst_step, int dst_type, int dim, int op)
{
    CV_HAL_CHECK_USE_IPP();

    const int depth  = CV_MAT_DEPTH(src_type);
    const int cn     = CV_MAT_CN(src_type);
    const int ddepth = CV_MAT_DEPTH(dst_type);
    if(depth > CV_64F || ddepth > CV_64F || cn < 1 || cn > 4 || width <= 0 || height <= 0 || (dim != 0 && dim != 1))
        return CV_HAL_ERROR_NOT_IMPLEMENTED;
    if(op != cv::REDUCE_SUM && op != cv::REDUCE_SUM2 && op != cv::REDUCE_MAX && op != cv::REDUCE_MIN)
        return CV_HAL_ERROR_NOT_IMPLEMENTED;

    // Supported combinations: bit masks of the destination depths. MIN/MAX: the destination has the source depth.
    // dim = 1 tables are [depth][channels - 1]. dim = 0 processes all channels as one, its tables are [depth].
    // In ICV builds, the combinations that need full IPP functions are declined by the getters above.
    enum { S = 1 << CV_32S, F = 1 << CV_32F, D = 1 << CV_64F };
#if defined(IPP_CALLS_ENFORCED)
    static const uchar sumRow[CV_64F + 1][4] = { /* C1     C2  C3     C4    */
        /* 8U  */ {S|F|D, 0, S|F|D, S|F|D},
        /* 8S  */ {0,     0, 0,     0    },
        /* 16U */ {F|D,   0, F|D,   F|D  },
        /* 16S */ {F|D,   0, F|D,   F|D  },
        /* 32S */ {0,     0, 0,     0    },
        /* 32F */ {F|D,   0, F|D,   F|D  },
        /* 64F */ {D,     0, 0,     0    }};
    static const uchar sumSqrRow[CV_64F + 1][4] = { /* C1     C2  C3     C4    */
        /* 8U  */ {S|F|D, 0, S|F|D, S|F|D},
        /* 8S  */ {0,     0, 0,     0    },
        /* 16U */ {F|D,   0, F|D,   F|D  },
        /* 16S */ {F|D,   0, F|D,   F|D  },
        /* 32S */ {D,     0, D,     D    },
        /* 32F */ {F|D,   0, F|D,   F|D  },
        /* 64F */ {D,     0, 0,     0    }};
    static const uchar minMaxRow[CV_64F + 1][4] = {
        /* 8U  */ {1, 0, 1, 1},
        /* 8S  */ {0, 0, 0, 0},
        /* 16U */ {1, 0, 1, 1},
        /* 16S */ {1, 0, 1, 1},
        /* 32S */ {1, 0, 0, 0},
        /* 32F */ {1, 0, 1, 1},
        /* 64F */ {1, 0, 0, 0}};
    //                                           8U 8S 16U 16S 32S 32F 64F
    static const uchar sumCol[CV_64F + 1]    = { F, 0, F,  0,  0,  F,  D };
    static const uchar sumSqrCol[CV_64F + 1] = { F, 0, F,  0,  0,  F,  0 };
    static const uchar minMaxCol[CV_64F + 1] = { 1, 0, 1,  1,  1,  1,  1 };
#else // IPP_CALLS_ENFORCED is not defined: combinations that are faster than OpenCV and keep OpenCV accuracy
    // Floating-point sums are accumulated in Ipp64f and converted once: the results may differ from OpenCV
    // in the last bits (different summation order), but not in accuracy.
    static const uchar sumRow[CV_64F + 1][4] = { /* C1     C2  C3     C4    */
        /* 8U  */ {D,     0, S|F|D, D    },  // C1/C4 to 32S/32F: slower than OpenCV SIMD for short rows
        /* 8S  */ {0,     0, 0,     0    },
        /* 16U */ {D,     0, F|D,   F|D  },  // C1 to 32F: slower than OpenCV SIMD for short rows
        /* 16S */ {D,     0, F|D,   F|D  },  // C1 to 32F: slower than OpenCV SIMD for short rows
        /* 32S */ {0,     0, 0,     0    },
        /* 32F */ {0,     0, F|D,   F|D  },  // C1: slower than OpenCV SIMD for short rows
        /* 64F */ {0,     0, 0,     0    }}; // slower than OpenCV SIMD for short rows
    static const uchar sumSqrRow[CV_64F + 1][4] = { /* C1     C2  C3     C4    */
        /* 8U  */ {S|F|D, 0, S|F|D, S|F|D},
        /* 8S  */ {0,     0, 0,     0    },
        /* 16U */ {F|D,   0, F|D,   F|D  },
        /* 16S */ {0,     0, 0,     0    },  // ippiDotProd_16s64f: not exact for some extreme values
        /* 32S */ {D,     0, D,     D    },
        /* 32F */ {F|D,   0, F|D,   F|D  },
        /* 64F */ {D,     0, 0,     0    }};
    static const uchar minMaxRow[CV_64F + 1][4] = {
        /* 8U  */ {1, 0, 1, 1},
        /* 8S  */ {0, 0, 0, 0},
        /* 16U */ {1, 0, 1, 1},
        /* 16S */ {1, 0, 1, 1},
        /* 32S */ {1, 0, 0, 0},
        /* 32F */ {0, 0, 0, 0},  // NaN: IPP ignores a leading NaN and returns NaN for a trailing one, OpenCV does the opposite
        /* 64F */ {0, 0, 0, 0}}; // NaN, as 32F
    // dim = 0 is not faster than OpenCV with 1 thread and slower with many threads
    // (a stripe needs long rows to pay for one IPP call per row).
    // Exactness if enabled: 8U/16U SUM and 8U SUM2 into 32F only while the column sums are below 2^24
    // (OpenCV accumulates in integers, IPP in float); 16U SUM2 rounds the squares, 32F SUM2 differs from FMA,
    // 32F/64F MIN/MAX differ for NaN.
    //                                           8U 8S 16U 16S 32S 32F 64F
    static const uchar sumCol[CV_64F + 1]    = { 0, 0, 0,  0,  0,  0,  0 };
    static const uchar sumSqrCol[CV_64F + 1] = { 0, 0, 0,  0,  0,  0,  0 };
    static const uchar minMaxCol[CV_64F + 1] = { 0, 0, 0,  0,  0,  0,  0 };
#endif // IPP_CALLS_ENFORCED

    int dstDepths;
    if(dim == 1)
        dstDepths = op == cv::REDUCE_SUM  ? sumRow[depth][cn - 1] :
                    op == cv::REDUCE_SUM2 ? sumSqrRow[depth][cn - 1] :
                    minMaxRow[depth][cn - 1] ? 1 << depth : 0;
    else
        dstDepths = op == cv::REDUCE_SUM  ? sumCol[depth] :
                    op == cv::REDUCE_SUM2 ? sumSqrCol[depth] :
                    minMaxCol[depth] ? 1 << depth : 0;
    if(!(dstDepths & (1 << ddepth)))
        return CV_HAL_ERROR_NOT_IMPLEMENTED;

    // Every IPP call processes a single row of 'len' elements, the steps passed to IPP are row sizes
    const size_t esz  = CV_ELEM_SIZE1(depth);
    const size_t desz = CV_ELEM_SIZE1(ddepth);
    const int64  len  = (int64)width * cn;
    if(len * (int64)std::max(esz, desz) > INT_MAX)
        return CV_HAL_ERROR_NOT_IMPLEMENTED;

    // Don't narrow a potentially overflowing sum into 32S (also used as an AVG intermediate).
    // ippiSum_8u_C4R: not exact for very long rows.
    if(dim == 1 && op == cv::REDUCE_SUM && depth == CV_8U && width > INT_MAX / 255 && (ddepth == CV_32S || cn == 4))
        return CV_HAL_ERROR_NOT_IMPLEMENTED;

    // 8U SUM2 (into any depth): OpenCV accumulates the squares in int
    if(dim == 1 && op == cv::REDUCE_SUM2 && depth == CV_8U && len * 255 * 255 > INT_MAX)
        return CV_HAL_ERROR_NOT_IMPLEMENTED;

#if !defined(IPP_CALLS_ENFORCED)
    // One IPP call per row is slower than OpenCV for short rows
    if(len < 16)
        return CV_HAL_ERROR_NOT_IMPLEMENTED;

#endif // IPP_CALLS_ENFORCED

    // IPP writes the destination while it still reads the source
    {
        const uchar* srcEnd = src_data + (size_t)(height - 1) * src_step + (size_t)len * esz;
        const uchar* dstEnd = dim == 1 ? dst_data + (size_t)(height - 1) * dst_step + cn * desz
                                       : dst_data + (size_t)len * desz;
        if(dst_data < srcEnd && src_data < dstEnd)
            return CV_HAL_ERROR_NOT_IMPLEMENTED;
    }

    // IPP errors depend only on the arguments; the destination doesn't overlap the source,
    // so OpenCV can always recompute it after a failure.
    std::atomic_bool ok{true};
    bool parallel = false;
    if(dim == 1)
    {
        IppReduceRowFunc reduceRow = getReduceRowFunc(op, src_type, ddepth);
        if(!reduceRow)  // full IPP function in an ICV build
            return CV_HAL_ERROR_NOT_IMPLEMENTED;

        // Every row is independent. A stripe gets at least 64K elements: with one stripe per thread (OpenCV reduceC_)
        // starting the threads takes longer than the IPP calls.
        const int nstripes = (int)std::max((int64)1, len * height / (64 * 1024));
        cv::parallel_for_(cv::Range(0, height), [&](const cv::Range& range)
        {
            for(int y = range.start; y < range.end; y++)
            {
                if(!ok.load(std::memory_order_relaxed))
                    return;
                if(reduceRow(src_data + (size_t)y * src_step, width, dst_data + (size_t)y * dst_step) < 0)
                    ok.store(false, std::memory_order_relaxed);
            }
        }, nstripes);
        parallel = nstripes > 1;
    }
    else
    {
        IppReduceAccFunc accumulate = getReduceAccFunc(op, depth, ddepth);
        if(!accumulate)  // full IPP function in an ICV build
            return CV_HAL_ERROR_NOT_IMPLEMENTED;

        // MIN/MAX and sums into the source depth start from the first row, like OpenCV;
        // sums into a wider depth and SUM2 start from zero
        const bool fromFirstRow = op == cv::REDUCE_MAX || op == cv::REDUCE_MIN || (op == cv::REDUCE_SUM && ddepth == depth);

        // Column stripes, every stripe accumulates all rows. OpenCV uses 64 byte stripes; one IPP call per row and
        // stripe needs longer stripes to pay for the call.
        const int nstripes = (int)std::max((int64)1, len * (int64)std::max(esz, desz) / 1024);
        cv::parallel_for_(cv::Range(0, (int)len), [&](const cv::Range& range)
        {
            const int n = range.end - range.start;
            uchar* acc = dst_data + (size_t)range.start * desz;
            const uchar* src = src_data + (size_t)range.start * esz;
            int y = 0;
            if(fromFirstRow)
            {
                memcpy(acc, src, n * esz);
                y = 1;
            }
            else
                memset(acc, 0, n * desz);
            for(; y < height; y++)
            {
                if(!ok.load(std::memory_order_relaxed))
                    return;
                if(accumulate(src + (size_t)y * src_step, acc, n) < 0)
                    ok.store(false, std::memory_order_relaxed);
            }
        }, nstripes);
        parallel = nstripes > 1;
    }
    if(!ok)
        return CV_HAL_ERROR_NOT_IMPLEMENTED;

    CV_UNUSED(parallel);  // CV_IMPL_ADD is empty unless implementation collection is enabled
    CV_IMPL_ADD(parallel ? CV_IMPL_IPP|CV_IMPL_MT : CV_IMPL_IPP);
    return CV_HAL_ERROR_OK;
}

#endif // IPP_VERSION_X100 >= 700

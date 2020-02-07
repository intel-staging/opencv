/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2016-2016, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective owners.
//
// @Authors
//    Pavel Vlasov, pavel.vlasov@intel.com
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors as is and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#ifndef __OPENCV_PERF_IPP__
#define __OPENCV_PERF_IPP__

#include "ts_perf.hpp"

#include "opencv2/opencv_modules.hpp"

#include "opencv2/ts.hpp"
#include "opencv2/core/ocl.hpp"
#include "opencv2/core/ovx.hpp"
#include "opencv2/core/private.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"

using namespace perf;

#ifdef _MSC_VER
    #pragma warning(disable:4505)
#endif

// For slow Atom chips
#define IPP_TEST_SLOW 0

#define IPP_TEST_MODE_ALL_BORDERS      0x01 // Include all borders into testing of filters
#define IPP_TEST_MODE_CONFORMANCE_ONLY 0x02 // No performance loops, only conformance checks
#define IPP_TEST_MODE_LESS_SIZES       0x04 // Remove sizes variations for faster checks
#define IPP_TEST_MODE_MORE_SIZES       0x08 // Add odd sizes to improve conformance checks

#if IPP_TEST_SLOW == 1
#define IPP_TEST_MODE (IPP_TEST_MODE_LESS_SIZES)
#else
#define IPP_TEST_MODE (0)
#endif

#define IPP_ON(FUN)  cv::ocl::setUseOpenCL(false); cv::setUseOpenVX(false); cv::ipp::setUseIPP(true); FUN;
#define IPP_OFF(FUN) cv::ocl::setUseOpenCL(false); cv::setUseOpenVX(false); cv::ipp::setUseIPP(false); FUN;

// Treat reference exceptions as skipped to avoid messing of OpenCV and IPP errors
#define RUN_REF(FUN)\
try {\
    IPP_OFF(FUN);\
}\
catch(cv::Exception)\
{\
    throw ::perf::TestBase::PerfSkipTestException();\
}
#define RUN_IPP(FUN) IPP_ON(FUN)

#define VERIFY_DATA(DST, REF, INT_THRES, FLOAT_THRES)\
{\
    if(!tsCompareImagesMaxDiff(DST, REF, INT_THRES, FLOAT_THRES, FLOAT_THRES))\
    {\
        this->setVerified();\
    }\
    this->setVerified();\
}

#define VERIFY_DATA_PRES(DST, REF, INT_THRES, FLOAT_THRES, DOUBLE_THRES)\
{\
    if(!tsCompareImagesMaxDiff(DST, REF, INT_THRES, FLOAT_THRES, DOUBLE_THRES))\
    {\
        this->setVerified();\
    }\
    this->setVerified();\
}

#define VERIFY_DATA_MATCHES(DST, REF, THRES)\
{\
    if(!tsCompareImagesMaxMismatch(DST, REF, THRES))\
    {\
        this->setVerified();\
    }\
    this->setVerified();\
}

#define TS_DEFAULT_FLOAT_MIN -1.
#define TS_DEFAULT_FLOAT_MAX 1.

enum BorderTypesEx {
    BORDER_INMEM = 6
};

#define CV_MAT_PAIR2(FIRST, SECOND)               (((SECOND)<<4)|(FIRST))
#define CV_MAT_PAIR3(FIRST, SECOND, THIRD)        (((THIRD)<<8)|((SECOND)<<4)|(FIRST))
#define CV_MAT_PAIR_SET_CHANNELS(PAIR, CHANNELS) ((PAIR)|((CHANNELS)<<28))
#define CV_MAT_PAIR_GET1(PAIR)  CV_MAKE_TYPE(((PAIR)&0xF), (PAIR)>>28)
#define CV_MAT_PAIR_GET2(PAIR)  CV_MAKE_TYPE(((PAIR)&0xF0)>>4, (PAIR)>>28)
#define CV_MAT_PAIR_GET3(PAIR)  CV_MAKE_TYPE(((PAIR)&0xF00)>>8, (PAIR)>>28)

enum MatPairsEnum
{
    CV_8U_8U_C1        = CV_MAT_PAIR_SET_CHANNELS(CV_MAT_PAIR2(CV_8U,   CV_8U), 1),
    CV_8U_8U_C3        = CV_MAT_PAIR_SET_CHANNELS(CV_MAT_PAIR2(CV_8U,   CV_8U), 3),
    CV_8U_8U_C4        = CV_MAT_PAIR_SET_CHANNELS(CV_MAT_PAIR2(CV_8U,   CV_8U), 4),
    CV_8U_32S_C1       = CV_MAT_PAIR_SET_CHANNELS(CV_MAT_PAIR2(CV_8U,   CV_32S), 1),
    CV_8U_32S_C3       = CV_MAT_PAIR_SET_CHANNELS(CV_MAT_PAIR2(CV_8U,   CV_32S), 3),
    CV_8U_32S_C4       = CV_MAT_PAIR_SET_CHANNELS(CV_MAT_PAIR2(CV_8U,   CV_32S), 4),
    CV_8U_32S_32S_C1   = CV_MAT_PAIR_SET_CHANNELS(CV_MAT_PAIR3(CV_8U,   CV_32S, CV_32S), 1),
    CV_8U_32S_64F_C1   = CV_MAT_PAIR_SET_CHANNELS(CV_MAT_PAIR3(CV_8U,   CV_32S, CV_64F), 1),
    CV_8U_32F_64F_C1   = CV_MAT_PAIR_SET_CHANNELS(CV_MAT_PAIR3(CV_8U,   CV_32F, CV_64F), 1),
    CV_8U_32F_C1       = CV_MAT_PAIR_SET_CHANNELS(CV_MAT_PAIR2(CV_8U,   CV_32F), 1),
    CV_8U_32F_C3       = CV_MAT_PAIR_SET_CHANNELS(CV_MAT_PAIR2(CV_8U,   CV_32F), 3),
    CV_8U_32F_C4       = CV_MAT_PAIR_SET_CHANNELS(CV_MAT_PAIR2(CV_8U,   CV_32F), 4),
    CV_8U_64F_C1       = CV_MAT_PAIR_SET_CHANNELS(CV_MAT_PAIR2(CV_8U,   CV_64F), 1),
    CV_8U_64F_C3       = CV_MAT_PAIR_SET_CHANNELS(CV_MAT_PAIR2(CV_8U,   CV_64F), 3),
    CV_8U_64F_C4       = CV_MAT_PAIR_SET_CHANNELS(CV_MAT_PAIR2(CV_8U,   CV_64F), 4),
    CV_16U_32F_C1      = CV_MAT_PAIR_SET_CHANNELS(CV_MAT_PAIR2(CV_16U,  CV_32F), 1),
    CV_16U_32F_C3      = CV_MAT_PAIR_SET_CHANNELS(CV_MAT_PAIR2(CV_16U,  CV_32F), 3),
    CV_16U_32F_C4      = CV_MAT_PAIR_SET_CHANNELS(CV_MAT_PAIR2(CV_16U,  CV_32F), 4),
    CV_16U_64F_C1      = CV_MAT_PAIR_SET_CHANNELS(CV_MAT_PAIR2(CV_16U,  CV_64F), 1),
    CV_16U_64F_C3      = CV_MAT_PAIR_SET_CHANNELS(CV_MAT_PAIR2(CV_16U,  CV_64F), 3),
    CV_16U_64F_C4      = CV_MAT_PAIR_SET_CHANNELS(CV_MAT_PAIR2(CV_16U,  CV_64F), 4),
    CV_16S_8U_C1       = CV_MAT_PAIR_SET_CHANNELS(CV_MAT_PAIR2(CV_16S,  CV_8U), 1),
    CV_16S_32F_C1      = CV_MAT_PAIR_SET_CHANNELS(CV_MAT_PAIR2(CV_16S,  CV_32F), 1),
    CV_16S_32F_C3      = CV_MAT_PAIR_SET_CHANNELS(CV_MAT_PAIR2(CV_16S,  CV_32F), 3),
    CV_16S_32F_C4      = CV_MAT_PAIR_SET_CHANNELS(CV_MAT_PAIR2(CV_16S,  CV_32F), 4),
    CV_16S_64F_C1      = CV_MAT_PAIR_SET_CHANNELS(CV_MAT_PAIR2(CV_16S,  CV_64F), 1),
    CV_16S_64F_C3      = CV_MAT_PAIR_SET_CHANNELS(CV_MAT_PAIR2(CV_16S,  CV_64F), 3),
    CV_16S_64F_C4      = CV_MAT_PAIR_SET_CHANNELS(CV_MAT_PAIR2(CV_16S,  CV_64F), 4),
    CV_32F_32F_C1      = CV_MAT_PAIR_SET_CHANNELS(CV_MAT_PAIR2(CV_32F,  CV_32F), 1),
    CV_32F_32F_C3      = CV_MAT_PAIR_SET_CHANNELS(CV_MAT_PAIR2(CV_32F,  CV_32F), 3),
    CV_32F_32F_C4      = CV_MAT_PAIR_SET_CHANNELS(CV_MAT_PAIR2(CV_32F,  CV_32F), 4),
    CV_32F_64F_C1      = CV_MAT_PAIR_SET_CHANNELS(CV_MAT_PAIR2(CV_32F,  CV_64F), 1),
    CV_32F_64F_C3      = CV_MAT_PAIR_SET_CHANNELS(CV_MAT_PAIR2(CV_32F,  CV_64F), 3),
    CV_32F_64F_C4      = CV_MAT_PAIR_SET_CHANNELS(CV_MAT_PAIR2(CV_32F,  CV_64F), 4),
    CV_64F_64F_C1      = CV_MAT_PAIR_SET_CHANNELS(CV_MAT_PAIR2(CV_64F,  CV_64F), 1),
    CV_64F_64F_C3      = CV_MAT_PAIR_SET_CHANNELS(CV_MAT_PAIR2(CV_64F,  CV_64F), 3),
    CV_64F_64F_C4      = CV_MAT_PAIR_SET_CHANNELS(CV_MAT_PAIR2(CV_64F,  CV_64F), 4),
};

CV_ENUM(MatPairs, CV_8U_8U_C1, CV_8U_8U_C3, CV_8U_8U_C4, CV_8U_32S_C1, CV_8U_32S_C3, CV_8U_32S_C4, CV_8U_32S_32S_C1, CV_8U_32S_64F_C1, CV_8U_32F_64F_C1, CV_8U_32F_C1, CV_8U_32F_C3, CV_8U_32F_C4, CV_8U_64F_C1, CV_8U_64F_C3, CV_8U_64F_C4,
    CV_16U_32F_C1, CV_16U_32F_C3, CV_16U_32F_C4, CV_16U_64F_C1, CV_16U_64F_C3, CV_16U_64F_C4,
    CV_16S_8U_C1, CV_16S_32F_C1, CV_16S_32F_C3, CV_16S_32F_C4, CV_16S_64F_C1, CV_16S_64F_C3, CV_16S_64F_C4,
    CV_32F_32F_C1, CV_32F_32F_C3, CV_32F_32F_C4, CV_32F_64F_C1, CV_32F_64F_C3, CV_32F_64F_C4,
    CV_64F_64F_C1, CV_64F_64F_C3, CV_64F_64F_C4,
)

#define CMP_EPS_OK 0
#define CMP_EPS_BIG_DIFF -1
#define CMP_EPS_INVALID_TEST_DATA -2 // there is NaN or Inf value in test data
#define CMP_EPS_INVALID_REF_DATA -3 // there is NaN or Inf value in reference data

enum TsThreads
{
    ST,
    MT
};

CV_ENUM(IppTestThreads, ST, MT)

#define TS_SKIP_TEST            throw ::perf::TestBase::PerfSkipTestException()
#define TS_SET_NOT_EXACT        ::cv::ipp::setUseIPP_NotExact(true); ippNotExact = ::cv::ipp::useIPP_NotExact()
#define TS_SET_THREADS(FLAG)    if((FLAG) == ST) {cv::setNumThreads(0);} else if((FLAG) == MT && cv::getNumThreads() == 1) {TS_SKIP_TEST;}

inline int tsCmpMaxEpsFloat(const cv::Mat& arr, const cv::Mat& refarr, double* _realmaxdiff,
                            double success_err_level)
{
    CV_Assert( arr.type() == refarr.type() && arr.size == refarr.size );

    int result = CMP_EPS_OK;

    const cv::Mat *arrays[]={&arr, &refarr, 0};
    cv::Mat planes[2];
    cv::NAryMatIterator it(arrays, planes);
    size_t total = planes[0].total()*planes[0].channels(), j = total;
    size_t i, nplanes = it.nplanes;
    int depth = arr.depth();
    double realmaxdiff = 0, maxval = 0;
    bool tooBigDiff = false;

    if(_realmaxdiff)
        *_realmaxdiff = 0;

    if(depth < CV_32F)
        return CMP_EPS_INVALID_TEST_DATA;

    maxval = cvtest::norm( refarr, cv::NORM_INF );

    for( i = 0; i < nplanes; i++, ++it )
    {
        const uchar* sptr1 = planes[0].ptr();
        const uchar* sptr2 = planes[1].ptr();

        for( j = 0; j < total; j++ )
        {
            double a_val = (depth == CV_32F)?((float*)sptr1)[j]:((double*)sptr1)[j];
            double b_val = (depth == CV_32F)?((float*)sptr2)[j]:((double*)sptr2)[j];
            double relDiff;
            if(a_val == b_val)
                continue;
            if( cvIsNaN(a_val) || cvIsInf(a_val) )
            {
                result = CMP_EPS_INVALID_TEST_DATA;
                break;
            }
            if( cvIsNaN(b_val) || cvIsInf(b_val) )
            {
                result = CMP_EPS_INVALID_REF_DATA;
                break;
            }
            relDiff = fabs(a_val - b_val)/maxval;
            if( relDiff > success_err_level )
            {
                tooBigDiff = true;
                if(realmaxdiff < relDiff)
                    realmaxdiff = relDiff;
            }
        }
        if(_realmaxdiff)
            *_realmaxdiff = MAX(*_realmaxdiff, realmaxdiff);
    }

    if( result == 0 && tooBigDiff )
        result = CMP_EPS_BIG_DIFF;

    if( result < -1 && _realmaxdiff )
        *_realmaxdiff = exp(1000.);

    return result;
}

inline bool tsCompareImagesMaxDiff(cv::Mat &dst, cv::Mat &ref, int intThres, double floatThres, double doubleThres = DBL_MAX)
{
    double eps;
    int    code;

    if(dst.depth() == CV_32F || (dst.depth() == CV_64F && doubleThres == DBL_MAX))
        code = tsCmpMaxEpsFloat(dst, ref, &eps, floatThres);
    else if(dst.depth() == CV_64F)
        code = tsCmpMaxEpsFloat(dst, ref, &eps, doubleThres);
    else
        code = cvtest::cmpEps(dst, ref, &eps, intThres, NULL, false);

    if(!code)
        return true;
    else
    {
        ADD_FAILURE() << "Too big difference\nDIFF =" << eps << "\nTHRES=" << ((dst.depth() == CV_64F)?doubleThres:(dst.depth() == CV_32F)?floatThres:intThres) << "\n";
        return false;
    }
}

inline bool tsCompareImagesMaxMismatch(cv::Mat &dst, cv::Mat &ref, size_t thres)
{
    double eps;
    int    code;
    std::vector<int> idx;

    code = cvtest::cmpEps(dst, ref, &eps, 0, &idx, false);
    if(!code || thres >= idx.size())
        return true;
    else
    {
        ADD_FAILURE() << "Too many mismatched pixels\nMISS =" << idx.size() << "\nTHRES=" << thres << "\n";
        return false;
    }

    return true;
}

inline void tsGenerateDefault(cv::Mat image)
{
    int depth = image.depth();
    if (depth == CV_8U)
        cv::randu(image, 0, UCHAR_MAX+1);
    else if (depth == CV_8S)
        cv::randu(image, SCHAR_MIN, SCHAR_MAX+1);
    else if (depth == CV_16U)
        cv::randu(image, 0, USHRT_MAX+1);
    else if (depth == CV_16S)
        cv::randu(image, SHRT_MIN, SHRT_MAX+1);
    else if (depth == CV_32S)
        cv::randu(image, INT_MIN, (double)(INT_MAX)+1);
    else if (depth == CV_32F || depth == CV_64F)
        cv::randu(image, TS_DEFAULT_FLOAT_MIN, TS_DEFAULT_FLOAT_MAX);
    else
        CV_Error(cv::Error::StsUnsupportedFormat, "Unsupported format");
}


inline double tsTypeGetMin(int depth)
{
    depth = CV_MAT_DEPTH(depth);
    switch(depth)
    {
    case CV_8U:  return 0;
    case CV_8S:  return SCHAR_MIN;
    case CV_16U: return 0;
    case CV_16S: return SHRT_MIN;
    case CV_32S: return INT_MIN;
    case CV_32F:
    case CV_64F: return TS_DEFAULT_FLOAT_MIN;
    default:     return 0;
    }
}

inline double tsTypeGetMax(int depth)
{
    depth = CV_MAT_DEPTH(depth);
    switch(depth)
    {
    case CV_8U:  return UCHAR_MAX;
    case CV_8S:  return SCHAR_MAX;
    case CV_16U: return USHRT_MAX;
    case CV_16S: return SHRT_MAX;
    case CV_32S: return INT_MAX;
    case CV_32F:
    case CV_64F: return TS_DEFAULT_FLOAT_MAX;
    default:     return 0;
    }
}

inline double tsTypeGetRange(int depth)
{
    depth = CV_MAT_DEPTH(depth);
    switch(depth)
    {
    case CV_8U:  return ((double)UCHAR_MAX  - 0);
    case CV_8S:  return ((double)SCHAR_MAX  - SCHAR_MIN);
    case CV_16U: return ((double)USHRT_MAX - 0);
    case CV_16S: return ((double)SHRT_MAX - SHRT_MIN);
    case CV_32S: return ((double)INT_MAX - INT_MIN);
    case CV_32F:
    case CV_64F: return (TS_DEFAULT_FLOAT_MAX - TS_DEFAULT_FLOAT_MIN);
    default:     return 0;
    }
}

inline int tsTypeIsFloat(int depth)
{
    depth = CV_MAT_DEPTH(depth);
    return (depth == CV_32F || depth == CV_64F)?1:0;
}

inline void tsGetScaleVals(int srcDepth, int dstDepth, double *pMulVal, double *pAddVal)
{
    double srcRange = 1, dstRange = 1;
    double srcMin   = 0, dstMin   = 0;

    srcDepth = CV_MAT_DEPTH(srcDepth);
    dstDepth = CV_MAT_DEPTH(dstDepth);

    if(srcDepth == dstDepth)
    {
        *pMulVal = 1;
        *pAddVal = 0;
    }

    if(!tsTypeIsFloat(srcDepth))
    {
        srcRange = tsTypeGetRange(srcDepth);
        srcMin   = tsTypeGetMin(srcDepth);
    }
    if(!tsTypeIsFloat(dstDepth))
    {
        dstRange = tsTypeGetRange(dstDepth);
        dstMin   = tsTypeGetMin(dstDepth);
    }
    if(!srcRange || !dstRange)
        return;

    *pMulVal = dstRange/srcRange;
    *pAddVal = dstMin - srcMin*(*pMulVal);

    return;
}

inline cv::Mat tsSrcSetBorder(cv::Mat &src, int &border, int borderSize = 16)
{
    IPP_OFF(;)

    if((border&~cv::BORDER_ISOLATED) != BORDER_INMEM)
        return src;
    else
    {
        if(border&cv::BORDER_ISOLATED)
        {
            border = cv::BORDER_DEFAULT|cv::BORDER_ISOLATED;
            // Create small border to pass isSubmatrix check but to fail border extrapolation for functions which doesn't support ISOLATED flags.
            cv::Mat borderSrc;
            cv::copyMakeBorder(src, borderSrc, 1, 1, 1, 1, border, 0);
            return cv::Mat(borderSrc, cv::Rect(1, 1, src.cols, src.rows));
        }
        else
        {
            border = cv::BORDER_DEFAULT;
            cv::Mat borderSrc;
            cv::copyMakeBorder(src, borderSrc, borderSize, borderSize, borderSize, borderSize, border, 0);
            return cv::Mat(borderSrc, cv::Rect(borderSize, borderSize, src.cols, src.rows));
        }
    }
}

inline cv::Mat tsLoadImage(std::string name, int dstType = -1, cv::Size dstSize = cv::Size(), int color = -1)
{
    IPP_OFF(;)

    cv::Mat image;
    if(dstType >= 0)
    {
        int dstChannels = CV_MAT_CN(dstType);
        int dstDepth    = CV_MAT_DEPTH(dstType);
        if(color >= 0)
        {
            ADD_FAILURE() << "Unsupported conversion\n";
        }
        else
        {
            if(dstChannels == 1)
                image = cv::imread(name, cv::IMREAD_GRAYSCALE);
            else if(dstChannels == 3)
                image = cv::imread(name, cv::IMREAD_COLOR);
            else if(dstChannels == 4)
            {
                cv::Mat dstImage;
                image = cv::imread(name, cv::IMREAD_UNCHANGED);
                if(image.channels() == 1)
                    cvtColor(image, dstImage, cv::COLOR_GRAY2BGRA);
                else if(image.channels() == 3)
                    cvtColor(image, dstImage, cv::COLOR_BGR2BGRA);
                else
                    ADD_FAILURE() << "Improper test image format\n";
                image = dstImage;
            }
            else
                ADD_FAILURE() << "Improper test image format\n";
        }
        if(dstDepth != image.depth())
        {
            cv::Mat dstImage;
            double mul = 1;
            double add = 0;
            tsGetScaleVals(image.depth(), dstDepth, &mul, &add);
            image.convertTo(dstImage, dstDepth, mul, add);
            image = dstImage;
        }
    }
    else
        image = cv::imread(name, cv::IMREAD_UNCHANGED);
    if(dstSize.width && dstSize.height)
    {
        if(dstSize.width != image.cols || dstSize.height != image.rows)
        {
            cv::Mat dstImage;
            resize(image, dstImage, dstSize, 0, 0, cv::INTER_LINEAR);
            image = dstImage;
        }
    }
    return image;
}

#ifdef HAVE_IPP
class TsSetCpuFeaturesRegion
{
public:
    TsSetCpuFeaturesRegion()
    {
        m_stored = ippGetEnabledCpuFeatures();
    }
    TsSetCpuFeaturesRegion(Ipp64u featuresMask)
    {
        m_stored = ippGetEnabledCpuFeatures();
        Set(featuresMask);
    }
    IppStatus Set(Ipp64u featuresMask)
    {
        return ippSetCpuFeatures(featuresMask);
    }

    ~TsSetCpuFeaturesRegion()
    {
        ippSetCpuFeatures(m_stored);
    }

private:
    Ipp64u m_stored;
};
#endif

#if (IPP_TEST_MODE&IPP_TEST_MODE_CONFORMANCE_ONLY)
#define IPP_TEST_CYCLE_N(n) startTimer(); stopTimer(); return;
#define IPP_TEST_CYCLE()    startTimer(); stopTimer(); return;
#define IPP_TEST_CYCLE_MULTIRUN(runsNum) (void)runsNum; startTimer(); stopTimer(); return;
#else
#define IPP_TEST_CYCLE_N(n) IPP_ON(;) for(declare.iterations(n); next() && startTimer(); stopTimer())
#define IPP_TEST_CYCLE() IPP_ON(;) for(; next() && startTimer(); stopTimer())
#define IPP_TEST_CYCLE_MULTIRUN(runsNum) IPP_ON(;) for(declare.runs(runsNum); next() && startTimer(); stopTimer()) for(int r = 0; r < runsNum; ++r)
#endif

#if (IPP_TEST_MODE&IPP_TEST_MODE_LESS_SIZES)
#define IPP_TEST_SIZES        ::testing::Values(sz720p)
#define IPP_TEST_RESIZE_SIZES ::testing::Values(sz720p)
#define IPP_TEST_RECT_SIZES   ::testing::Values(cv::Size(1280, 1280))
#elif  (IPP_TEST_MODE&IPP_TEST_MODE_MORE_SIZES)
#define IPP_TEST_SIZES        ::testing::Values(cv::Size(1, 1), cv::Size(1, 64), cv::Size(64, 1), szODD, szVGA, sz1080p)
#define IPP_TEST_RESIZE_SIZES ::testing::Values(cv::Size(1, 1), cv::Size(1, 64), cv::Size(64, 1), szODD, szVGA, sz1080p)
#define IPP_TEST_RECT_SIZES   ::testing::Values(cv::Size(1, 1), cv::Size(640, 640), cv::Size(1920, 1920))
#else
#define IPP_TEST_SIZES        ::testing::Values(sz1080p)
#define IPP_TEST_RESIZE_SIZES ::testing::Values(sz1080p)
#define IPP_TEST_RECT_SIZES   ::testing::Values(cv::Size(1920, 1920))
#endif
#define IPP_TEST_KERNELS      ::testing::Values(3, 5, 7, 21)
#define IPP_TEST_STHREAD      ::testing::Values(ST)

#if (IPP_TEST_MODE&IPP_TEST_MODE_ALL_BORDERS)
#define DEFAULT_BORDER(BORDER), BORDER
CV_ENUM(IppBorderNoParam, BORDER_DEFAULT, BORDER_INMEM)
CV_ENUM(IppBorder, BORDER_CONSTANT, BORDER_REPLICATE, BORDER_REFLECT_101, BORDER_INMEM, BORDER_INMEM|BORDER_ISOLATED)
CV_ENUM(IppBorderTrans, BORDER_CONSTANT, BORDER_REPLICATE, BORDER_REFLECT_101, BORDER_TRANSPARENT)
#else
#define DEFAULT_BORDER(BORDER)
CV_ENUM(IppBorderNoParam, BORDER_DEFAULT)
CV_ENUM(IppBorder, BORDER_DEFAULT)
CV_ENUM(IppBorderTrans, BORDER_DEFAULT)
#endif

#endif

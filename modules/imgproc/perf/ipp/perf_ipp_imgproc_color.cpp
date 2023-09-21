#include "../perf_precomp.hpp"

#ifdef HAVE_IPP
#include "opencv2/ts/ipp_perf.hpp"

namespace opencv_test
{
using namespace perf;

enum
{
    CX_BGRA2HLS      = COLOR_COLORCVT_MAX + COLOR_BGR2HLS,
    CX_BGRA2HLS_FULL = COLOR_COLORCVT_MAX + COLOR_BGR2HLS_FULL,
    CX_BGRA2HSV      = COLOR_COLORCVT_MAX + COLOR_BGR2HSV,
    CX_BGRA2HSV_FULL = COLOR_COLORCVT_MAX + COLOR_BGR2HSV_FULL,
    CX_BGRA2Lab      = COLOR_COLORCVT_MAX + COLOR_BGR2Lab,
    CX_BGRA2Luv      = COLOR_COLORCVT_MAX + COLOR_BGR2Luv,
    CX_BGRA2XYZ      = COLOR_COLORCVT_MAX + COLOR_BGR2XYZ,
    CX_BGRA2YCrCb    = COLOR_COLORCVT_MAX + COLOR_BGR2YCrCb,
    CX_BGRA2YUV      = COLOR_COLORCVT_MAX + COLOR_BGR2YUV,
    CX_HLS2BGRA      = COLOR_COLORCVT_MAX + COLOR_HLS2BGR,
    CX_HLS2BGRA_FULL = COLOR_COLORCVT_MAX + COLOR_HLS2BGR_FULL,
    CX_HLS2RGBA      = COLOR_COLORCVT_MAX + COLOR_HLS2RGB,
    CX_HLS2RGBA_FULL = COLOR_COLORCVT_MAX + COLOR_HLS2RGB_FULL,
    CX_HSV2BGRA      = COLOR_COLORCVT_MAX + COLOR_HSV2BGR,
    CX_HSV2BGRA_FULL = COLOR_COLORCVT_MAX + COLOR_HSV2BGR_FULL,
    CX_HSV2RGBA      = COLOR_COLORCVT_MAX + COLOR_HSV2RGB,
    CX_HSV2RGBA_FULL = COLOR_COLORCVT_MAX + COLOR_HSV2RGB_FULL,
    CX_Lab2BGRA      = COLOR_COLORCVT_MAX + COLOR_Lab2BGR,
    CX_Lab2LBGRA     = COLOR_COLORCVT_MAX + COLOR_Lab2LBGR,
    CX_Lab2LRGBA     = COLOR_COLORCVT_MAX + COLOR_Lab2LRGB,
    CX_Lab2RGBA      = COLOR_COLORCVT_MAX + COLOR_Lab2RGB,
    CX_LBGRA2Lab     = COLOR_COLORCVT_MAX + COLOR_LBGR2Lab,
    CX_LBGRA2Luv     = COLOR_COLORCVT_MAX + COLOR_LBGR2Luv,
    CX_LRGBA2Lab     = COLOR_COLORCVT_MAX + COLOR_LRGB2Lab,
    CX_LRGBA2Luv     = COLOR_COLORCVT_MAX + COLOR_LRGB2Luv,
    CX_Luv2BGRA      = COLOR_COLORCVT_MAX + COLOR_Luv2BGR,
    CX_Luv2LBGRA     = COLOR_COLORCVT_MAX + COLOR_Luv2LBGR,
    CX_Luv2LRGBA     = COLOR_COLORCVT_MAX + COLOR_Luv2LRGB,
    CX_Luv2RGBA      = COLOR_COLORCVT_MAX + COLOR_Luv2RGB,
    CX_RGBA2HLS      = COLOR_COLORCVT_MAX + COLOR_RGB2HLS,
    CX_RGBA2HLS_FULL = COLOR_COLORCVT_MAX + COLOR_RGB2HLS_FULL,
    CX_RGBA2HSV      = COLOR_COLORCVT_MAX + COLOR_RGB2HSV,
    CX_RGBA2HSV_FULL = COLOR_COLORCVT_MAX + COLOR_RGB2HSV_FULL,
    CX_RGBA2Lab      = COLOR_COLORCVT_MAX + COLOR_RGB2Lab,
    CX_RGBA2Luv      = COLOR_COLORCVT_MAX + COLOR_RGB2Luv,
    CX_RGBA2XYZ      = COLOR_COLORCVT_MAX + COLOR_RGB2XYZ,
    CX_RGBA2YCrCb    = COLOR_COLORCVT_MAX + COLOR_RGB2YCrCb,
    CX_RGBA2YUV      = COLOR_COLORCVT_MAX + COLOR_RGB2YUV,
    CX_XYZ2BGRA      = COLOR_COLORCVT_MAX + COLOR_XYZ2BGR,
    CX_XYZ2RGBA      = COLOR_COLORCVT_MAX + COLOR_XYZ2RGB,
    CX_YCrCb2BGRA    = COLOR_COLORCVT_MAX + COLOR_YCrCb2BGR,
    CX_YCrCb2RGBA    = COLOR_COLORCVT_MAX + COLOR_YCrCb2RGB,
    CX_YUV2BGRA      = COLOR_COLORCVT_MAX + COLOR_YUV2BGR,
    CX_YUV2RGBA      = COLOR_COLORCVT_MAX + COLOR_YUV2RGB
};

struct ChPair
{
    ChPair(int _scn, int _dcn): scn(_scn), dcn(_dcn) {}
    int scn, dcn;
};

//Copied implementation from perf_cvt_color.cpp, as it's declared static there since openCV version 4
static ChPair getConversionInfo(int cvtMode)
{
    switch(cvtMode)
    {
    case COLOR_BayerBG2GRAY: case COLOR_BayerGB2GRAY:
    case COLOR_BayerGR2GRAY: case COLOR_BayerRG2GRAY:
    case COLOR_YUV2GRAY_420:
        return ChPair(1,1);
    case COLOR_GRAY2BGR555: case COLOR_GRAY2BGR565:
        return ChPair(1,2);
    case COLOR_BayerBG2BGR: case COLOR_BayerBG2BGR_VNG:
    case COLOR_BayerGB2BGR: case COLOR_BayerGB2BGR_VNG:
    case COLOR_BayerGR2BGR: case COLOR_BayerGR2BGR_VNG:
    case COLOR_BayerRG2BGR: case COLOR_BayerRG2BGR_VNG:
	case COLOR_BayerBG2BGR_EA: case COLOR_BayerGB2BGR_EA:
    case COLOR_BayerRG2BGR_EA: case COLOR_BayerGR2BGR_EA:
    case COLOR_GRAY2BGR:
    case COLOR_YUV2BGR_NV12: case COLOR_YUV2RGB_NV12:
    case COLOR_YUV2BGR_NV21: case COLOR_YUV2RGB_NV21:
    case COLOR_YUV2BGR_YV12: case COLOR_YUV2RGB_YV12:
    case COLOR_YUV2BGR_IYUV: case COLOR_YUV2RGB_IYUV:
        return ChPair(1,3);
    case COLOR_GRAY2BGRA:
    case COLOR_YUV2BGRA_NV12: case COLOR_YUV2RGBA_NV12:
    case COLOR_YUV2BGRA_NV21: case COLOR_YUV2RGBA_NV21:
    case COLOR_YUV2BGRA_YV12: case COLOR_YUV2RGBA_YV12:
    case COLOR_YUV2BGRA_IYUV: case COLOR_YUV2RGBA_IYUV:
    case COLOR_BayerBG2BGRA: case COLOR_BayerGB2BGRA:
    case COLOR_BayerGR2BGRA: case COLOR_BayerRG2BGRA:
        return ChPair(1,4);
    case COLOR_BGR5552GRAY: case COLOR_BGR5652GRAY:
        return ChPair(2,1);
    case COLOR_BGR5552BGR: case COLOR_BGR5552RGB:
    case COLOR_BGR5652BGR: case COLOR_BGR5652RGB:
    case COLOR_YUV2RGB_UYVY: case COLOR_YUV2BGR_UYVY:
    case COLOR_YUV2RGB_YUY2: case COLOR_YUV2BGR_YUY2:
    case COLOR_YUV2RGB_YVYU: case COLOR_YUV2BGR_YVYU:
        return ChPair(2,3);
    case COLOR_BGR5552BGRA: case COLOR_BGR5552RGBA:
    case COLOR_BGR5652BGRA: case COLOR_BGR5652RGBA:
    case COLOR_YUV2RGBA_UYVY: case COLOR_YUV2BGRA_UYVY:
    case COLOR_YUV2RGBA_YUY2: case COLOR_YUV2BGRA_YUY2:
    case COLOR_YUV2RGBA_YVYU: case COLOR_YUV2BGRA_YVYU:
        return ChPair(2,4);
    case COLOR_BGR2GRAY: case COLOR_RGB2GRAY:
    case COLOR_RGB2YUV_IYUV: case COLOR_RGB2YUV_YV12:
    case COLOR_BGR2YUV_IYUV: case COLOR_BGR2YUV_YV12:
        return ChPair(3,1);
    case COLOR_BGR2BGR555: case COLOR_BGR2BGR565:
    case COLOR_RGB2BGR555: case COLOR_RGB2BGR565:
        return ChPair(3,2);
    case COLOR_BGR2HLS: case COLOR_BGR2HLS_FULL:
    case COLOR_BGR2HSV: case COLOR_BGR2HSV_FULL:
    case COLOR_BGR2Lab: case COLOR_BGR2Luv:
    case COLOR_BGR2RGB: case COLOR_BGR2XYZ:
    case COLOR_BGR2YCrCb: case COLOR_BGR2YUV:
    case COLOR_HLS2BGR: case COLOR_HLS2BGR_FULL:
    case COLOR_HLS2RGB: case COLOR_HLS2RGB_FULL:
    case COLOR_HSV2BGR: case COLOR_HSV2BGR_FULL:
    case COLOR_HSV2RGB: case COLOR_HSV2RGB_FULL:
    case COLOR_Lab2BGR: case COLOR_Lab2LBGR:
    case COLOR_Lab2LRGB: case COLOR_Lab2RGB:
    case COLOR_LBGR2Lab: case COLOR_LBGR2Luv:
    case COLOR_LRGB2Lab: case COLOR_LRGB2Luv:
    case COLOR_Luv2BGR: case COLOR_Luv2LBGR:
    case COLOR_Luv2LRGB: case COLOR_Luv2RGB:
    case COLOR_RGB2HLS: case COLOR_RGB2HLS_FULL:
    case COLOR_RGB2HSV: case COLOR_RGB2HSV_FULL:
    case COLOR_RGB2Lab: case COLOR_RGB2Luv:
    case COLOR_RGB2XYZ: case COLOR_RGB2YCrCb:
    case COLOR_RGB2YUV: case COLOR_XYZ2BGR:
    case COLOR_XYZ2RGB: case COLOR_YCrCb2BGR:
    case COLOR_YCrCb2RGB: case COLOR_YUV2BGR:
    case COLOR_YUV2RGB:
        return ChPair(3,3);
    case COLOR_BGR2BGRA: case COLOR_BGR2RGBA:
    case CX_HLS2BGRA: case CX_HLS2BGRA_FULL:
    case CX_HLS2RGBA: case CX_HLS2RGBA_FULL:
    case CX_HSV2BGRA: case CX_HSV2BGRA_FULL:
    case CX_HSV2RGBA: case CX_HSV2RGBA_FULL:
    case CX_Lab2BGRA: case CX_Lab2LBGRA:
    case CX_Lab2LRGBA: case CX_Lab2RGBA:
    case CX_Luv2BGRA: case CX_Luv2LBGRA:
    case CX_Luv2LRGBA: case CX_Luv2RGBA:
    case CX_XYZ2BGRA: case CX_XYZ2RGBA:
    case CX_YCrCb2BGRA: case CX_YCrCb2RGBA:
    case CX_YUV2BGRA: case CX_YUV2RGBA:
        return ChPair(3,4);
    case COLOR_BGRA2GRAY: case COLOR_RGBA2GRAY:
    case COLOR_RGBA2YUV_IYUV: case COLOR_RGBA2YUV_YV12:
    case COLOR_BGRA2YUV_IYUV: case COLOR_BGRA2YUV_YV12:
        return ChPair(4,1);
    case COLOR_BGRA2BGR555: case COLOR_BGRA2BGR565:
    case COLOR_RGBA2BGR555: case COLOR_RGBA2BGR565:
        return ChPair(4,2);
    case COLOR_BGRA2BGR: case CX_BGRA2HLS:
    case CX_BGRA2HLS_FULL: case CX_BGRA2HSV:
    case CX_BGRA2HSV_FULL: case CX_BGRA2Lab:
    case CX_BGRA2Luv: case CX_BGRA2XYZ:
    case CX_BGRA2YCrCb: case CX_BGRA2YUV:
    case CX_LBGRA2Lab: case CX_LBGRA2Luv:
    case CX_LRGBA2Lab: case CX_LRGBA2Luv:
    case COLOR_RGBA2BGR: case CX_RGBA2HLS:
    case CX_RGBA2HLS_FULL: case CX_RGBA2HSV:
    case CX_RGBA2HSV_FULL: case CX_RGBA2Lab:
    case CX_RGBA2Luv: case CX_RGBA2XYZ:
    case CX_RGBA2YCrCb: case CX_RGBA2YUV:
        return ChPair(4,3);
    case COLOR_BGRA2RGBA:
	case COLOR_RGBA2mRGBA:
    case COLOR_mRGBA2RGBA:
        return ChPair(4,4);
    default:
        ADD_FAILURE() << "Unknown conversion type";
        break;
    };
    return ChPair(0,0);
}

static bool no16u(int mode)
{
    switch(mode)
    {
    case COLOR_RGBA2mRGBA:
    case COLOR_mRGBA2RGBA:

    case COLOR_YUV2BGR_NV12:
    case COLOR_YUV2BGRA_NV12:
    case COLOR_YUV2RGB_NV12:
    case COLOR_YUV2RGBA_NV12:
    case COLOR_YUV2BGR_NV21:
    case COLOR_YUV2BGRA_NV21:
    case COLOR_YUV2RGB_NV21:
    case COLOR_YUV2RGBA_NV21:
    case COLOR_YUV2BGR_YV12:
    case COLOR_YUV2BGRA_YV12:
    case COLOR_YUV2RGB_YV12:
    case COLOR_YUV2RGBA_YV12:
    case COLOR_YUV2BGR_IYUV:
    case COLOR_YUV2BGRA_IYUV:
    case COLOR_YUV2RGB_IYUV:
    case COLOR_YUV2RGBA_IYUV:
    case COLOR_YUV2GRAY_420:
    case COLOR_YUV2RGB_UYVY:
    case COLOR_YUV2BGR_UYVY:
    case COLOR_YUV2RGBA_UYVY:
    case COLOR_YUV2BGRA_UYVY:
    case COLOR_YUV2RGB_YUY2:
    case COLOR_YUV2BGR_YUY2:
    case COLOR_YUV2RGB_YVYU:
    case COLOR_YUV2BGR_YVYU:
    case COLOR_YUV2RGBA_YUY2:
    case COLOR_YUV2BGRA_YUY2:
    case COLOR_YUV2RGBA_YVYU:
    case COLOR_YUV2BGRA_YVYU:

    case COLOR_RGB2YUV_IYUV:
    case COLOR_BGR2YUV_IYUV:
    case COLOR_RGBA2YUV_IYUV:
    case COLOR_BGRA2YUV_IYUV:
    case COLOR_RGB2YUV_YV12:
    case COLOR_BGR2YUV_YV12:
    case COLOR_RGBA2YUV_YV12:
    case COLOR_BGRA2YUV_YV12:

    case COLOR_BGR2HLS_FULL:
    case COLOR_HLS2BGR_FULL:
    case COLOR_HLS2RGB_FULL:
    case COLOR_RGB2HLS_FULL:
    case COLOR_BGR2HSV_FULL:
    case COLOR_HSV2BGR_FULL:
    case COLOR_HSV2RGB_FULL:
    case COLOR_RGB2HSV_FULL:
    case COLOR_Lab2LBGR:
    case COLOR_Lab2LRGB:
    case COLOR_LBGR2Lab:
    case COLOR_LRGB2Lab:
    case COLOR_LBGR2Luv:
    case COLOR_LRGB2Luv:
    case COLOR_Luv2LBGR:
    case COLOR_Luv2LRGB:
    case COLOR_BayerBG2BGR_VNG:
    case COLOR_BayerGB2BGR_VNG:
    case COLOR_BayerGR2BGR_VNG:
    case COLOR_BayerRG2BGR_VNG:
             return true;
    default: return false;
    }
}

static bool no32f(int mode)
{
    switch(mode)
    {
    case COLOR_RGBA2mRGBA:
    case COLOR_mRGBA2RGBA:

    case COLOR_YUV2BGR_NV12:
    case COLOR_YUV2BGRA_NV12:
    case COLOR_YUV2RGB_NV12:
    case COLOR_YUV2RGBA_NV12:
    case COLOR_YUV2BGR_NV21:
    case COLOR_YUV2BGRA_NV21:
    case COLOR_YUV2RGB_NV21:
    case COLOR_YUV2RGBA_NV21:
    case COLOR_YUV2BGR_YV12:
    case COLOR_YUV2BGRA_YV12:
    case COLOR_YUV2RGB_YV12:
    case COLOR_YUV2RGBA_YV12:
    case COLOR_YUV2BGR_IYUV:
    case COLOR_YUV2BGRA_IYUV:
    case COLOR_YUV2RGB_IYUV:
    case COLOR_YUV2RGBA_IYUV:
    case COLOR_YUV2GRAY_420:
    case COLOR_YUV2RGB_UYVY:
    case COLOR_YUV2BGR_UYVY:
    case COLOR_YUV2RGBA_UYVY:
    case COLOR_YUV2BGRA_UYVY:
    case COLOR_YUV2RGB_YUY2:
    case COLOR_YUV2BGR_YUY2:
    case COLOR_YUV2RGB_YVYU:
    case COLOR_YUV2BGR_YVYU:
    case COLOR_YUV2RGBA_YUY2:
    case COLOR_YUV2BGRA_YUY2:
    case COLOR_YUV2RGBA_YVYU:
    case COLOR_YUV2BGRA_YVYU:

    case COLOR_RGB2YUV_IYUV:
    case COLOR_BGR2YUV_IYUV:
    case COLOR_RGBA2YUV_IYUV:
    case COLOR_BGRA2YUV_IYUV:
    case COLOR_RGB2YUV_YV12:
    case COLOR_BGR2YUV_YV12:
    case COLOR_RGBA2YUV_YV12:
    case COLOR_BGRA2YUV_YV12:

    case COLOR_BayerBG2BGR:
    case COLOR_BayerBG2BGR_VNG:
    case COLOR_BayerBG2GRAY:
    case COLOR_BayerGB2BGR:
    case COLOR_BayerGB2BGR_VNG:
    case COLOR_BayerGB2GRAY:
    case COLOR_BayerGR2BGR:
    case COLOR_BayerGR2BGR_VNG:
    case COLOR_BayerGR2GRAY:
    case COLOR_BayerRG2BGR:
    case COLOR_BayerRG2BGR_VNG:
    case COLOR_BayerRG2GRAY:
    case COLOR_BayerBG2BGR_EA:
    case COLOR_BayerGB2BGR_EA:
    case COLOR_BayerRG2BGR_EA:
    case COLOR_BayerGR2BGR_EA:
             return true;
    default: return false;
    }
}

static void getPlanarSizes(int mode, Size size, Size &srcSize, Size &dstSize)
{
    switch(mode)
    {
    case COLOR_YUV2BGR_NV12:
    case COLOR_YUV2BGRA_NV12:
    case COLOR_YUV2RGB_NV12:
    case COLOR_YUV2RGBA_NV12:
    case COLOR_YUV2BGR_NV21:
    case COLOR_YUV2BGRA_NV21:
    case COLOR_YUV2RGB_NV21:
    case COLOR_YUV2RGBA_NV21:
    case COLOR_YUV2BGR_YV12:
    case COLOR_YUV2BGRA_YV12:
    case COLOR_YUV2RGB_YV12:
    case COLOR_YUV2RGBA_YV12:
    case COLOR_YUV2BGR_IYUV:
    case COLOR_YUV2BGRA_IYUV:
    case COLOR_YUV2RGB_IYUV:
    case COLOR_YUV2RGBA_IYUV:
    case COLOR_YUV2GRAY_420:
    case COLOR_YUV2RGB_UYVY:
    case COLOR_YUV2BGR_UYVY:
    case COLOR_YUV2RGBA_UYVY:
    case COLOR_YUV2BGRA_UYVY:
    case COLOR_YUV2RGB_YUY2:
    case COLOR_YUV2BGR_YUY2:
    case COLOR_YUV2RGB_YVYU:
    case COLOR_YUV2BGR_YVYU:
    case COLOR_YUV2RGBA_YUY2:
    case COLOR_YUV2BGRA_YUY2:
    case COLOR_YUV2RGBA_YVYU:
    case COLOR_YUV2BGRA_YVYU:
        srcSize.width  = size.width;
        srcSize.height = size.height + size.height/2;
        dstSize = size;
        return;

    case COLOR_RGB2YUV_IYUV:
    case COLOR_BGR2YUV_IYUV:
    case COLOR_RGBA2YUV_IYUV:
    case COLOR_BGRA2YUV_IYUV:
    case COLOR_RGB2YUV_YV12:
    case COLOR_BGR2YUV_YV12:
    case COLOR_RGBA2YUV_YV12:
    case COLOR_BGRA2YUV_YV12:
        srcSize = size;
        dstSize.width  = size.width;
        dstSize.height = size.height + size.height/2;
        return;

    default:
        srcSize = size;
        dstSize = size;
        return;
    }
}

CV_ENUM(CvtMode,
    COLOR_BGR2BGRA, COLOR_BGR2GRAY,
    COLOR_BGR2HLS_FULL, COLOR_BGR2HSV_FULL,
    COLOR_BGR2RGB, COLOR_BGR2RGBA, COLOR_BGR2XYZ,
    COLOR_BGR2YCrCb, COLOR_BGR2YUV,

    COLOR_BGRA2BGR, COLOR_BGRA2GRAY, COLOR_BGRA2RGBA,
    CX_BGRA2HLS_FULL, CX_BGRA2HSV_FULL,
    CX_BGRA2XYZ,
    CX_BGRA2YCrCb, CX_BGRA2YUV,

    COLOR_GRAY2BGR, COLOR_GRAY2BGRA,

    COLOR_HLS2BGR_FULL, COLOR_HLS2RGB_FULL,
    CX_HLS2BGRA_FULL, CX_HLS2RGBA_FULL,

    COLOR_HSV2BGR_FULL, COLOR_HSV2RGB_FULL,
    CX_HSV2BGRA_FULL, CX_HSV2RGBA_FULL,

    COLOR_Lab2LBGR, COLOR_Lab2LRGB,
    CX_Lab2LBGRA, CX_Lab2LRGBA,

    COLOR_LBGR2Lab, COLOR_LBGR2Luv, COLOR_LRGB2Lab, COLOR_LRGB2Luv,
    CX_LBGRA2Lab, CX_LBGRA2Luv, CX_LRGBA2Lab, CX_LRGBA2Luv,

    COLOR_Luv2LBGR, COLOR_Luv2LRGB,
    CX_Luv2LBGRA, CX_Luv2LRGBA,

    COLOR_RGB2GRAY,
    COLOR_RGB2HLS_FULL, COLOR_RGB2HSV_FULL,
    COLOR_RGB2XYZ, COLOR_RGB2YCrCb, COLOR_RGB2YUV,

    COLOR_RGBA2BGR, COLOR_RGBA2GRAY,
    CX_RGBA2HLS_FULL, CX_RGBA2HSV_FULL,
    CX_RGBA2XYZ,
    CX_RGBA2YCrCb, CX_RGBA2YUV,

    COLOR_XYZ2BGR, COLOR_XYZ2RGB, CX_XYZ2BGRA, CX_XYZ2RGBA,

    COLOR_YCrCb2BGR, COLOR_YCrCb2RGB, CX_YCrCb2BGRA, CX_YCrCb2RGBA,
    COLOR_YUV2BGR, COLOR_YUV2RGB, CX_YUV2BGRA, CX_YUV2RGBA,

    COLOR_RGBA2mRGBA, COLOR_mRGBA2RGBA,

    // Demosaic
    /* No IPP yet
    COLOR_BayerBG2BGR, COLOR_BayerBG2BGR_VNG, COLOR_BayerBG2GRAY,
    COLOR_BayerGB2BGR, COLOR_BayerGB2BGR_VNG, COLOR_BayerGB2GRAY,
    COLOR_BayerGR2BGR, COLOR_BayerGR2BGR_VNG, COLOR_BayerGR2GRAY,
    COLOR_BayerRG2BGR, COLOR_BayerRG2BGR_VNG, COLOR_BayerRG2GRAY,

    COLOR_BayerBG2BGR_EA, COLOR_BayerGB2BGR_EA, COLOR_BayerRG2BGR_EA, COLOR_BayerGR2BGR_EA,
    */

    /* No IPP yet
    // Planar formats
    COLOR_YUV2BGR_NV12, COLOR_YUV2BGRA_NV12, COLOR_YUV2RGB_NV12, COLOR_YUV2RGBA_NV12, COLOR_YUV2BGR_NV21, COLOR_YUV2BGRA_NV21, COLOR_YUV2RGB_NV21, COLOR_YUV2RGBA_NV21,
    COLOR_YUV2BGR_YV12, COLOR_YUV2BGRA_YV12, COLOR_YUV2RGB_YV12, COLOR_YUV2RGBA_YV12, COLOR_YUV2BGR_IYUV, COLOR_YUV2BGRA_IYUV, COLOR_YUV2RGB_IYUV, COLOR_YUV2RGBA_IYUV,
    COLOR_YUV2GRAY_420, COLOR_YUV2RGB_UYVY, COLOR_YUV2BGR_UYVY, COLOR_YUV2RGBA_UYVY, COLOR_YUV2BGRA_UYVY, COLOR_YUV2RGB_YUY2, COLOR_YUV2BGR_YUY2, COLOR_YUV2RGB_YVYU,
    COLOR_YUV2BGR_YVYU, COLOR_YUV2RGBA_YUY2, COLOR_YUV2BGRA_YUY2, COLOR_YUV2RGBA_YVYU, COLOR_YUV2BGRA_YVYU,

    COLOR_RGB2YUV_IYUV, COLOR_BGR2YUV_IYUV, COLOR_RGBA2YUV_IYUV, COLOR_BGRA2YUV_IYUV,
    COLOR_RGB2YUV_YV12, COLOR_BGR2YUV_YV12, COLOR_RGBA2YUV_YV12, COLOR_BGRA2YUV_YV12
    */
    )

TEST(IppRegressionsFixture, rppCvtColor_Issues)
{
    // Init IPP extensions
    {
        TS_SET_THREADS(ST);
    }
    TsSetCpuFeaturesRegion feat(ippCPUID_SSE);

    {
        // COLOR_BGR2HSV_FULL
        {
            printf("COLOR_BGR2HSV_FULL - ");
            Size size = cv::Size(64, 64);
            Mat src(size, CV_8UC3);
            Mat dst(size, CV_8UC3);
            Mat dstRef(size, CV_8UC3);

            randu(src, 0, 256);

            // Conformance check
            RUN_IPP(cv::cvtColor(src, dst, COLOR_BGR2HSV_FULL));
            RUN_REF(cv::cvtColor(src, dstRef, COLOR_BGR2HSV_FULL));
            if(tsCompareImagesMaxDiff(dst, dstRef, 1, 1e-5))
                printf("passed\n");
        }

        // COLOR_BGR2YUV
        {
            printf("COLOR_BGR2YUV - ");
            Size size = cv::Size(24, 24);
            Mat src(size, CV_8UC3);
            Mat dst(size, CV_8UC3);
            Mat dstRef(size, CV_8UC3);

            randu(src, 0, 256);

            // Conformance check
            RUN_IPP(cv::cvtColor(src, dst, COLOR_BGR2YUV));
            RUN_REF(cv::cvtColor(src, dstRef, COLOR_BGR2YUV));
            if(tsCompareImagesMaxDiff(dst, dstRef, 1, 1e-5))
                printf("passed\n");
        }

        // COLOR_YUV2BGR
        {
            printf("COLOR_YUV2BGR - ");
            Size size = cv::Size(24, 24);
            Mat src(size, CV_8UC3);
            Mat dst(size, CV_8UC3);
            Mat dstRef(size, CV_8UC3);

            randu(src, 0, 256);

            // Conformance check
            RUN_IPP(cv::cvtColor(src, dst, COLOR_YUV2BGR));
            RUN_REF(cv::cvtColor(src, dstRef, COLOR_YUV2BGR));
            if(tsCompareImagesMaxDiff(dst, dstRef, 1, 1e-5))
                printf("passed\n");
        }

        // COLOR_Lab2LBGR
        {
            printf("COLOR_Lab2LBGR - ");
            Size size = cv::Size(24, 24);
            Mat src(size, CV_8UC3);
            Mat dst(size, CV_8UC3);
            Mat dstRef(size, CV_8UC3);

            randu(src, 0, 256);

            // Conformance check
            RUN_IPP(cv::cvtColor(src, dst, COLOR_Lab2LBGR));
            RUN_REF(cv::cvtColor(src, dstRef, COLOR_Lab2LBGR));
            if(tsCompareImagesMaxDiff(dst, dstRef, 1, 1e-5))
                printf("passed\n");
        }

        // COLOR_LBGR2Lab
        {
            printf("COLOR_LBGR2Lab - ");
            Size size = cv::Size(24, 24);
            Mat src(size, CV_8UC3);
            Mat dst(size, CV_8UC3);
            Mat dstRef(size, CV_8UC3);

            randu(src, 0, 256);

            // Conformance check
            RUN_IPP(cv::cvtColor(src, dst, COLOR_LBGR2Lab));
            RUN_REF(cv::cvtColor(src, dstRef, COLOR_LBGR2Lab));
            if(tsCompareImagesMaxDiff(dst, dstRef, 1, 1e-5))
                printf("passed\n");
        }

        // COLOR_BGR2XYZ
        {
            printf("COLOR_BGR2XYZ - ");
            Size size = cv::Size(24, 24);
            Mat src(size, CV_16UC3);
            Mat dst(size, CV_16UC3);
            Mat dstRef(size, CV_16UC3);

            randu(src, 0, 65536);

            // Conformance check
            RUN_IPP(cv::cvtColor(src, dst, COLOR_BGR2XYZ));
            RUN_REF(cv::cvtColor(src, dstRef, COLOR_BGR2XYZ));
            if(tsCompareImagesMaxDiff(dst, dstRef, 1, 1e-5))
                printf("passed\n");
        }

        // COLOR_XYZ2BGR
        {
            printf("COLOR_XYZ2BGR - ");
            Size size = cv::Size(24, 24);
            Mat src(size, CV_16UC3);
            Mat dst(size, CV_16UC3);
            Mat dstRef(size, CV_16UC3);

            randu(src, 0, 65536);

            // Conformance check
            RUN_IPP(cv::cvtColor(src, dst, COLOR_XYZ2BGR));
            RUN_REF(cv::cvtColor(src, dstRef, COLOR_XYZ2BGR));
            if(tsCompareImagesMaxDiff(dst, dstRef, 1, 1e-5))
                printf("passed\n");
        }

        // COLOR_Luv2LBGR
        {
            printf("COLOR_Luv2LBGR - ");
            Size size = cv::Size(24, 24);
            Mat src(size, CV_32FC3);
            Mat dst(size, CV_32FC3);
            Mat dstRef(size, CV_32FC3);

            randu(src, 0, 1);

            // Conformance check
            RUN_IPP(cv::cvtColor(src, dst, COLOR_Luv2LBGR));
            RUN_REF(cv::cvtColor(src, dstRef, COLOR_Luv2LBGR));
            if(tsCompareImagesMaxDiff(dst, dstRef, 1, 1e-5))
                printf("passed\n");
        }

        // COLOR_LBGR2Luv
        {
            printf("COLOR_LBGR2Luv - ");
            Size size = cv::Size(24, 24);
            Mat src(size, CV_32FC3);
            Mat dst(size, CV_32FC3);
            Mat dstRef(size, CV_32FC3);

            randu(src, 0, 1);

            // Conformance check
            RUN_IPP(cv::cvtColor(src, dst, COLOR_LBGR2Luv));
            RUN_REF(cv::cvtColor(src, dstRef, COLOR_LBGR2Luv));
            if(tsCompareImagesMaxDiff(dst, dstRef, 1, 1e-5))
                printf("passed\n");
        }
    }
}

typedef tuple<IppTestThreads, Size, MatType, CvtMode> CvtColorParamType;
typedef TestBaseWithParam<CvtColorParamType> CvtColorFixture;

PERF_TEST_P(CvtColorFixture, ippCvtColor,
            ::testing::Combine(
                IppTestThreads::all(),
                IPP_TEST_SIZES,
                ::testing::Values(CV_8U, CV_16U, CV_32F),
                CvtMode::all()
                )
            )
{
    const CvtColorParamType params = GetParam();
    const Size     size        = get<1>(params);
    const int      depth       = get<2>(params);
          int      mode        = get<3>(params);

    Size   srcSize;
    Size   dstSize;
    ChPair ch = getConversionInfo(mode);
    if(!ch.scn || !ch.dcn)
        return;
    mode %= COLOR_COLORCVT_MAX;

    if(depth == CV_16U && no16u(mode))
        throw ::perf::TestBase::PerfSkipTestException();
    if(depth == CV_32F && no32f(mode))
        throw ::perf::TestBase::PerfSkipTestException();

    getPlanarSizes(mode, size, srcSize, dstSize);

    // Init IPP extensions
    {
        TS_SET_THREADS(get<0>(params));
    }

    Mat src(srcSize, CV_MAKE_TYPE(depth, ch.scn));
    Mat dst(dstSize, CV_MAKE_TYPE(depth, ch.dcn));
    Mat dstRef(dstSize, CV_MAKE_TYPE(depth, ch.dcn));

    declare.in(src, WARMUP_RNG).out(dst);
    if(depth == CV_32F)
    {
        // Fit into range
        if(mode == COLOR_Luv2LBGR || mode == COLOR_Luv2LRGB)
        {
            Mat split[3];
            cv::split(src, split);
            randu(split[0], 0, 100);
            randu(split[0], -134, 220);
            randu(split[0], -140, 122);
            cv::merge(split, 3, src);
        }
        else if(mode == COLOR_Lab2LBGR || mode == COLOR_Lab2LRGB)
        {
            Mat split[3];
            cv::split(src, split);
            randu(split[0], 0, 100);
            randu(split[0], -128, 127);
            randu(split[0], -128, 127);
            cv::merge(split, 3, src);
        }
        else
            randu(src, 0, 1);
    }

    // Conformance check
    RUN_IPP(cv::cvtColor(src, dst, mode, ch.dcn));
    RUN_REF(cv::cvtColor(src, dstRef, mode, ch.dcn));
    VERIFY_DATA(dst, dstRef, 1, 1e-5);

    IPP_TEST_CYCLE() cv::cvtColor(src, dst, mode, ch.dcn);
}

} // namespace
#endif // HAVE_IPP

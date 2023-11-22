#pragma once

#include <opencv2/opencv.hpp>

#ifndef WITHOUT_CUDA
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudastereo.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/cudaarithm.hpp>
#include <libsgm.h>
#include <opencv2/cudafeatures2d.hpp>
#include <opencv2/cudaoptflow.hpp>
#else
namespace cv {
namespace cuda {
#ifndef HAVE_OPENCV_CUDAIMGPROC
// typedef cv::Mat GpuMat;
#endif
}; // namespace cuda
}; // namespace cv
#endif



typedef std::pair<Eigen::Matrix3d, Eigen::Vector3d> EigenPose;
typedef std::vector<cv::Mat>						CvImages;
typedef std::vector<cv::cuda::GpuMat>				CvCudaImages;
typedef std::shared_ptr<cv::Mat>					CVImagePtr;

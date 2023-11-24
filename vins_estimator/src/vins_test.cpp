#include <ros/ros.h>
#include <memory>
#include <opencv2/opencv.hpp>
#include "utility/queue_wrapper.hpp"
#include "utility/tic_toc.h"
#include <algorithm>

#include <vpi/OpenCVInterop.hpp>
#include <vpi/Image.h>
#include <vpi/Status.h>
#include <vpi/Stream.h>
#include <vpi/algo/ConvertImageFormat.h>
#include <vpi/algo/Rescale.h>
#include <vpi/algo/StereoDisparity.h>

#include <estimator/depth_estimator.hpp>

#include <vpi/OpenCVInterop.hpp>
#include <vpi/Array.h>
#include <vpi/Pyramid.h>
#include <vpi/Image.h>
#include <vpi/Status.h>
#include <vpi/Stream.h>
#include <vpi/algo/ConvertImageFormat.h>
#include <vpi/algo/Rescale.h>
#include <vpi/algo/GaussianPyramid.h>
#include <vpi/algo/HarrisCorners.h>
#include <vpi/algo/OpticalFlowPyrLK.h>

#include "featureTracker/feature_tracker_base.h"

#include <opencv2/cudastereo.hpp>

#include <libsgm.h>

#define CHECK_STATUS(STMT)                                                                                             \
	do {                                                                                                               \
		VPIStatus _status__ = (STMT);                                                                                  \
		if (_status__ != VPI_SUCCESS) {                                                                                \
			char buffer[VPI_MAX_STATUS_MESSAGE_LENGTH];                                                                \
			vpiGetLastStatusMessage(buffer, sizeof(buffer));                                                           \
			std::ostringstream ss;                                                                                     \
			ss << "at: " << #STMT << "\nfile: " << __FILE__ << ":" << __LINE__ << "  \n"                               \
			   << vpiStatusGetName(_status__) << ": " << buffer;                                                       \
			throw std::runtime_error(ss.str());                                                                        \
		}                                                                                                              \
	} while (0)

std::string vpi_image_format_to_str(uint64_t format) {
	std::map<uint64_t, std::string> format_map = {
		{VPI_IMAGE_FORMAT_INVALID, "VPI_IMAGE_FORMAT_INVALID"},
		{VPI_IMAGE_FORMAT_U8, "VPI_IMAGE_FORMAT_U8"},
		{VPI_IMAGE_FORMAT_U8_BL, "VPI_IMAGE_FORMAT_U8_BL"},
		{VPI_IMAGE_FORMAT_S8, "VPI_IMAGE_FORMAT_S8"},
		{VPI_IMAGE_FORMAT_U16, "VPI_IMAGE_FORMAT_U16"},
		{VPI_IMAGE_FORMAT_U32, "VPI_IMAGE_FORMAT_U32"},
		{VPI_IMAGE_FORMAT_S32, "VPI_IMAGE_FORMAT_S32"},
		{VPI_IMAGE_FORMAT_S16, "VPI_IMAGE_FORMAT_S16"},
		{VPI_IMAGE_FORMAT_S16_BL, "VPI_IMAGE_FORMAT_S16_BL"},
		{VPI_IMAGE_FORMAT_2S16, "VPI_IMAGE_FORMAT_2S16"},
		{VPI_IMAGE_FORMAT_2S16_BL, "VPI_IMAGE_FORMAT_2S16_BL"},
		{VPI_IMAGE_FORMAT_F32, "VPI_IMAGE_FORMAT_F32"},
		{VPI_IMAGE_FORMAT_F64, "VPI_IMAGE_FORMAT_F64"},
		{VPI_IMAGE_FORMAT_2F32, "VPI_IMAGE_FORMAT_2F32"},
		{VPI_IMAGE_FORMAT_Y8, "VPI_IMAGE_FORMAT_Y8"},
		{VPI_IMAGE_FORMAT_Y8_BL, "VPI_IMAGE_FORMAT_Y8_BL"},
		{VPI_IMAGE_FORMAT_Y8_ER, "VPI_IMAGE_FORMAT_Y8_ER"},
		{VPI_IMAGE_FORMAT_Y8_ER_BL, "VPI_IMAGE_FORMAT_Y8_ER_BL"},
		{VPI_IMAGE_FORMAT_Y16, "VPI_IMAGE_FORMAT_Y16"},
		{VPI_IMAGE_FORMAT_Y16_BL, "VPI_IMAGE_FORMAT_Y16_BL"},
		{VPI_IMAGE_FORMAT_Y16_ER, "VPI_IMAGE_FORMAT_Y16_ER"},
		{VPI_IMAGE_FORMAT_Y16_ER_BL, "VPI_IMAGE_FORMAT_Y16_ER_BL"},
		{VPI_IMAGE_FORMAT_NV12, "VPI_IMAGE_FORMAT_NV12"},
		{VPI_IMAGE_FORMAT_NV12_BL, "VPI_IMAGE_FORMAT_NV12_BL"},
		{VPI_IMAGE_FORMAT_NV12_ER, "VPI_IMAGE_FORMAT_NV12_ER"},
		{VPI_IMAGE_FORMAT_NV12_ER_BL, "VPI_IMAGE_FORMAT_NV12_ER_BL"},
		{VPI_IMAGE_FORMAT_NV24, "VPI_IMAGE_FORMAT_NV24"},
		{VPI_IMAGE_FORMAT_NV24_BL, "VPI_IMAGE_FORMAT_NV24_BL"},
		{VPI_IMAGE_FORMAT_NV24_ER, "VPI_IMAGE_FORMAT_NV24_ER"},
		{VPI_IMAGE_FORMAT_NV24_ER_BL, "VPI_IMAGE_FORMAT_NV24_ER_BL"},
		{VPI_IMAGE_FORMAT_UYVY, "VPI_IMAGE_FORMAT_UYVY"},
		{VPI_IMAGE_FORMAT_UYVY_BL, "VPI_IMAGE_FORMAT_UYVY_BL"},
		{VPI_IMAGE_FORMAT_UYVY_ER, "VPI_IMAGE_FORMAT_UYVY_ER"},
		{VPI_IMAGE_FORMAT_UYVY_ER_BL, "VPI_IMAGE_FORMAT_UYVY_ER_BL"},
		{VPI_IMAGE_FORMAT_YUYV, "VPI_IMAGE_FORMAT_YUYV"},
		{VPI_IMAGE_FORMAT_YUYV_BL, "VPI_IMAGE_FORMAT_YUYV_BL"},
		{VPI_IMAGE_FORMAT_YUYV_ER, "VPI_IMAGE_FORMAT_YUYV_ER"},
		{VPI_IMAGE_FORMAT_YUYV_ER_BL, "VPI_IMAGE_FORMAT_YUYV_ER_BL"},
		{VPI_IMAGE_FORMAT_RGB8, "VPI_IMAGE_FORMAT_RGB8"},
		{VPI_IMAGE_FORMAT_BGR8, "VPI_IMAGE_FORMAT_BGR8"},
		{VPI_IMAGE_FORMAT_RGBA8, "VPI_IMAGE_FORMAT_RGBA8"},
		{VPI_IMAGE_FORMAT_BGRA8, "VPI_IMAGE_FORMAT_BGRA8"},
	};
	if (format_map.find(format) == format_map.end()) {
		return "unknown";
	} else {
		return format_map[format];
	}
}

cv::Mat depth1(cv::Mat &img1, cv::Mat &img2, int disp_size, double confidence) {
	TicToc t_read;
	int	   width  = img1.cols;
	int	   height = img1.rows;

	ROS_INFO("time read:%3.1f ms", t_read.toc());


	VPIImage   vpi_left_img, vpi_right_img, vpi_depth_img;
	VPIPayload vpi_estimator_payload;
	VPIStream  stream;

	TicToc t_create;
	CHECK_STATUS(vpiStreamCreate(0, &stream));
	CHECK_STATUS(vpiImageCreateWrapperOpenCVMat(img1, 0, &vpi_left_img));
	CHECK_STATUS(vpiImageCreateWrapperOpenCVMat(img2, 0, &vpi_right_img));

	// VPIImageFormat format;
	// vpiImageGetFormat(vpi_left_img, &format);
	// ROS_INFO_STREAM("vpi_left_img formate:" << vpi_image_format_to_str(format));

	CHECK_STATUS(vpiImageCreate(width, height, VPI_IMAGE_FORMAT_S16, 0, &vpi_depth_img));
	VPIStereoDisparityEstimatorCreationParams stereo_params;
	CHECK_STATUS(vpiInitStereoDisparityEstimatorCreationParams(&stereo_params));
	stereo_params.maxDisparity = 128;
	CHECK_STATUS(vpiCreateStereoDisparityEstimator(VPI_BACKEND_CUDA, width, height, VPI_IMAGE_FORMAT_U8, &stereo_params,
												   &vpi_estimator_payload));
	VPIStereoDisparityEstimatorParams disparity_params;
	CHECK_STATUS(vpiInitStereoDisparityEstimatorParams(&disparity_params));
	disparity_params.confidenceThreshold = confidence * 65535; // 0~1

	ROS_INFO("time create:%3.1f ms", t_create.toc());

	TicToc t_submit;
	CHECK_STATUS(vpiSubmitStereoDisparityEstimator(stream, VPI_BACKEND_CUDA, vpi_estimator_payload, vpi_left_img,
												   vpi_right_img, vpi_depth_img, nullptr, &disparity_params));
	ROS_INFO("time submit:%3.1f ms", t_submit.toc());

	TicToc t_sync;
	CHECK_STATUS(vpiStreamSync(stream));
	ROS_INFO("time sync:%3.1f ms", t_sync.toc());

	TicToc		 t_copy;
	VPIImageData data;
	CHECK_STATUS(vpiImageLockData(vpi_depth_img, VPI_LOCK_READ, VPI_IMAGE_BUFFER_HOST_PITCH_LINEAR, &data));

	cv::Mat ref;
	CHECK_STATUS(vpiImageDataExportOpenCVMat(data, &ref));
	cv::Mat cv_depth_img = ref.clone();
	CHECK_STATUS(vpiImageUnlock(vpi_depth_img));
	cv_depth_img.convertTo(cv_depth_img, CV_8UC1, 255.0 / (32 * stereo_params.maxDisparity), 0);
	cv::Mat cv_depth_img_color;
	cv::applyColorMap(cv_depth_img, cv_depth_img_color, cv::COLORMAP_JET);
	ROS_INFO("time copy:%3.1f ms", t_copy.toc());
	// cv::imshow("depth", cv_depth_img);
	// cv::waitKey(0);
	// cv::imshow("color", cv_depth_img_color);
	// cv::waitKey(0);
	TicToc t_destory;
	vpiStreamDestroy(stream);
	vpiImageDestroy(vpi_left_img);
	vpiImageDestroy(vpi_right_img);
	vpiImageDestroy(vpi_depth_img);
	vpiPayloadDestroy(vpi_estimator_payload);
	ROS_INFO("time destory:%3.1f ms", t_destory.toc());

	ROS_INFO("\n\n\n");

	return cv_depth_img_color;
}
cv::Mat lib_sgm(cv::Mat &img1, cv::Mat &img2, int disp_size) {
	// int	  disp_size	  = 64;
	int	  P1		  = 10;
	int	  P2		  = 120;
	float uniqueness  = 0.95;
	int	  num_paths	  = 8;
	int	  min_disp	  = 0;
	int	  LR_max_diff = 1;
	auto  census_type = sgm::CensusType::SYMMETRIC_CENSUS_9x7;

	const int			src_depth = img1.type() == CV_8U ? 8 : 16;
	const int			dst_depth = 16;
	const sgm::PathType path_type = num_paths == 8 ? sgm::PathType::SCAN_8PATH : sgm::PathType::SCAN_4PATH;

	const sgm::StereoSGM::Parameters param(P1, P2, uniqueness, false, path_type, min_disp, LR_max_diff, census_type);
	sgm::StereoSGM ssgm(img1.cols, img1.rows, disp_size, src_depth, dst_depth, sgm::EXECUTE_INOUT_HOST2HOST, param);
	cv::Mat		   disparity(img1.size(), CV_16S);
	ssgm.execute(img1.data, img2.data, disparity.data);

	// create mask for invalid disp
	const cv::Mat mask = disparity == ssgm.get_invalid_disparity();

	cv::Mat disparity_8u, disparity_color;
	disparity.convertTo(disparity_8u, CV_8U, 255.0 / disp_size, 0);
	cv::applyColorMap(disparity_8u, disparity_color, cv::COLORMAP_TURBO);
	// disparity_8u.setTo(cv::Scalar(0), mask);
	// disparity_color.setTo(cv::Scalar(0, 0, 0), mask);

	return disparity_color;

	// cv::Mat show;
	// cv::cvtColor(disparity_8u, disparity_8u, cv::COLOR_GRAY2BGR);
	// cv::hconcat(disparity_8u, disparity_color, show);
	// cv::imshow("show", show);
	// cv::waitKey(0);
}

cv::Mat cv_gpu(cv::Mat &img1, cv::Mat &img2, int disp_size, int p1, int p2) {
	cv::cuda::GpuMat d_img1, d_img2; // 在GPU上的输入图像
	cv::cuda::GpuMat d_disp;		 // 在GPU上的视差图

	// 将输入图像上传到GPU
	d_img1.upload(img1);
	d_img2.upload(img2);

	cv::Mat						 disparity;
	int							 block_size	 = 3;
	int							 P1			 = 10;
	int							 P2			 = 120;
	float						 uniqueness	 = 0.95;
	int							 num_paths	 = 8;
	int							 min_disp	 = 0;
	int							 LR_max_diff = 1;
	cv::Ptr<cv::cuda::StereoSGM> sgbm		 = cv::cuda::createStereoSGM();
	sgbm->setDisp12MaxDiff(2);
	sgbm->setNumDisparities(disp_size);
	sgbm->setP1(20);
	sgbm->setP2(300);
	sgbm->compute(d_img1, d_img2, d_disp);
	d_disp.download(disparity);
	cv::Mat disparity_8u, disparity_color;
	disparity.convertTo(disparity_8u, CV_8U, 255.0 / disp_size, 0);
	cv::applyColorMap(disparity_8u, disparity_color, cv::COLORMAP_TURBO);
	return disparity_color;
}


void TEST_vpi_depth_main() {
	vpi::DepthEstimator depth_estimator;
	cv::Mat				cv_left_img, cv_right_img;
	cv_left_img	 = cv::imread("/swarm/fisheye_ws/output/img_left1.png", cv::IMREAD_GRAYSCALE);
	cv_right_img = cv::imread("/swarm/fisheye_ws/output/img_right1.png", cv::IMREAD_GRAYSCALE);
	depth_estimator.calculate_depth(cv_left_img, cv_right_img);
	depth_estimator.calculate_depth(cv_left_img, cv_right_img);
	depth_estimator.calculate_depth(cv_left_img, cv_right_img);
	TicToc t_est;
	depth_estimator.calculate_depth(cv_left_img, cv_right_img);
	ROS_INFO_STREAM("estimate time:" << t_est.toc());
	cv::imshow("depth", depth_estimator.depth_img);
	cv::waitKey(0);


	// ros::init(argc, argv, "vin_test");
	// ros::NodeHandle n("~");

	// cv::Mat img1;
	// int		start_num = 1;
	// int		perf_num  = 100;
	// TicToc	t_1;
	// for (int i = 0; i < start_num; i++) {
	// 	img1 = depth1(0.9);
	// }
	// double t1_sum = t_1.toc();
	// TicToc t_2;
	// for (int i = 0; i < perf_num; i++) {
	// 	img1 = depth1(0.9);
	// }
	// ROS_INFO_STREAM("first init time:" << t1_sum / start_num);
	// ROS_INFO_STREAM("depth1 time:" << t_2.toc() / perf_num);


	// // 64 - 15ms
	// 128 - 17ms
	// 256 -30 ms

	// cv::Mat img1 = depth1(0.5);
	// cv::Mat img2 = depth1(0.1);
	// cv::Mat show;
	// cv::hconcat(img1, img2, show);
	// cv::imshow("show", show);
	// cv::waitKey(0);
}
// 深度128  都在16ms左右
// 深度256 SGM 25ms  VPI 17ms


int main(int argc, char **argv) {
	TEST_vpi_depth_main();
	// cv::Mat cv_left_img, cv_right_img;
	// cv_left_img	 = cv::imread("/swarm/fisheye_ws/output/img_left1.png", cv::IMREAD_GRAYSCALE);
	// cv_right_img = cv::imread("/swarm/fisheye_ws/output/img_right1.png", cv::IMREAD_GRAYSCALE);
	// // cv_gpu(cv_left_img, cv_right_img, 128);
	// cv::Mat show = cv_gpu(cv_left_img, cv_right_img, 128, 10, 120);
	// cv::imshow("show", show);
	// cv::waitKey(0);
	// int init_num = 3;
	// int perf_num = 100;

	// TicToc t_sgm1;
	// for (int i = 0; i < init_num; i++) {
	// 	lib_sgm(cv_left_img, cv_right_img, 256);
	// }
	// double t_sgm1_sum = t_sgm1.toc();
	// TicToc t_sgm2;
	// for (int i = 0; i < perf_num; i++) {
	// 	lib_sgm(cv_left_img, cv_right_img, 256);
	// }
	// double t_sgm2_sum = t_sgm2.toc();

	// TicToc t_vpi1;
	// for (int i = 0; i < init_num; i++) {
	// 	depth1(cv_left_img, cv_right_img, 256, 0.1);
	// }
	// double t_vpi1_sum = t_vpi1.toc();
	// TicToc t_vpi2;
	// for (int i = 0; i < perf_num; i++) {
	// 	depth1(cv_left_img, cv_right_img, 256, 0.1);
	// }
	// double t_vpi2_sum = t_vpi2.toc();

	// ROS_INFO_STREAM("sgm init time:" << t_sgm1_sum / init_num);
	// ROS_INFO_STREAM("sgm time:" << t_sgm2_sum / perf_num);
	// ROS_INFO_STREAM("vpi init time:" << t_vpi1_sum / init_num);
	// ROS_INFO_STREAM("vpi time:" << t_vpi2_sum / perf_num);

	// cv::Mat disparity_color	 = lib_sgm(cv_left_img, cv_right_img, 256);
	// cv::Mat disparity_color2 = depth1(cv_left_img, cv_right_img, 256, 0.1);
	// cv::Mat show;
	// cv::hconcat(disparity_color, disparity_color2, show);
	// cv::imshow("disparity_color", show);
	// cv::waitKey(0);
}
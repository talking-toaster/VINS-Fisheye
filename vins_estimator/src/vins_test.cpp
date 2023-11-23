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

cv::Mat depth1(double confidence) {
	TicToc	t_read;
	cv::Mat cv_left_img, cv_right_img;
	cv_left_img	 = cv::imread("/swarm/fisheye_ws/output/img_left1.png", cv::IMREAD_GRAYSCALE);
	cv_right_img = cv::imread("/swarm/fisheye_ws/output/img_right1.png", cv::IMREAD_GRAYSCALE);
	int width	 = cv_left_img.cols;
	int height	 = cv_left_img.rows;

	ROS_INFO("time read:%3.1f ms", t_read.toc());


	VPIImage   vpi_left_img, vpi_right_img, vpi_depth_img;
	VPIPayload vpi_estimator_payload;
	VPIStream  stream;

	TicToc t_create;
	CHECK_STATUS(vpiStreamCreate(0, &stream));
	CHECK_STATUS(vpiImageCreateWrapperOpenCVMat(cv_left_img, 0, &vpi_left_img));
	CHECK_STATUS(vpiImageCreateWrapperOpenCVMat(cv_right_img, 0, &vpi_right_img));

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

void depth2() {

	cv::Mat cvImageLeft, cvImageRight;

	// VPI objects that will be used
	VPIImage   inLeft		 = NULL;
	VPIImage   inRight		 = NULL;
	VPIImage   tmpLeft		 = NULL;
	VPIImage   tmpRight		 = NULL;
	VPIImage   stereoLeft	 = NULL;
	VPIImage   stereoRight	 = NULL;
	VPIImage   disparity	 = NULL;
	VPIImage   confidenceMap = NULL;
	VPIStream  stream		 = NULL;
	VPIPayload stereo		 = NULL;

	int retval = 0;

	uint64_t backends = VPI_BACKEND_CUDA;

	cvImageLeft = cv::imread("/swarm/fisheye_ws/output/img_left1.png");
	if (cvImageLeft.empty())
		throw std::runtime_error("Can't open  image");


	cvImageRight = cv::imread("/swarm/fisheye_ws/output/img_right1.png");
	if (cvImageRight.empty())
		throw std::runtime_error("Can't open  image");



	// =================================
	// Allocate all VPI resources needed

	int32_t width  = cvImageLeft.cols;
	int32_t height = cvImageLeft.rows;

	// Create the stream that will be used for processing.
	CHECK_STATUS(vpiStreamCreate(0, &stream));

	// We now wrap the loaded images into a VPIImage object to be used by VPI.
	// VPI won't make a copy of it, so the original image must be in scope at all times.
	CHECK_STATUS(vpiImageCreateWrapperOpenCVMat(cvImageLeft, 0, &inLeft)); // inLeft 为 VPI_IMAGE_FORMAT_BGR8
	CHECK_STATUS(vpiImageCreateWrapperOpenCVMat(cvImageRight, 0, &inRight));

	VPIImageFormat format;
	vpiImageGetFormat(inLeft, &format);
	ROS_INFO_STREAM("in image formate:" << vpi_image_format_to_str(format));

	// Format conversion parameters needed for input pre-processing
	VPIConvertImageFormatParams convParams;
	CHECK_STATUS(vpiInitConvertImageFormatParams(&convParams));

	// Set algorithm parameters to be used. Only values what differs from defaults will be overwritten.
	VPIStereoDisparityEstimatorCreationParams stereoParams;
	CHECK_STATUS(vpiInitStereoDisparityEstimatorCreationParams(&stereoParams));

	// Default format and size for inputs and outputs
	VPIImageFormat stereoFormat	   = VPI_IMAGE_FORMAT_Y16_ER;
	VPIImageFormat disparityFormat = VPI_IMAGE_FORMAT_S16;

	int stereoWidth	 = width;
	int stereoHeight = height;
	int outputWidth	 = width;
	int outputHeight = height;

	// Create the payload for Stereo Disparity algorithm.
	// Payload is created before the image objects so that non-supported backends can be trapped with an error.
	CHECK_STATUS(
		vpiCreateStereoDisparityEstimator(backends, stereoWidth, stereoHeight, stereoFormat, &stereoParams, &stereo));

	// Create the image where the disparity map will be stored.
	CHECK_STATUS(vpiImageCreate(outputWidth, outputHeight, disparityFormat, 0, &disparity));

	// Create the input stereo images
	CHECK_STATUS(vpiImageCreate(stereoWidth, stereoHeight, stereoFormat, 0, &stereoLeft));
	CHECK_STATUS(vpiImageCreate(stereoWidth, stereoHeight, stereoFormat, 0, &stereoRight));

	// Create some temporary images, and the confidence image if the backend can support it

	// CHECK_STATUS(vpiImageCreate(width, height, VPI_IMAGE_FORMAT_U16, 0, &confidenceMap));


	// ================
	// Processing stage

	// -----------------
	// Pre-process input

	// Convert opencv input to grayscale format using CUDA
	CHECK_STATUS(vpiSubmitConvertImageFormat(stream, VPI_BACKEND_CUDA, inLeft, stereoLeft,
											 &convParams)); // stereoLeft VPI_IMAGE_FORMAT_Y16_ER
	CHECK_STATUS(vpiSubmitConvertImageFormat(stream, VPI_BACKEND_CUDA, inRight, stereoRight, &convParams));


	vpiImageGetFormat(stereoLeft, &format);
	ROS_INFO_STREAM("stereo image formate:" << vpi_image_format_to_str(format));

	// ------------------------------
	// Do stereo disparity estimation

	// Submit it with the input and output images
	CHECK_STATUS(
		vpiSubmitStereoDisparityEstimator(stream, backends, stereo, stereoLeft, stereoRight, disparity, NULL, NULL));

	// Wait until the algorithm finishes processing
	CHECK_STATUS(vpiStreamSync(stream));

	vpiImageGetFormat(disparity, &format);
	ROS_INFO_STREAM("disparity image formate:" << vpi_image_format_to_str(format));

	// ========================================
	// Output pre-processing and saving to disk
	// Lock output to retrieve its data on cpu memory
	VPIImageData data;
	CHECK_STATUS(vpiImageLockData(disparity, VPI_LOCK_READ, VPI_IMAGE_BUFFER_HOST_PITCH_LINEAR, &data));

	// Make an OpenCV matrix out of this image
	cv::Mat cvDisparity;
	CHECK_STATUS(vpiImageDataExportOpenCVMat(data, &cvDisparity));

	// Scale result and write it to disk. Disparities are in Q10.5 format,
	// so to map it to float, it gets divided by 32. Then the resulting disparity range,
	// from 0 to stereo.maxDisparity gets mapped to 0-255 for proper output.
	cvDisparity.convertTo(cvDisparity, CV_8UC1, 255.0 / (32 * stereoParams.maxDisparity), 0);

	// Apply JET colormap to turn the disparities into color, reddish hues
	// represent objects closer to the camera, blueish are farther away.
	cv::Mat cvDisparityColor;
	applyColorMap(cvDisparity, cvDisparityColor, cv::COLORMAP_JET);

	// Done handling output, don't forget to unlock it.
	CHECK_STATUS(vpiImageUnlock(disparity));

	cv::imshow("depth", cvDisparity);
	cv::waitKey(0);
	cv::imshow("color", cvDisparityColor);
	cv::waitKey(0);



	// ========
	// Clean up

	// Destroying stream first makes sure that all work submitted to
	// it is finished.
	vpiStreamDestroy(stream);

	// Only then we can destroy the other objects, as we're sure they
	// aren't being used anymore.

	vpiImageDestroy(inLeft);
	vpiImageDestroy(inRight);
	vpiImageDestroy(tmpLeft);
	vpiImageDestroy(tmpRight);
	vpiImageDestroy(stereoLeft);
	vpiImageDestroy(stereoRight);
	vpiImageDestroy(confidenceMap);
	vpiImageDestroy(disparity);
	vpiPayloadDestroy(stereo);
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

cv::Mat draw_keypoints(cv::Mat &img, std::vector<VPIKeypointF32> &keypoints, std::vector<uint32_t> score) {
	cv::Mat img_keypoints;
	cv::cvtColor(img, img_keypoints, cv::COLOR_GRAY2BGR);
	for (size_t i = 0; i < keypoints.size(); i++) {
		cv::circle(img_keypoints, cv::Point(keypoints[i].x, keypoints[i].y), 2, cv::Scalar(0, 0, 255), -1);
		cv::putText(img_keypoints, std::to_string(score[i]), cv::Point(keypoints[i].x, keypoints[i].y),
					cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 1);
	}
	return img_keypoints;
}

void vpi_harris() {
	cv::Mat cv_left_img, cv_right_img;
	cv_left_img	 = cv::imread("/swarm/fisheye_ws/output/img_left1.png", cv::IMREAD_GRAYSCALE);
	cv_right_img = cv::imread("/swarm/fisheye_ws/output/img_right1.png", cv::IMREAD_GRAYSCALE);
	int width	 = cv_left_img.cols;
	int height	 = cv_left_img.rows;

	VPIStream					  stream;
	VPIImage					  vpi_left_img;
	VPIArray					  pts, score;
	VPIPayload					  harris;
	VPIHarrisCornerDetectorParams harrisParams;
	CHECK_STATUS(vpiStreamCreate(0, &stream));
	CHECK_STATUS(vpiImageCreateWrapperOpenCVMat(cv_left_img, 0, &vpi_left_img));
	CHECK_STATUS(vpiArrayCreate(8192, VPI_ARRAY_TYPE_KEYPOINT_F32, 0, &pts));
	CHECK_STATUS(vpiArrayCreate(8192, VPI_ARRAY_TYPE_U32, 0, &score));
	CHECK_STATUS(vpiCreateHarrisCornerDetector(VPI_BACKEND_CUDA, width, height, &harris));
	CHECK_STATUS(vpiInitHarrisCornerDetectorParams(&harrisParams));
	harrisParams.gradientSize	= 3;
	harrisParams.blockSize		= 5;
	harrisParams.strengthThresh = 20;
	harrisParams.sensitivity	= 0.001;
	harrisParams.minNMSDistance = 30;
	TicToc t_harris;
	CHECK_STATUS(
		vpiSubmitHarrisCornerDetector(stream, VPI_BACKEND_CUDA, harris, vpi_left_img, pts, score, &harrisParams));
	CHECK_STATUS(vpiStreamSync(stream));

	VPIArrayData ptsDataWrapper, scoreDataWrapper;
	CHECK_STATUS(vpiArrayLockData(pts, VPI_LOCK_READ_WRITE, VPI_ARRAY_BUFFER_HOST_AOS, &ptsDataWrapper));
	CHECK_STATUS(vpiArrayLockData(score, VPI_LOCK_READ_WRITE, VPI_ARRAY_BUFFER_HOST_AOS, &scoreDataWrapper));


	VPIArrayBufferAOS &pts_data	  = ptsDataWrapper.buffer.aos;
	VPIArrayBufferAOS &score_data = scoreDataWrapper.buffer.aos;

	VPIKeypointF32 *kptData = reinterpret_cast<VPIKeypointF32 *>(pts_data.data);
	uint32_t	   *pscore	= reinterpret_cast<uint32_t *>(score_data.data);

	std::vector<int> indices(*pts_data.sizePointer);
	std::iota(indices.begin(), indices.end(), 0);

	stable_sort(indices.begin(), indices.end(), [&score_data](int a, int b) {
		uint32_t *score = reinterpret_cast<uint32_t *>(score_data.data);
		return score[a] >= score[b]; // decreasing score order
	});

	// keep the only 'max' indexes.
	indices.resize(std::min<size_t>(indices.size(), 100));



	// reorder the keypoints to keep the first 'max' with highest scores.
	std::vector<VPIKeypointF32> kpt;
	kpt.reserve(indices.size());
	for (auto idx : indices)
		kpt.push_back(kptData[idx]);
	// std::transform(indices.begin(), indices.end(), std::back_inserter(kpt),
	// 			   [kptData](int idx) { return kptData[idx]; });
	std::copy(kpt.begin(), kpt.end(), kptData);

	*pts_data.sizePointer = kpt.size();

	std::vector<uint32_t> score_vec;
	score_vec.reserve(indices.size());
	for (auto idx : indices)
		score_vec.push_back(pscore[idx]);
	std::copy(score_vec.begin(), score_vec.end(), pscore);

	*score_data.sizePointer = score_vec.size();

	CHECK_STATUS(vpiArrayUnlock(pts));
	CHECK_STATUS(vpiArrayUnlock(score));

	ROS_INFO_STREAM("harris time:" << t_harris.toc());


	vpiStreamDestroy(stream);
	vpiImageDestroy(vpi_left_img);
	vpiArrayDestroy(pts);
	vpiArrayDestroy(score);
	vpiPayloadDestroy(harris);
}

int main(int argc, char **argv) {

	cv::Mat cv_left_img, cv_right_img;
	cv_left_img	 = cv::imread("/swarm/fisheye_ws/output/img_left1.png", cv::IMREAD_GRAYSCALE);
	cv_right_img = cv::imread("/swarm/fisheye_ws/output/img_right1.png", cv::IMREAD_GRAYSCALE);
	cv::cuda::GpuMat				   img		= cv::cuda::GpuMat(cv_left_img);
	cv::Ptr<cv::cuda::CornersDetector> detector = cv::cuda::createGoodFeaturesToTrackDetector(img.type(), 100, 0.3, 10);
	cv::cuda::GpuMat				   d_prevPts;
	detector->detect(img, d_prevPts);


	cv::Mat show;
	cv::cvtColor(cv_left_img, show, cv::COLOR_GRAY2BGR);

	std::vector<cv::Point2f> prevPts = cv::Mat_<cv::Point2f>(cv::Mat(d_prevPts));
	ROS_INFO_STREAM("detect size:" << prevPts.size());
	for (int i = 0; i < d_prevPts.cols; i++) {
		cv::circle(show, prevPts[i], 2, cv::Scalar(0, 0, 255), -1);
	}
	cv::namedWindow("show", cv::WINDOW_NORMAL);
	cv::imshow("show", show);
	cv::waitKey(0);
}
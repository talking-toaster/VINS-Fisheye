#pragma once
#include <opencv2/opencv.hpp>

#include <vpi/OpenCVInterop.hpp>
#include <vpi/Image.h>
#include <vpi/Status.h>
#include <vpi/Stream.h>
#include <vpi/algo/ConvertImageFormat.h>
#include <vpi/algo/Rescale.h>
#include <vpi/algo/StereoDisparity.h>

#include "estimator/parameters.h"


#define VPI_CHECK_STATUS(STMT)                                                                                         \
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


namespace vpi {
class DepthEstimator {
  private:
	VPIImage   vpi_left_img, vpi_right_img, vpi_depth_img;
	VPIPayload vpi_estimator_payload;
	VPIStream  stream;

	VPIStereoDisparityEstimatorCreationParams depth_params;
	VPIStereoDisparityEstimatorParams		  algorithm_params;

	const int	 max_disparity		  = 128; // 最大视差
	const double confidence_threshold = 0.1; // 置信度

	VPIImageFormat input_img_format;

  public:
	cv::Mat depth_img;
	cv::Mat depth_confidence_img;

	DepthEstimator() {
	}

	~DepthEstimator() {
		ROS_INFO("DepthEstimator destructor called");
	}



	void calculate_depth(cv::Mat img_left, cv::Mat img_right) {
		input_img_format = VPI_IMAGE_FORMAT_U8;
		VPI_CHECK_STATUS(vpiStreamCreate(0, &stream));
		VPI_CHECK_STATUS(vpiInitStereoDisparityEstimatorCreationParams(&depth_params));
		depth_params.maxDisparity = this->max_disparity;
		VPI_CHECK_STATUS(vpiCreateStereoDisparityEstimator(VPI_BACKEND_CUDA, 848, 480, VPI_IMAGE_FORMAT_U8,
														   &depth_params, &vpi_estimator_payload));
		VPI_CHECK_STATUS(vpiInitStereoDisparityEstimatorParams(&algorithm_params));
		algorithm_params.confidenceThreshold = this->confidence_threshold * 65535;



		// ROS_INFO_STREAM("w:" << img_left.cols << "  h:" << img_left.rows);
		VPI_CHECK_STATUS(vpiImageCreateWrapperOpenCVMat(img_left, 0, &vpi_left_img));
		VPI_CHECK_STATUS(vpiImageCreateWrapperOpenCVMat(img_right, 0, &vpi_right_img));
		VPIImageFormat format;
		vpiImageGetFormat(vpi_left_img, &format);
		if (format != input_img_format)
			ROS_ERROR_STREAM("depth estimator input image format: " << vpi_image_format_to_str(format)
																	<< " , but set format is:"
																	<< vpi_image_format_to_str(input_img_format));
		VPI_CHECK_STATUS(vpiImageCreate(848, 480, VPI_IMAGE_FORMAT_S16, 0, &vpi_depth_img));
		VPI_CHECK_STATUS(vpiSubmitStereoDisparityEstimator(stream, VPI_BACKEND_CUDA, vpi_estimator_payload,
														   vpi_left_img, vpi_right_img, vpi_depth_img, nullptr,
														   &algorithm_params));
		VPI_CHECK_STATUS(vpiStreamSync(stream));
		VPIImageData data;
		VPI_CHECK_STATUS(vpiImageLockData(vpi_depth_img, VPI_LOCK_READ, VPI_IMAGE_BUFFER_HOST_PITCH_LINEAR, &data));
		cv::Mat depth_img_ref;
		VPI_CHECK_STATUS(vpiImageDataExportOpenCVMat(data, &depth_img_ref));
		depth_img = depth_img_ref.clone();
		VPI_CHECK_STATUS(vpiImageUnlock(vpi_depth_img));
		depth_img.convertTo(depth_img, CV_8UC1, 255.0 / (32 * depth_params.maxDisparity), 0);
		cv::Mat cv_depth_img_color;
		cv::applyColorMap(depth_img, cv_depth_img_color, cv::COLORMAP_JET);

		depth_img = cv_depth_img_color.clone();


		vpiStreamDestroy(stream);
		vpiImageDestroy(vpi_left_img);
		vpiImageDestroy(vpi_right_img);
		vpiImageDestroy(vpi_depth_img);
		vpiPayloadDestroy(vpi_estimator_payload);
	}
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
};



} // namespace vpi

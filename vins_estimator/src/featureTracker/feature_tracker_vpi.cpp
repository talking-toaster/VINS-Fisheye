#include "featureTracker/feature_tracker_vpi.hpp"
#include "estimator/depth_estimator.hpp"


namespace FeatureTracker {
FeatureTrackerVPI::FeatureTrackerVPI(Estimator *_estimator) : BaseFeatureTracker(_estimator) {
	VPI_CHECK_STATUS(vpiStreamCreate(0, &stream_left));
	VPI_CHECK_STATUS(vpiStreamCreate(0, &stream_right));
}
FeatureTrackerVPI::~FeatureTrackerVPI() {
	ROS_INFO("FeatureTrackerVPI destructor called");
	vpiStreamDestroy(stream_left);
	vpiStreamDestroy(stream_right);
	vpiImageDestroy(vpi_left_img);
	vpiImageDestroy(vpi_right_img);
}

void FeatureTrackerVPI::readIntrinsicParameter(const vector<string> &calib_file) {
	for (size_t i = 0; i < calib_file.size(); i++) {
		ROS_INFO("reading paramerter of camera %s", calib_file[i].c_str());
		camodocal::CameraPtr camera = CameraFactory::instance()->generateCameraFromYamlFile(calib_file[i]);
		m_camera.push_back(camera);
		height = camera->imageHeight();
		width  = camera->imageWidth();
		if (camera->modelType() == camodocal::Camera::PINHOLE) {
			FOCAL_LENGTH = ((camodocal::PinholeCamera *)camera.get())->getParameters().fx();
		} else {
			ROS_WARN("camera use default focal fx");
		}
	}
	if (calib_file.size() == 2)
		stereo_cam = 1;
}

void FeatureTrackerVPI::submit_detect_points() {

	VPI_CHECK_STATUS(vpiCreateHarrisCornerDetector(VPI_BACKEND_CPU, width, height, &harris));
	VPI_CHECK_STATUS(vpiInitHarrisCornerDetectorParams(&harrisParams));
	harrisParams.strengthThresh = 0;
	harrisParams.sensitivity	= 0.01;
	VPI_CHECK_STATUS(vpiSubmitHarrisCornerDetector(stream_left, VPI_BACKEND_CPU, harris, vpi_left_img, vpi_left_feature,
												   harris_scores, &harrisParams));
}
void FeatureTrackerVPI::addPoints(int pts_num) {
	VPIArrayData ptsDataWrapper, scoreDataWrapper;
	VPI_CHECK_STATUS(
		vpiArrayLockData(vpi_left_feature, VPI_LOCK_READ_WRITE, VPI_ARRAY_BUFFER_HOST_AOS, &ptsDataWrapper));
	VPI_CHECK_STATUS(
		vpiArrayLockData(harris_scores, VPI_LOCK_READ_WRITE, VPI_ARRAY_BUFFER_HOST_AOS, &scoreDataWrapper));


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
	indices.resize(std::min<size_t>(indices.size(), pts_num));

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

	VPI_CHECK_STATUS(vpiArrayUnlock(vpi_left_feature));
	VPI_CHECK_STATUS(vpiArrayUnlock(harris_scores));

	for (auto &p : kpt) {
		cur_pts.push_back({p.x, p.y});
		ids.push_back(generate_pt_id++);
		track_cnt.push_back(1);
	}
}

void FeatureTrackerVPI::reduce_points(bool is_left) {
	VPIArrayData pts_wrapper, status_wrapper;
	VPI_CHECK_STATUS(vpiArrayLockData(vpi_left_feature, VPI_LOCK_READ, VPI_ARRAY_BUFFER_HOST_AOS, &pts_wrapper));
	VPI_CHECK_STATUS(vpiArrayLockData(pts_status_left, VPI_LOCK_READ, VPI_ARRAY_BUFFER_HOST_AOS, &status_wrapper));

	VPIArrayBufferAOS &pts_data	   = pts_wrapper.buffer.aos;
	VPIArrayBufferAOS &status_data = status_wrapper.buffer.aos;

	VPIKeypointF32 *p_pts	 = reinterpret_cast<VPIKeypointF32 *>(pts_data.data);
	uint8_t		   *p_status = reinterpret_cast<uint8_t *>(status_data.data);
}


FeatureFrame FeatureTrackerVPI::trackImage(double _cur_time, cv::InputArray _img, cv::InputArray _img1) {
	cv::Mat cv_left_img	 = _img.getMat().clone();
	cv::Mat cv_right_img = _img1.getMat().clone();

	width  = cv_left_img.cols;
	height = cv_right_img.rows;

	if (frame_cnt++ == 0) {
		// 初始化参数
		VPI_CHECK_STATUS(vpiImageCreateWrapperOpenCVMat(cv_left_img, 0, &vpi_left_img));
		VPI_CHECK_STATUS(vpiImageCreateWrapperOpenCVMat(cv_right_img, 0, &vpi_right_img));
		VPI_CHECK_STATUS(vpiArrayCreate(MAX_HARRIS_CORNERS, VPI_ARRAY_TYPE_KEYPOINT_F32, 0, &vpi_left_feature));
		VPI_CHECK_STATUS(vpiArrayCreate(MAX_HARRIS_CORNERS, VPI_ARRAY_TYPE_KEYPOINT_F32, 0, &vpi_right_feature));
		VPI_CHECK_STATUS(vpiArrayCreate(MAX_HARRIS_CORNERS, VPI_ARRAY_TYPE_KEYPOINT_F32, 0, &vpi_prev_feature));
		VPI_CHECK_STATUS(vpiArrayCreate(MAX_HARRIS_CORNERS, VPI_ARRAY_TYPE_KEYPOINT_F32, 0, &vpi_fb_left_feature));
		VPI_CHECK_STATUS(vpiArrayCreate(MAX_HARRIS_CORNERS, VPI_ARRAY_TYPE_KEYPOINT_F32, 0, &vpi_fb_right_feature));

		VPI_CHECK_STATUS(vpiArrayCreate(MAX_HARRIS_CORNERS, VPI_ARRAY_TYPE_U32, 0, &harris_scores));
		VPI_CHECK_STATUS(
			vpiPyramidCreate(width, height, VPI_IMAGE_FORMAT_U8, pyramid_level, 0.5, 0, &vpi_left_pyramid));
		VPI_CHECK_STATUS(
			vpiPyramidCreate(width, height, VPI_IMAGE_FORMAT_U8, pyramid_level, 0.5, 0, &vpi_right_pyramid));
		VPI_CHECK_STATUS(
			vpiPyramidCreate(width, height, VPI_IMAGE_FORMAT_U8, pyramid_level, 0.5, 0, &vpi_prev_pyramid));

		VPI_CHECK_STATUS(vpiArrayCreate(MAX_HARRIS_CORNERS, VPI_ARRAY_TYPE_U8, 0, &pts_status_left));
		VPI_CHECK_STATUS(vpiArrayCreate(MAX_HARRIS_CORNERS, VPI_ARRAY_TYPE_U8, 0, &pts_status_right));
		VPI_CHECK_STATUS(vpiArrayCreate(MAX_HARRIS_CORNERS, VPI_ARRAY_TYPE_U8, 0, &pts_fb_status_left));
		VPI_CHECK_STATUS(vpiArrayCreate(MAX_HARRIS_CORNERS, VPI_ARRAY_TYPE_U8, 0, &pts_fb_status_right));
		// harris 检测

		submit_detect_points();
		// 图像金字塔
		VPI_CHECK_STATUS(
			vpiSubmitGaussianPyramidGenerator(stream_left, backend, vpi_left_img, vpi_left_pyramid, VPI_BORDER_CLAMP));
		// sync
		VPI_CHECK_STATUS(vpiStreamSync(stream_right));
		VPI_CHECK_STATUS(vpiStreamSync(stream_left));

		addPoints(100);

	} else {
		VPI_CHECK_STATUS(vpiImageSetWrappedOpenCVMat(vpi_left_img, cv_left_img));
		VPI_CHECK_STATUS(vpiImageSetWrappedOpenCVMat(vpi_right_img, cv_right_img));
		// 生成图像金字塔
		VPI_CHECK_STATUS(
			vpiSubmitGaussianPyramidGenerator(stream_left, backend, vpi_left_img, vpi_left_pyramid, VPI_BORDER_CLAMP));
		VPI_CHECK_STATUS(vpiSubmitGaussianPyramidGenerator(stream_right, backend, vpi_right_img, vpi_right_pyramid,
														   VPI_BORDER_CLAMP));
		VPI_CHECK_STATUS(vpiStreamSync(stream_right));
		VPI_CHECK_STATUS(vpiStreamSync(stream_left));
		// LK光流
		VPI_CHECK_STATUS(
			vpiCreateOpticalFlowPyrLK(backend, width, height, VPI_IMAGE_FORMAT_U8, pyramid_level, 0.5, &optflow));
		VPIOpticalFlowPyrLKParams lkParams;
		VPI_CHECK_STATUS(vpiInitOpticalFlowPyrLKParams(&lkParams));

		VPI_CHECK_STATUS(vpiSubmitOpticalFlowPyrLK(stream_left, backend, optflow, vpi_prev_pyramid, vpi_left_pyramid,
												   vpi_prev_feature, vpi_left_feature, pts_status_left,
												   &lkParams)); // prev -> left

		// VPI_CHECK_STATUS(vpiStreamSync(stream_left));
		//  flow reduce pts

		VPIArrayData prev_wrapper, left_wrapper, status_wrapper;
		VPI_CHECK_STATUS(
			vpiArrayLockData(vpi_prev_feature, VPI_LOCK_READ_WRITE, VPI_ARRAY_BUFFER_HOST_AOS, &prev_wrapper));
		VPI_CHECK_STATUS(
			vpiArrayLockData(vpi_left_feature, VPI_LOCK_READ_WRITE, VPI_ARRAY_BUFFER_HOST_AOS, &left_wrapper));
		VPI_CHECK_STATUS(
			vpiArrayLockData(pts_status_left, VPI_LOCK_READ_WRITE, VPI_ARRAY_BUFFER_HOST_AOS, &status_wrapper));


		VPIArrayBufferAOS &prev_data   = prev_wrapper.buffer.aos;
		VPIArrayBufferAOS &left_data   = left_wrapper.buffer.aos;
		VPIArrayBufferAOS &status_data = status_wrapper.buffer.aos;

		VPIKeypointF32 *p_prev	 = reinterpret_cast<VPIKeypointF32 *>(prev_data.data);
		VPIKeypointF32 *p_left	 = reinterpret_cast<VPIKeypointF32 *>(left_data.data);
		uint32_t	   *p_status = reinterpret_cast<uint32_t *>(status_data.data);
	}
	VPI_CHECK_STATUS(vpiStreamSync(stream_left));

	vpi_prev_feature = vpi_left_feature;
	vpi_prev_pyramid = vpi_left_pyramid;
	clean_vpi_object();
	return FeatureFrame();
}

void FeatureTrackerVPI::setPrediction(const map<int, Eigen::Vector3d> &predictPts_cam0,
									  const map<int, Eigen::Vector3d> &predictPt_cam1) {
}

void FeatureTrackerVPI::clean_vpi_object() {
}



} // namespace FeatureTracker
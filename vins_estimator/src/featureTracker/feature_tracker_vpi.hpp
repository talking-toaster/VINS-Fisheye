#pragma once

#include "featureTracker/feature_tracker_base.h"

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

namespace FeatureTracker {


class FeatureTrackerVPI : public BaseFeatureTracker {
  public:
	virtual FeatureFrame trackImage(double _cur_time, cv::InputArray _img,
									cv::InputArray _img1 = cv::noArray()) override;

	virtual void readIntrinsicParameter(const vector<string> &calib_file) override;
	FeatureTrackerVPI(Estimator *_estimator);
	~FeatureTrackerVPI();

  protected:
	void				setMask();
	void				addPoints(int pt_num);
	void				reduce_points(bool is_left);
	void				submit_detect_points();
	void				clean_vpi_object();
	vector<cv::Point3f> ptsVelocity(vector<int> &ids, vector<cv::Point3f> &pts, map<int, cv::Point3f> &cur_id_pts,
									map<int, cv::Point3f> &prev_id_pts);
	vector<cv::Point3f> undistortedPts(vector<cv::Point2f> &pts, camodocal::CameraPtr cam);

	void drawTrack(const cv::Mat &imLeft, const cv::Mat &imRight, vector<int> &curLeftIds,
				   vector<cv::Point2f> &curLeftPts, vector<cv::Point2f> &curRightPts,
				   map<int, cv::Point2f> &prevLeftPtsMap);

	virtual void setPrediction(const map<int, Eigen::Vector3d> &predictPts_cam0,
							   const map<int, Eigen::Vector3d> &predictPt_cam1 = map<int, Eigen::Vector3d>()) override;

  private:
	VPIStream  stream_left, stream_right;
	VPIImage   vpi_left_img, vpi_right_img;
	VPIPyramid vpi_left_pyramid, vpi_right_pyramid, vpi_prev_pyramid;
	VPIArray   vpi_left_feature, vpi_right_feature, vpi_prev_feature;
	VPIArray   vpi_fb_left_feature, vpi_fb_right_feature;


	VPIArray pts_status_left, pts_status_right; // 0:tracked
	VPIArray pts_fb_status_left, pts_fb_status_right;

	VPIPayload					  harris;
	VPIHarrisCornerDetectorParams harrisParams;
	VPIArray					  harris_scores;

	VPIPayload optflow;

	const VPIBackend backend = VPI_BACKEND_CUDA;

	int		  frame_cnt			 = 0;
	const int pyramid_level		 = 5;
	const int MAX_HARRIS_CORNERS = 8192;


	vector<cv::Point2f>	  n_pts;
	vector<cv::Point2f>	  predict_pts;
	vector<cv::Point2f>	  prev_pts, cur_pts, cur_right_pts;
	vector<cv::Point3f>	  prev_un_pts, cur_un_pts, cur_un_right_pts;
	vector<cv::Point3f>	  pts_velocity, right_pts_velocity;
	vector<int>			  ids, ids_right;
	vector<int>			  pts_img_id, pts_img_id_right;
	vector<int>			  track_cnt, track_right_cnt;
	map<int, cv::Point3f> cur_un_pts_map, prev_un_pts_map;
	map<int, cv::Point3f> cur_un_right_pts_map, prev_un_right_pts_map;
	map<int, cv::Point2f> prevLeftPtsMap;

	int generate_pt_id = 0;
};



} // namespace FeatureTracker
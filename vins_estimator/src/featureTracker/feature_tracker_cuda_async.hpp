#pragma once

#include "featureTracker/feature_tracker_base.h"
#include "featureTracker/feature_tracker_pinhole.hpp"


namespace FeatureTracker {


class DetectPointsAsync {
  private:
	cv::cuda::Stream				   stream;
	cv::Ptr<cv::cuda::CornersDetector> detector;
	cv::cuda::GpuMat				   new_pts_gpu;
	bool							   need_detect_points;

  public:
	void submit(const cv::cuda::GpuMat &img, int require_pts);
	void end(const vector<cv::Point2f> &cur_pts, vector<cv::Point2f> &n_pts);

  public:
};


class PinholeFeatureTrackerAsync : public PinholeFeatureTracker<cv::cuda::GpuMat> {
  protected:
	cv::cuda::GpuMat prev_gpu_img;
	cv::Mat			 cur_img, rightImg;

  public:
	PinholeFeatureTrackerAsync(Estimator *_estimator) : PinholeFeatureTracker<cv::cuda::GpuMat>(_estimator) {
	}
	virtual FeatureFrame trackImage(double _cur_time, cv::InputArray _img,
									cv::InputArray _img1 = cv::noArray()) override;
};
} // namespace FeatureTracker
#include "feature_tracker_cuda_async.hpp"



namespace FeatureTracker {
void DetectPointsAsync::submit(const cv::cuda::GpuMat &img, int require_pts) {
	need_detect_points = require_pts > MAX_CNT / 4;
	if (need_detect_points) {
		detector = cv::cuda::createGoodFeaturesToTrackDetector(CV_8UC1, require_pts, 0.01, MIN_DIST);
		detector->detect(img, new_pts_gpu, cv::noArray(), stream);
	}
}

void DetectPointsAsync::end(const vector<cv::Point2f> &cur_pts, vector<cv::Point2f> &n_pts) {
	if (need_detect_points) {
		stream.waitForCompletion();
		std::vector<cv::Point2f> n_pts_tmp;
		if (!new_pts_gpu.empty()) {
			n_pts_tmp = cv::Mat_<cv::Point2f>(cv::Mat(new_pts_gpu));
		} else {
			n_pts_tmp.clear();
		}
		n_pts.clear();
		std::vector<cv::Point2f> all_pts = cur_pts;
		for (auto &pt : n_pts_tmp) {
			bool has_nearby = false;
			for (auto &pt_j : all_pts) {
				if (cv::norm(pt - pt_j) < MIN_DIST) {
					has_nearby = true;
					break;
				}
			}
			if (!has_nearby) {
				n_pts.push_back(pt);
				all_pts.push_back(pt);
			}
		}
	} else {
		n_pts.clear();
	}
}



FeatureFrame PinholeFeatureTrackerAsync::trackImage(double _cur_time, cv::InputArray _img, cv::InputArray _img1) {
#ifndef WITHOUT_CUDA

	static double detected_time_submit_sum = 0;
	static double detected_time_end_sum	   = 0;
	static double flow_track_time_sum	   = 0;
	static double copy_to_gpu_sum		   = 0;
	static double show_track_sum		   = 0;
	static double front_end_sum			   = 0;
	static int	  count					   = 0;
	count++;
	TicToc t_trackImage;
	TicToc t_copy_to_gpu;

	cur_time = _cur_time;
	cv::Mat			 rightImg;
	cv::cuda::GpuMat cur_gpu_img   = cv::cuda::GpuMat(_img);
	cv::cuda::GpuMat right_gpu_img = cv::cuda::GpuMat(_img1);
	height						   = cur_gpu_img.rows;
	width						   = cur_gpu_img.cols;

	copy_to_gpu_sum += t_copy_to_gpu.toc();
	TicToc			  t_detect_submit;
	DetectPointsAsync detectPointsAsync;
	detectPointsAsync.submit(cur_gpu_img, MAX_CNT - int(cur_pts.size()));
	detected_time_submit_sum += t_detect_submit.toc();

	{
		cur_pts.clear();
		TicToc t_ft;
		cur_pts = opticalflow_track(cur_gpu_img, prev_pyr, prev_pts, ids, track_cnt, removed_pts, false);
		flow_track_time_sum += t_ft.toc();
	}

	TicToc t_detect_end;
	detectPointsAsync.end(cur_pts, n_pts);
	addPoints();
	detected_time_end_sum += t_detect_end.toc();

	if (!_img1.empty() && stereo_cam) {
		TicToc t_ft2;
		ids_right									 = ids;
		std::vector<cv::Point2f> right_side_init_pts = cur_pts;
		cur_right_pts	   = opticalflow_track(right_gpu_img, prev_pyr, right_side_init_pts, ids_right, track_right_cnt,
											   removed_pts, true);
		cur_un_right_pts   = undistortedPts(cur_right_pts, m_camera[1]);
		right_pts_velocity = ptsVelocity(ids_right, cur_un_right_pts, cur_un_right_pts_map, prev_un_right_pts_map);
		flow_track_time_sum += t_ft2.toc();
	}



	cur_un_pts	 = undistortedPts(cur_pts, m_camera[0]);
	pts_velocity = ptsVelocity(ids, cur_un_pts, cur_un_pts_map, prev_un_pts_map);


	if (SHOW_TRACK) {
		TicToc t_show;
		// cur_gpu_img.download(cur_img);
		// right_gpu_img.download(rightImg);
		cur_img	 = _img.getMat();
		rightImg = _img1.getMat();
		drawTrack(cur_img, rightImg, ids, cur_pts, cur_right_pts, prevLeftPtsMap);
		show_track_sum += t_show.toc();
	}

	prev_gpu_img		  = cur_gpu_img;
	prev_pts			  = cur_pts;
	prev_un_pts			  = cur_un_pts;
	prev_un_pts_map		  = cur_un_pts_map;
	prev_un_right_pts_map = cur_un_right_pts_map;

	prev_time	  = cur_time;
	hasPrediction = false;

	prevLeftPtsMap.clear();
	for (size_t i = 0; i < cur_pts.size(); i++)
		prevLeftPtsMap[ids[i]] = cur_pts[i];

	FeatureFrame featureFrame;
	BaseFeatureTracker::setup_feature_frame(featureFrame, ids, cur_pts, cur_un_pts, pts_velocity, 0);
	BaseFeatureTracker::setup_feature_frame(featureFrame, ids_right, cur_right_pts, cur_un_right_pts,
											right_pts_velocity, 1);
	double t_front_end = t_trackImage.toc();
	front_end_sum += t_front_end;
	ROS_INFO("[frontend] Img: %d: trackImage ALL: %3.1fms; PT NUM: %ld, STEREO: %ld; Avg: %3.1fms COPY_TO_GPU: %3.1fms "
			 "GFTT submit %3.1fms end %3.1fms LKFlow %3.1fms SHOW %3.1fms in GPU\n",
			 count, t_front_end, cur_pts.size(), cur_right_pts.size(), front_end_sum / count, copy_to_gpu_sum / count,
			 detected_time_submit_sum / count, detected_time_end_sum / count, flow_track_time_sum / count,
			 show_track_sum / count);
	return featureFrame;
#endif
} // namespace FeatureTracker
} // namespace FeatureTracker

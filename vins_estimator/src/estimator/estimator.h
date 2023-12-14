/*******************************************************
 * Copyright (C) 2019, Aerial Robotics Group, Hong Kong University of Science and Technology
 *
 * This file is part of VINS.
 *
 * Licensed under the GNU General Public License v3.0;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#pragma once

#include <thread>
#include <mutex>
#include <std_msgs/Header.h>
#include <std_msgs/Float32.h>
#include <ceres/ceres.h>
#include <unordered_map>
#include <queue>
#include <opencv2/core/eigen.hpp>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Geometry>

#include "parameters.h"
#include "feature_manager.h"
#include "utility/utility.h"
#include "utility/tic_toc.h"
#include "initial/solve_5pts.h"
#include "initial/initial_sfm.h"
#include "initial/initial_alignment.h"
#include "initial/initial_ex_rotation.h"
#include "factor/imu_factor.h"
#include "factor/pose_local_parameterization.h"
#include "factor/marginalization_factor.h"
#include "factor/projectionTwoFrameOneCamFactor.h"
#include "factor/projectionTwoFrameTwoCamFactor.h"
#include "factor/projectionOneFrameTwoCamFactor.h"
#include "featureTracker/feature_tracker_base.h"
#include "utility/opencv_cuda.h"
#include "utility/queue_wrapper.hpp"


class Estimator {
  public:
	Estimator();

	void setParameter();

	// interface
	void initFirstPose(Eigen::Vector3d p, Eigen::Matrix3d r);
	void inputIMU(double t, const Vector3d &linearAcceleration, const Vector3d &angularVelocity);
	void inputFeature(double t, const FeatureFrame &featureFrame);
	void inputImage(double t, const cv::Mat &_img, const cv::Mat &_img1 = cv::Mat());
	void inputMag(double t, const Vector3d &mag);

	bool is_next_odometry_frame();
	void processIMU(double t, double dt, const Vector3d &linear_acceleration, const Vector3d &angular_velocity);
	void processImage(const FeatureFrame &image, const double header);
	void processMeasurements();
	void processMag(const Vector3d &mag);

	void processDepthGeneration();

	// internal
	void   clearState();
	void   slideWindow();
	void   slideWindowNew();
	void   slideWindowOld();
	void   optimization();
	void   vector2double();
	void   double2vector();
	bool   failureDetection();
	bool   getIMUInterval(double t0, double t1, vector<pair<double, Eigen::Vector3d>> &accVector,
						  vector<pair<double, Eigen::Vector3d>> &gyrVector);
	bool   getMag(double t, Eigen::Vector3d &mag);
	void   getPoseInWorldFrame(Eigen::Matrix4d &T);
	void   getPoseInWorldFrame(int index, Eigen::Matrix4d &T);
	void   predictPtsInNextFrame();
	void   outliersRejection(set<int> &removeIndex);
	double reprojectionError(Matrix3d &Ri, Vector3d &Pi, Matrix3d &rici, Vector3d &tici, Matrix3d &Rj, Vector3d &Pj,
							 Matrix3d &ricj, Vector3d &ticj, double depth, Vector3d &uvi, Vector3d &uvj);
	void   updateLatestStates();
	void   fastPredictIMU(double t, Eigen::Vector3d linear_acceleration, Eigen::Vector3d angular_velocity);
	bool   IMUAvailable(double t);
	void   initFirstIMUPose(vector<pair<double, Eigen::Vector3d>> &accVector,
							Eigen::Vector3d						   mag = Eigen::Vector3d(0, 0, 0));

	enum SolverFlag { INITIAL, NON_LINEAR };

	enum MarginalizationFlag { MARGIN_OLD = 0, MARGIN_SECOND_NEW = 1 };

	std::mutex									 mBuf;
	std::mutex									 odomBuf;
	queue<pair<double, Eigen::Vector3d>>		 accBuf;
	queue<pair<double, Eigen::Vector3d>>		 gyrBuf;
	RW_Queue<pair<double, FeatureFrame>>		 featureBuf;
	double										 prevTime, curTime;
	bool										 openExEstimation;
	RW_Queue<cv::Mat>							 image_show_buf;
	RW_Queue<std::pair<double, Eigen::Vector3d>> magBuf;

	std::thread trackThread;
	std::thread processThread;
	std::thread depthThread;

	FeatureTracker::BaseFeatureTracker *featureTracker = nullptr;

	SolverFlag			solver_flag;
	MarginalizationFlag marginalization_flag;
	Vector3d			g;

	Matrix3d ric[2];
	Vector3d tic[2];

	Vector3d Ps[(WINDOW_SIZE + 1)];
	Vector3d Vs[(WINDOW_SIZE + 1)];
	Matrix3d Rs[(WINDOW_SIZE + 1)];
	Vector3d Bas[(WINDOW_SIZE + 1)];
	Vector3d Bgs[(WINDOW_SIZE + 1)];
	Vector3d Mw[(WINDOW_SIZE + 1)];	 // mag in world frame , ENU
	Vector3d Bms[(WINDOW_SIZE + 1)]; // bias of mag
	double	 td;

	Matrix3d back_R0, last_R, last_R0;
	Vector3d back_P0, last_P, last_P0;
	double	 Headers[(WINDOW_SIZE + 1)];
	Vector3d mag_measure[(WINDOW_SIZE + 1)]; // 磁力计测量值

	IntegrationBase *pre_integrations[(WINDOW_SIZE + 1)] = {0};
	Vector3d		 acc_0, gyr_0;

	vector<double>	 dt_buf[(WINDOW_SIZE + 1)];
	vector<Vector3d> linear_acceleration_buf[(WINDOW_SIZE + 1)];
	vector<Vector3d> angular_velocity_buf[(WINDOW_SIZE + 1)];

	int	   frame_count;
	int	   sum_of_outlier, sum_of_back, sum_of_front, sum_of_invalid;
	int	   inputImageCnt;
	float  sum_t_feature;
	int	   begin_time_count;
	int	   mea_track_count = 0;
	double mea_sum_time	   = 0;

	FeatureManager	  f_manager;
	InitialEXRotation initial_ex_rotation;

	bool first_imu;
	bool is_valid, is_key;
	bool failure_occur;

	vector<Vector3d> point_cloud;
	vector<Vector3d> margin_cloud;
	vector<Vector3d> key_poses;
	double			 initial_timestamp;


	double			   para_Pose[WINDOW_SIZE + 1][SIZE_POSE];
	double			   para_SpeedBias[WINDOW_SIZE + 1][SIZE_SPEEDBIAS];
	double			   para_mag[WINDOW_SIZE + 1][SIZE_MAG];
	double			   para_Feature[NUM_OF_F][SIZE_FEATURE];
	std::vector<int>   param_feature_id; // 后端参于计算的特征点id
	std::map<int, int> param_feature_id_to_index;
	double			   para_Ex_Pose[2][SIZE_POSE];
	double			   para_Retrive_Pose[SIZE_POSE];
	double			   para_Td[1][1];
	double			   para_Tr[1][1];

	int loop_window_index;

	MarginalizationInfo *last_marginalization_info = nullptr;
	vector<double *>	 last_marginalization_parameter_blocks;

	map<double, ImageFrame> all_image_frame;
	IntegrationBase		   *tmp_pre_integration = nullptr;

	Eigen::Vector3d initP;
	Eigen::Matrix3d initR;

	double			latest_time;
	Eigen::Vector3d latest_P, latest_V, latest_Ba, latest_Bg, latest_acc_0, latest_gyr_0, latest_Mw, latest_Bm,
		latest_mag_measure;
	Eigen::Quaterniond latest_Q;
	bool			   fast_prop_inited;

	bool initFirstPoseFlag;

	queue<double> fisheye_imgs_stampBuf;

	queue<std::vector<cv::cuda::GpuMat>> fisheye_imgs_upBuf_cuda;
	queue<std::vector<cv::cuda::GpuMat>> fisheye_imgs_downBuf_cuda;

	queue<std::vector<cv::Mat>>			fisheye_imgs_upBuf;
	queue<std::vector<cv::Mat>>			fisheye_imgs_downBuf;
	queue<std::pair<double, EigenPose>> odometry_buf;
};

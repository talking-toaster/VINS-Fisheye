/*******************************************************
 * Copyright (C) 2019, Aerial Robotics Group, Hong Kong University of Science and Technology
 *
 * This file is part of VINS.
 *
 * Licensed under the GNU General Public License v3.0;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#include "estimator/parameters.h"

double MAG_MEASURE_NOISE;
double MAG_WORLD_NOISE;
double MAG_BIAS_NOISE;


double INIT_DEPTH;
double MIN_PARALLAX;
double ACC_N, ACC_W;
double GYR_N, GYR_W;
double THRES_OUTLIER;
double triangulate_max_err = 0.5;

double IMU_FREQ;
double IMAGE_FREQ;
double FOCAL_LENGTH = 460;

std::vector<Eigen::Matrix3d> RIC;
std::vector<Eigen::Vector3d> TIC;

Eigen::Vector3d G{0.0, 0.0, 9.8};

double		BIAS_ACC_THRESHOLD;
double		BIAS_GYR_THRESHOLD;
double		SOLVER_TIME;
int			NUM_ITERATIONS;
int			ESTIMATE_EXTRINSIC;
int			ESTIMATE_TD;
int			ROLLING_SHUTTER;
std::string EX_CALIB_RESULT_PATH;
std::string VINS_RESULT_PATH;
std::string OUTPUT_FOLDER;
std::string IMU_TOPIC;
int			HEIGHT, WIDTH;
double		TD;
int			NUM_OF_CAM;
int			STEREO;


double depth_estimate_baseline;

int						  USE_IMU;
int						  USE_GPU;
int						  USE_NVIDIA_VPI;
int						  USE_ORB;
map<int, Eigen::Vector3d> pts_gt;
std::string				  IMAGE0_TOPIC, IMAGE1_TOPIC;
std::string				  FISHEYE_MASK;
std::vector<std::string>  CAM_NAMES;
int						  MAX_CNT;
int						  MAX_SOLVE_CNT;
int						  ENABLE_DEPTH;
int						  ENABLE_PERF_OUTPUT;
int						  MIN_DIST;
double					  F_THRESHOLD;
int						  SHOW_TRACK;
int						  FLOW_BACK;
int						  WARN_IMU_DURATION;

std::string MAG_TOPIC;
bool		USE_MAG;

std::string		configPath;
cv::FileStorage config;

template <typename T>
T readParam(std::string name) {
	T			 ans;
	cv::FileNode n = config[name];
	if (n.empty()) {
		ROS_ERROR_STREAM("Failed to find " << name << " in config file.");
		exit(-1);
	}
	n >> ans;
	return ans;
}

void readParameters(std::string config_file) {
	try {
		config.open(config_file.c_str(), cv::FileStorage::READ);
	} catch (cv::Exception &ex) {
		std::cerr << "ERROR:" << ex.what() << " Can't open config file" << std::endl;
	}
	if (!config.isOpened()) {
		std::cerr << "ERROR: Wrong path to settings" << std::endl;
	}
	USE_MAG = readParam<bool>("use_mag");
	if (USE_MAG) {
		MAG_TOPIC		  = readParam<std::string>("mag_topic");
		MAG_MEASURE_NOISE = readParam<double>("mag_measure_noise");
		MAG_WORLD_NOISE	  = readParam<double>("mag_world_noise");
		MAG_BIAS_NOISE	  = readParam<double>("mag_bias_noise");
	}


	IMAGE0_TOPIC = readParam<std::string>("image0_topic");
	IMAGE1_TOPIC = readParam<std::string>("image1_topic");

	MAX_CNT		  = readParam<int>("max_cnt");
	MAX_SOLVE_CNT = readParam<int>("max_solve_cnt");
	MIN_DIST	  = readParam<int>("min_dist");
	USE_ORB		  = readParam<int>("use_orb");

	SHOW_TRACK			= readParam<int>("show_track");
	FLOW_BACK			= readParam<int>("flow_back");
	ENABLE_DEPTH		= readParam<int>("enable_depth");
	THRES_OUTLIER		= readParam<double>("thres_outlier");
	triangulate_max_err = readParam<double>("tri_max_err");
	USE_GPU				= readParam<int>("use_gpu");
	USE_NVIDIA_VPI		= readParam<int>("use_nvidia_vpi");

#ifdef WITHOUT_CUDA
	if (USE_GPU) {
		std::cerr << "Compile with WITHOUT_CUDA mode, use_gpu is not supported!!!" << std::endl;
		exit(-1);
	}
#endif
	depth_estimate_baseline = readParam<double>("depth_estimate_baseline");
	ENABLE_PERF_OUTPUT		= readParam<int>("enable_perf_output");

	IMU_FREQ		  = readParam<double>("imu_freq");
	IMAGE_FREQ		  = readParam<double>("image_freq");
	WARN_IMU_DURATION = readParam<int>("warn_imu_duration");
	USE_IMU			  = readParam<int>("imu");

	printf("USE_IMU: %d\n", USE_IMU);
	if (USE_IMU) {
		IMU_TOPIC = readParam<std::string>("imu_topic");
		printf("IMU_TOPIC: %s\n", IMU_TOPIC.c_str());
		ACC_N = readParam<double>("acc_n");
		ACC_W = readParam<double>("acc_w");
		GYR_N = readParam<double>("gyr_n");
		GYR_W = readParam<double>("gyr_w");
		G.z() = readParam<double>("g_norm");
	}

	SOLVER_TIME	   = readParam<double>("max_solver_time");
	NUM_ITERATIONS = readParam<int>("max_num_iterations");
	MIN_PARALLAX   = readParam<double>("keyframe_parallax");
	MIN_PARALLAX   = MIN_PARALLAX / FOCAL_LENGTH;

	OUTPUT_FOLDER	 = readParam<std::string>("output_path");
	VINS_RESULT_PATH = OUTPUT_FOLDER + "/vio.csv";
	std::cout << "result path " << VINS_RESULT_PATH << std::endl;
	std::ofstream fout(VINS_RESULT_PATH, std::ios::out);
	fout.close();

	RIC.resize(2);
	TIC.resize(2);

	ESTIMATE_EXTRINSIC = readParam<int>("estimate_extrinsic");
	if (ESTIMATE_EXTRINSIC == 2) {
		ROS_WARN("have no prior about extrinsic param, calibrate extrinsic param");
		RIC[0]				 = Eigen::Matrix3d::Identity();
		TIC[0]				 = Eigen::Vector3d::Zero();
		EX_CALIB_RESULT_PATH = OUTPUT_FOLDER + "/extrinsic_parameter.csv";
	} else {
		if (ESTIMATE_EXTRINSIC == 1) {
			ROS_WARN(" Optimize extrinsic param around initial guess!");
			EX_CALIB_RESULT_PATH = OUTPUT_FOLDER + "/extrinsic_parameter.csv";
			ROS_WARN("extrinsic_parameter path: %s", EX_CALIB_RESULT_PATH.c_str());
		}
		if (ESTIMATE_EXTRINSIC == 0)
			ROS_WARN(" fix extrinsic param ");

		cv::Mat			cv_T = readParam<cv::Mat>("body_T_cam0");
		Eigen::Matrix4d T;
		cv::cv2eigen(cv_T, T);
		RIC[0] = T.block<3, 3>(0, 0);
		TIC[0] = T.block<3, 1>(0, 3);
	}
	NUM_OF_CAM = readParam<int>("num_of_cam");
	printf("camera number %d\n", NUM_OF_CAM);

	if (NUM_OF_CAM != 1 && NUM_OF_CAM != 2) {
		printf("num_of_cam should be 1 or 2\n");
		assert(0);
	}


	int pn	   = config_file.find_last_of('/');
	configPath = config_file.substr(0, pn);


	std::string cam0Calib = readParam<std::string>("cam0_calib");
	std::string cam0Path  = configPath + "/" + cam0Calib;
	CAM_NAMES.resize(2);

	CAM_NAMES[0] = cam0Path;

	if (NUM_OF_CAM == 2) {
		STEREO				  = 1;
		std::string cam1Calib = readParam<std::string>("cam1_calib");
		std::string cam1Path  = configPath + "/" + cam1Calib;

		CAM_NAMES[1] = cam1Path;

		cv::Mat			cv_T = readParam<cv::Mat>("body_T_cam1");
		Eigen::Matrix4d T;
		cv::cv2eigen(cv_T, T);
		RIC[1] = T.block<3, 3>(0, 0);
		TIC[1] = T.block<3, 1>(0, 3);
	}

	INIT_DEPTH		   = 5.0;
	BIAS_ACC_THRESHOLD = 0.1;
	BIAS_GYR_THRESHOLD = 0.1;

	TD			= readParam<double>("td");
	ESTIMATE_TD = readParam<int>("estimate_td");
	if (ESTIMATE_TD)
		ROS_INFO_STREAM("Unsynchronized sensors, online estimate time offset, initial td: " << TD);
	else
		ROS_INFO_STREAM("Synchronized sensors, fix time offset: " << TD);

	HEIGHT = readParam<int>("image_height");
	WIDTH  = readParam<int>("image_width");
	config.release();
}

/*******************************************************
 * Copyright (C) 2019, Aerial Robotics Group, Hong Kong University of Science and Technology
 *
 * This file is part of VINS.
 *
 * Licensed under the GNU General Public License v3.0;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#include "estimator.h"
#include "utility/visualization.h"
#include "featureTracker/feature_tracker_cpu.hpp"
#include "featureTracker/feature_tracker_cuda.hpp"
#include "featureTracker/feature_tracker_cuda_async.hpp"
#include "factor/mag_factor.hpp"

Estimator::Estimator() : f_manager{Rs} {
	ROS_INFO("init begins");
	clearState();
	prevTime		 = -1;
	curTime			 = 0;
	openExEstimation = 0;
	initP			 = Eigen::Vector3d(0, 0, 0);
	initR			 = Eigen::Matrix3d::Identity();
	inputImageCnt	 = 0;
	// sum_t_feature = 0.0;
	// begin_time_count = 10;
	initFirstPoseFlag = false;
}

void Estimator::setParameter() {

	if (USE_GPU) {
		// featureTracker = new FeatureTracker::PinholeFeatureTrackerCuda(this);
		featureTracker = new FeatureTracker::PinholeFeatureTrackerAsync(this);
	} else {
		featureTracker = new FeatureTracker::PinholeFeatureTrackerCPU(this);
	}

	f_manager.ft = featureTracker;

	for (int i = 0; i < NUM_OF_CAM; i++) {
		tic[i] = TIC[i];
		ric[i] = RIC[i];
		std::cout << " exitrinsic cam " << i << endl << ric[i] << endl << tic[i].transpose() << endl;
	}
	f_manager.setRic(ric);
	ProjectionTwoFrameOneCamFactor::sqrt_info = FOCAL_LENGTH / 1.5 * Matrix2d::Identity();
	ProjectionTwoFrameTwoCamFactor::sqrt_info = FOCAL_LENGTH / 1.5 * Matrix2d::Identity();
	ProjectionOneFrameTwoCamFactor::sqrt_info = FOCAL_LENGTH / 1.5 * Matrix2d::Identity();
	td										  = TD;
	g										  = G;
	std::cout << "set g " << g.transpose() << endl;

	featureTracker->readIntrinsicParameter(CAM_NAMES);

	processThread = std::thread(&Estimator::processMeasurements, this);
}

void Estimator::inputImage(double t, const cv::Mat &_img, const cv::Mat &_img1) {
	FeatureFrame featureFrame;

	featureFrame = featureTracker->trackImage(t, _img, _img1);

	if (SHOW_TRACK)
		image_show_buf.push(featureTracker->image_show);

	if ((++inputImageCnt) > 1) {
		// 原版是下面每2帧处理一次，这里改成每帧都处理，注意第一帧为空白帧(启动gpu)，传入会导致初始值无穷大
		// if ((inputImageCnt) % 2 == 0) {
		featureBuf.push(make_pair(t, featureFrame));
	}
}


bool Estimator::is_next_odometry_frame() {
	return (inputImageCnt % 2 == 1);
}



double base = 0;

void Estimator::inputIMU(double t, const Vector3d &linearAcceleration, const Vector3d &angularVelocity) {
	double dt_device = t - ros::Time::now().toSec();
	mBuf.lock();

	accBuf.push(make_pair(t, linearAcceleration));
	gyrBuf.push(make_pair(t, angularVelocity));

	if (fast_prop_inited) {
		double dt = t - latest_time;
		if (WARN_IMU_DURATION && (dt > (1.5 / IMU_FREQ) || dt < (0.5 / IMU_FREQ))) {
			ROS_WARN("[inputIMU] IMU sample duration not stable %4.2fms. Check your IMU and system performance",
					 dt * 1000);
		}

		fastPredictIMU(t, linearAcceleration, angularVelocity);
		pubLatestOdometry(latest_P, latest_Q, latest_V, t);

		// static int count = 0;
		// if (count++ % (int)(2*IMU_FREQ/IMAGE_FREQ) == 0) {
		//     double imu_propagate_dt = t - (Headers[frame_count] + td);
		//     printf("[inputIMU] IMU Propagate dt %4.1f ms Device dt %3.1fms", imu_propagate_dt*1000, dt_device*1000);
		// }
	}

	mBuf.unlock();
}

void Estimator::inputMag(double t, const Vector3d &mag) {
	static Utility::LPF mag_lpf(1 / 80, 80);
	if (magBuf.size() > 100)
		magBuf.pop();
	Eigen::Vector3d mag_lpfed = mag_lpf.update(mag);
	magBuf.push(make_pair(t, mag_lpfed));
}

void Estimator::inputFeature(double t, const FeatureFrame &featureFrame) {
	featureBuf.push(make_pair(t, featureFrame));
}

// 获取 (t0, t1] 间imu数据
bool Estimator::getIMUInterval(double t0, double t1, vector<pair<double, Eigen::Vector3d>> &accVector,
							   vector<pair<double, Eigen::Vector3d>> &gyrVector) {
	if (accBuf.empty()) {
		printf("not receive imu\n");
		return false;
	}
	// printf("get imu from %f %f\n", t0, t1);
	double t_ss = 0;
	double t_s	= 0;
	double t_e	= 0;
	if (t1 <= accBuf.back().first) {
		t_ss = accBuf.front().first;

		while (accBuf.front().first <= t0) {
			accBuf.pop();
			gyrBuf.pop();
		}

		t_s = accBuf.front().first;
		while (accBuf.front().first < t1) {
			t_e = accBuf.front().first;
			accVector.push_back(accBuf.front());
			accBuf.pop();
			gyrVector.push_back(gyrBuf.front());
			gyrBuf.pop();
		}
		accVector.push_back(accBuf.front());
		gyrVector.push_back(gyrBuf.front());
	} else {
		printf("wait for imu\n");
		return false;
	}

	if (fabs(t_s - t0) > 0.01 || fabs(t_e - t1) > 0.01) {
		ROS_WARN("IMU wrong sampling dt1 %f dts0 %fms dts %f dte %f\n", t1 - t0, t_ss - t0, t_s - t0, t_e - t0);
	}


	return true;
}

bool Estimator::getMag(double t, Eigen::Vector3d &mag) {
	if (magBuf.empty()) {
		printf("not receive mag\n");
		return false;
	}
	double			t1 = 0, t2, dt;
	Eigen::Vector3d prev_mag;
	Eigen::Vector3d next_mag;

	while (magBuf.front() && magBuf.front()->first < t) {
		std::pair<double, Eigen::Vector3d> mag_stamped;
		if (!magBuf.try_pop(mag_stamped)) {
			ROS_WARN("magBuf try_pop failed");
			return false;
		}
		prev_mag = mag_stamped.second;
		t1		 = t - mag_stamped.first;
	}
	// assert(!magBuf.empty() && "magBuf should not be empty");
	if (!magBuf.empty()) {
		next_mag = magBuf.front()->second;
		t2		 = magBuf.front()->first - t;
		assert(t1 >= 0 && t2 >= 0 && "t1 t2 should be positive");
		dt	= t1 + t2;
		mag = (t2 / dt) * prev_mag + (t1 / dt) * next_mag;
	} else {
		mag = prev_mag;
	}
	return true;
}

bool Estimator::IMUAvailable(double t) {
	if (!accBuf.empty() && t <= accBuf.back().first)
		return true;
	else
		return false;
}


void Estimator::processMeasurements() {

	while (1) {
		TicToc								  t_process;
		pair<double, FeatureFrame>			  feature;
		vector<pair<double, Eigen::Vector3d>> accVector, gyrVector;
		bool								  mag_ok = false;
		Eigen::Vector3d						  mag;
		if (featureBuf.try_pop(feature)) {
			if (solver_flag == Estimator::SolverFlag::NON_LINEAR) {
				while (featureBuf.try_pop(feature))
					;
			}


			curTime = feature.first + td;
			while (1) {
				if ((IMUAvailable(curTime)))
					break;
				else {
					ROS_INFO_THROTTLE(0.1, "wait for imu ... TD%f\n", td);
					std::this_thread::sleep_for(std::chrono::milliseconds(5));
				}
			}
			mBuf.lock();
			if (USE_IMU) {
				getIMUInterval(prevTime, curTime, accVector, gyrVector);
				if (curTime - prevTime > 2.1 / IMAGE_FREQ || accVector.size() / (curTime - prevTime) < IMU_FREQ * 0.8) {
					ROS_WARN("Long image dt %fms or wrong IMU rate %fhz", (curTime - prevTime) * 1000,
							 accVector.size() / (curTime - prevTime));
				}
			}
			mBuf.unlock();


			if (USE_IMU) {
				if (!initFirstPoseFlag) {
					if (USE_MAG) {
						while (!getMag(curTime, mag)) {
							ROS_WARN("wait for mag");
							std::this_thread::sleep_for(std::chrono::milliseconds(5));
						}
						// ROS_INFO_STREAM("========mag:" << mag.transpose());
						initFirstIMUPose(accVector, mag);
					} else {
						initFirstIMUPose(accVector);
					}
				}
				for (size_t i = 0; i < accVector.size(); i++) {
					double dt;
					if (i == 0)
						dt = accVector[i].first - prevTime;
					else if (i == accVector.size() - 1)
						dt = curTime - accVector[i - 1].first;
					else
						dt = accVector[i].first - accVector[i - 1].first;
					processIMU(accVector[i].first, dt, accVector[i].second, gyrVector[i].second);
				}
				if (USE_MAG)
					processMag(mag);
			}

			processImage(feature.second, feature.first);
			prevTime = curTime;

			printStatistics(*this, feature.first);

			std_msgs::Header header;
			header.frame_id = "world";
			header.stamp	= ros::Time(feature.first);

			pubIMUBias(latest_Ba, latest_Bg, header);
			pubOdometry(*this, header);
			pubKeyPoses(*this, header);
			pubCameraPose(*this, header);
			pubPointCloud(*this, header);
			pubKeyframe(*this);
			pubTF(*this, header);
			pub_Log(*this, header);



			if (ENABLE_PERF_OUTPUT) {
				double dt = t_process.toc();
				mea_sum_time += dt;
				mea_track_count++;
				ROS_INFO("%s[backend]process measurement time: AVG %f NOW %f%s %s \n", GREEN,
						 mea_sum_time / mea_track_count, dt, GREEN, C_RESET);
				// ROS_INFO("feature buf size:%d", int(featureBuf.size()));
				//  avg cost 5 ms
			}


		} else {
			std::chrono::milliseconds dura(2);
			std::this_thread::sleep_for(dura);
		}
	}
}


void Estimator::initFirstIMUPose(vector<pair<double, Eigen::Vector3d>> &accVector, Eigen::Vector3d mag) {
	printf("init first imu pose\n");
	initFirstPoseFlag = true;
	// return;
	Eigen::Vector3d averAcc(0, 0, 0);
	int				n = (int)accVector.size();
	for (size_t i = 0; i < accVector.size(); i++) {
		averAcc = averAcc + accVector[i].second;
	}
	averAcc = averAcc / n;
	printf("averge acc %f %f %f\n", averAcc.x(), averAcc.y(), averAcc.z());
	Matrix3d R0	 = Utility::g2R(averAcc);
	double	 yaw = Utility::R2ypr(R0).x();
	R0			 = Utility::ypr2R(Eigen::Vector3d{-yaw, 0, 0}) * R0;
	ROS_INFO_STREAM("mag:" << mag.transpose());
	if (USE_MAG && mag.norm() > 0) {
		Vector3d mag_enu = R0 * mag;
		if (mag_enu.y() == 0)
			mag_enu.y() = 1e-20;
		double mag_yaw = atan2(mag_enu.x(), mag_enu.y()) * 180 / M_PI; // 正东为0度
		mag_yaw += 6;
		R0 = Utility::ypr2R(Eigen::Vector3d{mag_yaw, 0, 0}) * R0;
		ROS_WARN("init yaw by mag %f", mag_yaw);
	}
	Rs[0] = R0;
	std::cout << "init R0 " << endl << Rs[0] << endl;
	ROS_INFO_STREAM("init ypr:" << Utility::R2ypr(R0).transpose());
}

void Estimator::processMag(const Vector3d &mag) {
	if (frame_count == 0) {
		Mw[0]  = Rs[0] * mag;
		Bms[0] = Vector3d(0, 0, 0);
	} else {
		Mw[frame_count]	 = Mw[frame_count - 1];
		Bms[frame_count] = Bms[frame_count - 1];
	}
	mag_measure[frame_count] = mag;
}

void Estimator::processIMU(double t, double dt, const Vector3d &linear_acceleration, const Vector3d &angular_velocity) {
	if (!first_imu) {
		first_imu = true;
		acc_0	  = linear_acceleration;
		gyr_0	  = angular_velocity;
	}

	if (!pre_integrations[frame_count]) {
		pre_integrations[frame_count] = new IntegrationBase{acc_0, gyr_0, Bas[frame_count], Bgs[frame_count]};
	}
	if (frame_count != 0) {
		pre_integrations[frame_count]->push_back(dt, linear_acceleration, angular_velocity);
		// if(solver_flag != NON_LINEAR)
		tmp_pre_integration->push_back(dt, linear_acceleration, angular_velocity);

		dt_buf[frame_count].push_back(dt);
		linear_acceleration_buf[frame_count].push_back(linear_acceleration);
		angular_velocity_buf[frame_count].push_back(angular_velocity);

		int		 j		  = frame_count;
		Vector3d un_acc_0 = Rs[j] * (acc_0 - Bas[j]) - g;
		Vector3d un_gyr	  = 0.5 * (gyr_0 + angular_velocity) - Bgs[j];
		Rs[j] *= Utility::deltaQ(un_gyr * dt).toRotationMatrix();
		Vector3d un_acc_1 = Rs[j] * (linear_acceleration - Bas[j]) - g;
		Vector3d un_acc	  = 0.5 * (un_acc_0 + un_acc_1);
		Ps[j] += dt * Vs[j] + 0.5 * dt * dt * un_acc;
		Vs[j] += dt * un_acc;
	}
	acc_0 = linear_acceleration;
	gyr_0 = angular_velocity;
}

void Estimator::processImage(const FeatureFrame &image, const double header) {
	ROS_DEBUG("new image coming ------------------------------------------");
	ROS_DEBUG("Adding feature points %lu", image.size());
	if (f_manager.addFeatureCheckParallax(frame_count, image, td)) {
		marginalization_flag = MARGIN_OLD;
		// printf("keyframe\n");
	} else {
		marginalization_flag = MARGIN_SECOND_NEW;
		// printf("non-keyframe\n");
	}

	ROS_DEBUG("%s", marginalization_flag ? "Non-keyframe" : "Keyframe");
	ROS_DEBUG("Solving %d", frame_count);
	ROS_DEBUG("number of feature: %d", f_manager.getFeatureCount());
	Headers[frame_count] = header;

	ImageFrame imageframe(image, header);
	imageframe.pre_integration = tmp_pre_integration;
	all_image_frame.insert(make_pair(header, imageframe));
	tmp_pre_integration = new IntegrationBase{acc_0, gyr_0, Bas[frame_count], Bgs[frame_count]};

	if (ESTIMATE_EXTRINSIC == 2) {
		ROS_INFO("calibrating extrinsic param, rotation movement is needed");
		if (frame_count != 0) {
			vector<pair<Vector3d, Vector3d>> corres = f_manager.getCorresponding(frame_count - 1, frame_count);
			Matrix3d						 calib_ric;
			if (initial_ex_rotation.CalibrationExRotation(corres, pre_integrations[frame_count]->delta_q, calib_ric)) {
				ROS_WARN("initial extrinsic rotation calib success");
				ROS_WARN_STREAM("initial extrinsic rotation: " << endl << calib_ric);
				ric[0]			   = calib_ric;
				RIC[0]			   = calib_ric;
				ESTIMATE_EXTRINSIC = 1;
			}
		}
	}

	if (solver_flag == INITIAL) {

		base = ros::Time::now().toSec();

		// stereo + IMU initilization
		// if(STEREO && USE_IMU)
		{
			ROS_INFO("Init by pose pnp...");
			f_manager.initFramePoseByPnP(frame_count, Ps, Rs, tic, ric);
			TicToc t_ic;
			f_manager.triangulate(frame_count, Ps, Rs, tic, ric);
			if (ENABLE_PERF_OUTPUT) {
				ROS_INFO("Triangulation cost %3.1fms..", t_ic.toc());
			}
			if (frame_count == WINDOW_SIZE) {
				map<double, ImageFrame>::iterator frame_it;
				int								  i = 0;
				for (frame_it = all_image_frame.begin(); frame_it != all_image_frame.end(); frame_it++) {
					frame_it->second.R = Rs[i];
					frame_it->second.T = Ps[i];
					i++;
				}
				solveGyroscopeBias(all_image_frame, Bgs);
				for (int i = 0; i <= WINDOW_SIZE; i++) {
					pre_integrations[i]->repropagate(Vector3d::Zero(), Bgs[i]);
				}

				solver_flag = NON_LINEAR;
				optimization();
				slideWindow();

				// set<int> removeIndex;
				// outliersRejection(removeIndex);
				// exit(-1);

				ROS_INFO("Initialization finish!");
			}
		}

		if (frame_count < WINDOW_SIZE) {
			frame_count++;
			int prev_frame	 = frame_count - 1;
			Ps[frame_count]	 = Ps[prev_frame];
			Vs[frame_count]	 = Vs[prev_frame];
			Rs[frame_count]	 = Rs[prev_frame];
			Bas[frame_count] = Bas[prev_frame];
			Bgs[frame_count] = Bgs[prev_frame];
			if (USE_MAG) {
				Mw[frame_count]			 = Mw[prev_frame];
				Bms[frame_count]		 = Bms[prev_frame];
				mag_measure[frame_count] = mag_measure[prev_frame];
			}
		}

	} else {
		TicToc t_solve;
		f_manager.triangulate(frame_count, Ps, Rs, tic, ric);

		if (ENABLE_PERF_OUTPUT) {
			ROS_INFO("Triangulation cost %3.1fms..", t_solve.toc());
			t_solve.tic();
		}

		optimization();

		if (ENABLE_PERF_OUTPUT) {
			ROS_INFO("after optimization cost %fms..", t_solve.toc());
			t_solve.tic();
		}

		set<int> removeIndex;
		outliersRejection(removeIndex);
		if (ENABLE_PERF_OUTPUT) {
			ROS_INFO("Remove %ld outlier", removeIndex.size());
		}

		f_manager.removeOutlier(removeIndex);
		predictPtsInNextFrame();

		if (ENABLE_PERF_OUTPUT) {
			ROS_INFO("outlier reject costs: %fms", t_solve.toc());
			t_solve.tic();
		}

		if (failureDetection()) {
			ROS_WARN("failure detection!");
			failure_occur = 1;
			clearState();
			setParameter();
			ROS_WARN("system reboot!");
			// exit(-1);
			cv::waitKey(-1);
			return;
		}

		slideWindow();

		if (ENABLE_PERF_OUTPUT) {
			ROS_INFO("to slideWindow costs: %fms", t_solve.toc());
			t_solve.tic();
		}

		f_manager.removeFailures();
		// prepare output of VINS
		key_poses.clear();
		for (int i = 0; i <= WINDOW_SIZE; i++)
			key_poses.push_back(Ps[i]);

		last_R	= Rs[WINDOW_SIZE];
		last_P	= Ps[WINDOW_SIZE];
		last_R0 = Rs[0];
		last_P0 = Ps[0];

		odomBuf.lock();
		odometry_buf.push(make_pair(header, make_pair(last_R, last_P)));
		odomBuf.unlock();

		updateLatestStates();
		if (ENABLE_PERF_OUTPUT) {
			ROS_INFO("after updateLatestStates costs: %fms", t_solve.toc());
		}
	}
}
void Estimator::initFirstPose(Eigen::Vector3d p, Eigen::Matrix3d r) {
	Ps[0] = p;
	Rs[0] = r;
	initP = p;
	initR = r;
}


void Estimator::clearState() {
	for (int i = 0; i < WINDOW_SIZE + 1; i++) {
		Rs[i].setIdentity();
		Ps[i].setZero();
		Vs[i].setZero();
		Bas[i].setZero();
		Bgs[i].setZero();
		if (USE_MAG) {
			Mw[i].setZero();
			Bms[i].setZero();
			mag_measure[i].setZero();
		}
		dt_buf[i].clear();
		linear_acceleration_buf[i].clear();
		angular_velocity_buf[i].clear();

		if (pre_integrations[i] != nullptr) {
			delete pre_integrations[i];
		}
		pre_integrations[i] = nullptr;
	}

	for (int i = 0; i < NUM_OF_CAM; i++) {
		tic[i] = Vector3d::Zero();
		ric[i] = Matrix3d::Identity();
	}

	first_imu = false, sum_of_back = 0;
	sum_of_front	  = 0;
	frame_count		  = 0;
	solver_flag		  = INITIAL;
	latest_P		  = Eigen::Vector3d::Zero();
	latest_V		  = Eigen::Vector3d::Zero();
	latest_Q		  = Eigen::Quaterniond::Identity();
	fast_prop_inited  = false;
	initial_timestamp = 0;
	all_image_frame.clear();

	if (tmp_pre_integration != nullptr)
		delete tmp_pre_integration;
	if (last_marginalization_info != nullptr)
		delete last_marginalization_info;

	tmp_pre_integration		  = nullptr;
	last_marginalization_info = nullptr;
	last_marginalization_parameter_blocks.clear();

	f_manager.clearState();

	failure_occur = 0;
}
void Estimator::vector2double() {
	for (int i = 0; i <= WINDOW_SIZE; i++) {
		para_Pose[i][0] = Ps[i].x();
		para_Pose[i][1] = Ps[i].y();
		para_Pose[i][2] = Ps[i].z();
		Quaterniond q{Rs[i]};
		para_Pose[i][3] = q.x();
		para_Pose[i][4] = q.y();
		para_Pose[i][5] = q.z();
		para_Pose[i][6] = q.w();

		if (USE_IMU) {
			para_SpeedBias[i][0] = Vs[i].x();
			para_SpeedBias[i][1] = Vs[i].y();
			para_SpeedBias[i][2] = Vs[i].z();

			para_SpeedBias[i][3] = Bas[i].x();
			para_SpeedBias[i][4] = Bas[i].y();
			para_SpeedBias[i][5] = Bas[i].z();

			para_SpeedBias[i][6] = Bgs[i].x();
			para_SpeedBias[i][7] = Bgs[i].y();
			para_SpeedBias[i][8] = Bgs[i].z();
		}
		if (USE_MAG) {
			para_mag[i][0] = Mw[i].x();
			para_mag[i][1] = Mw[i].y();
			para_mag[i][2] = Mw[i].z();
			para_mag[i][3] = Bms[i].x();
			para_mag[i][4] = Bms[i].y();
			para_mag[i][5] = Bms[i].z();
		}
	}

	for (int i = 0; i < NUM_OF_CAM; i++) {
		para_Ex_Pose[i][0] = tic[i].x();
		para_Ex_Pose[i][1] = tic[i].y();
		para_Ex_Pose[i][2] = tic[i].z();
		Quaterniond q{ric[i]};
		para_Ex_Pose[i][3] = q.x();
		para_Ex_Pose[i][4] = q.y();
		para_Ex_Pose[i][5] = q.z();
		para_Ex_Pose[i][6] = q.w();
	}


	auto deps = f_manager.getDepthVector();
	param_feature_id.clear();
	// printf("Solve features: %ld;", deps.size());
	for (auto &it : deps) {
		// ROS_INFO("Feature %d invdepth %f feature index %d", it.first, it.second, param_feature_id.size());
		para_Feature[param_feature_id.size()][0] = it.second;
		param_feature_id_to_index[it.first]		 = param_feature_id.size();
		param_feature_id.push_back(it.first);
	}


	para_Td[0][0] = td;
}

void Estimator::double2vector() {
	Vector3d origin_R0 = Utility::R2ypr(Rs[0]);
	Vector3d origin_P0 = Ps[0];

	if (failure_occur) {
		origin_R0	  = Utility::R2ypr(last_R0);
		origin_P0	  = last_P0;
		failure_occur = 0;
	}

	if (USE_IMU) {
		Vector3d origin_R00 = Utility::R2ypr(
			Quaterniond(para_Pose[0][6], para_Pose[0][3], para_Pose[0][4], para_Pose[0][5]).toRotationMatrix());
		double y_diff = origin_R0.x() - origin_R00.x();
		// TODO
		// ???
		Matrix3d rot_diff = Utility::ypr2R(Vector3d(y_diff, 0, 0));
		if (abs(abs(origin_R0.y()) - 90) < 1.0 || abs(abs(origin_R00.y()) - 90) < 1.0) {
			ROS_DEBUG("euler singular point!");
			rot_diff = Rs[0] * Quaterniond(para_Pose[0][6], para_Pose[0][3], para_Pose[0][4], para_Pose[0][5])
								   .toRotationMatrix()
								   .transpose();
		}

		for (int i = 0; i <= WINDOW_SIZE; i++) {

			Rs[i] = rot_diff * Quaterniond(para_Pose[i][6], para_Pose[i][3], para_Pose[i][4], para_Pose[i][5])
								   .normalized()
								   .toRotationMatrix();

			Ps[i] = rot_diff * Vector3d(para_Pose[i][0] - para_Pose[0][0], para_Pose[i][1] - para_Pose[0][1],
										para_Pose[i][2] - para_Pose[0][2]) +
					origin_P0;


			Vs[i] = rot_diff * Vector3d(para_SpeedBias[i][0], para_SpeedBias[i][1], para_SpeedBias[i][2]);

			Bas[i] = Vector3d(para_SpeedBias[i][3], para_SpeedBias[i][4], para_SpeedBias[i][5]);

			Bgs[i] = Vector3d(para_SpeedBias[i][6], para_SpeedBias[i][7], para_SpeedBias[i][8]);
		}
	} else {
		for (int i = 0; i <= WINDOW_SIZE; i++) {
			Rs[i] = Quaterniond(para_Pose[i][6], para_Pose[i][3], para_Pose[i][4], para_Pose[i][5])
						.normalized()
						.toRotationMatrix();

			Ps[i] = Vector3d(para_Pose[i][0], para_Pose[i][1], para_Pose[i][2]);
		}
	}
	if (USE_MAG) {
		for (int i = 0; i <= WINDOW_SIZE; i++) {
			Mw[i]  = Vector3d(para_mag[i][0], para_mag[i][1], para_mag[i][2]);
			Bms[i] = Vector3d(para_mag[i][3], para_mag[i][4], para_mag[i][5]);
		}
	}

	if (USE_IMU) {
		for (int i = 0; i < NUM_OF_CAM; i++) {
			tic[i] = Vector3d(para_Ex_Pose[i][0], para_Ex_Pose[i][1], para_Ex_Pose[i][2]);
			ric[i] = Quaterniond(para_Ex_Pose[i][6], para_Ex_Pose[i][3], para_Ex_Pose[i][4], para_Ex_Pose[i][5])
						 .normalized()
						 .toRotationMatrix();
		}
	}

	std::map<int, double> deps;
	for (unsigned int i = 0; i < param_feature_id.size(); i++) {
		int _id = param_feature_id[i];
		// ROS_INFO("Id %d depth %f", i, 1/para_Feature[i][0]);
		deps[_id] = para_Feature[i][0];
	}

	f_manager.setDepth(deps);

	if (USE_IMU)
		td = para_Td[0][0];
}

bool Estimator::failureDetection() {
	return false;
	if (f_manager.last_track_num < 2) {
		ROS_INFO(" little feature %d", f_manager.last_track_num);
		// return true;
	}
	if (Bas[WINDOW_SIZE].norm() > 2.5) {
		ROS_INFO(" big IMU acc bias estimation %f", Bas[WINDOW_SIZE].norm());
		return true;
	}
	if (Bgs[WINDOW_SIZE].norm() > 1.0) {
		ROS_INFO(" big IMU gyr bias estimation %f", Bgs[WINDOW_SIZE].norm());
		return true;
	}
	if (USE_MAG) {
		if (Bms[WINDOW_SIZE].norm() > 0.5) {
			ROS_INFO(" big mag bias estimation %f", Bms[WINDOW_SIZE].norm());
			return true;
		}
	}
	/*
	if (tic(0) > 1)
	{
		ROS_INFO(" big extri param estimation %d", tic(0) > 1);
		return true;
	}
	*/
	Vector3d tmp_P = Ps[WINDOW_SIZE];
	if ((tmp_P - last_P).norm() > 5) {
		// ROS_INFO(" big translation");
		// return true;
	}
	if (abs(tmp_P.z() - last_P.z()) > 1) {
		// ROS_INFO(" big z translation");
		// return true;
	}
	Matrix3d	tmp_R	= Rs[WINDOW_SIZE];
	Matrix3d	delta_R = tmp_R.transpose() * last_R;
	Quaterniond delta_Q(delta_R);
	double		delta_angle;
	delta_angle = acos(delta_Q.w()) * 2.0 / 3.14 * 180.0;
	if (delta_angle > 50) {
		ROS_INFO(" big delta_angle ");
		// return true;
	}
	return false;
}

void Estimator::optimization() {
	TicToc t_whole, t_prepare;
	vector2double();

	ceres::Problem		 problem;
	ceres::LossFunction *loss_function;
	// loss_function = NULL;
	loss_function = new ceres::HuberLoss(1.0);
	// loss_function = new ceres::CauchyLoss(1.0 / FOCAL_LENGTH);
	// ceres::LossFunction* loss_function = new ceres::HuberLoss(1.0);
	for (int i = 0; i < frame_count + 1; i++) {
		ceres::LocalParameterization *local_parameterization = new PoseLocalParameterization();
		problem.AddParameterBlock(para_Pose[i], SIZE_POSE, local_parameterization);
		if (USE_IMU)
			problem.AddParameterBlock(para_SpeedBias[i], SIZE_SPEEDBIAS);
		if (USE_MAG)
			problem.AddParameterBlock(para_mag[i], SIZE_MAG);
	}
	if (!USE_IMU)
		problem.SetParameterBlockConstant(para_Pose[0]);

	for (int i = 0; i < NUM_OF_CAM; i++) {
		ceres::LocalParameterization *local_parameterization = new PoseLocalParameterization();
		problem.AddParameterBlock(para_Ex_Pose[i], SIZE_POSE, local_parameterization);
		if ((ESTIMATE_EXTRINSIC && frame_count == WINDOW_SIZE && Vs[0].norm() > 0.2) || openExEstimation) {
			// ROS_INFO("estimate extinsic param");
			openExEstimation = 1;
		} else {
			// ROS_INFO("fix extinsic param");
			problem.SetParameterBlockConstant(para_Ex_Pose[i]);
		}
	}
	problem.AddParameterBlock(para_Td[0], 1);

	if (!ESTIMATE_TD || Vs[0].norm() < 0.2)
		problem.SetParameterBlockConstant(para_Td[0]);

	if (last_marginalization_info && last_marginalization_info->valid) {
		// construct new marginlization_factor
		MarginalizationFactor *marginalization_factor = new MarginalizationFactor(last_marginalization_info);
		problem.AddResidualBlock(marginalization_factor, NULL, last_marginalization_parameter_blocks);
	}
	if (USE_IMU) {
		for (int i = 0; i < frame_count; i++) {
			int j = i + 1;
			if (pre_integrations[j]->sum_dt > 10.0)
				continue;
			IMUFactor *imu_factor = new IMUFactor(pre_integrations[j]);
			problem.AddResidualBlock(imu_factor, NULL, para_Pose[i], para_SpeedBias[i], para_Pose[j],
									 para_SpeedBias[j]);
		}
	}
	if (USE_MAG) {
		Eigen::Matrix3d mag_measure_sqrt_info =
			Eigen::LLT<Eigen::Matrix3d>(Eigen::Matrix3d::Identity() / MAG_MEASURE_NOISE).matrixL().transpose();
		Eigen::Matrix3d mag_world_sqrt_info =
			Eigen::LLT<Eigen::Matrix3d>(Eigen::Matrix3d::Identity() / MAG_WORLD_NOISE).matrixL().transpose();
		Eigen::Matrix3d mag_bias_sqrt_info =
			Eigen::LLT<Eigen::Matrix3d>(Eigen::Matrix3d::Identity() / MAG_BIAS_NOISE).matrixL().transpose();
		for (int i = 0; i < frame_count; i++) {
			int	 j			= i + 1;
			auto mag_factor = MagFactor::Create(mag_measure[i], mag_measure_sqrt_info);
			problem.AddResidualBlock(mag_factor, NULL, para_Pose[i], para_mag[i]);
			auto mag_world_factor = MagWorldFactor::Create(mag_world_sqrt_info);
			problem.AddResidualBlock(mag_world_factor, NULL, para_mag[i], para_mag[j]);
			auto mag_bias_factor = MagBiasFactor::Create(mag_bias_sqrt_info);
			problem.AddResidualBlock(mag_bias_factor, NULL, para_mag[i], para_mag[j]);
		}
	}

	int f_m_cnt = 0;

	// for (auto &_it : f_manager.feature)
	for (int _id : param_feature_id) {
		auto &it_per_id	   = f_manager.feature[_id];
		it_per_id.used_num = it_per_id.feature_per_frame.size();

		int feature_index = param_feature_id_to_index[it_per_id.feature_id];

		int imu_i = it_per_id.start_frame, imu_j = imu_i - 1;

		Vector3d pts_i = it_per_id.feature_per_frame[0].point;

		// ROS_INFO("Adding feature id %d initial depth", it_per_id.feature_id, it_);
		for (auto &it_per_frame : it_per_id.feature_per_frame) {
			imu_j++;
			if (imu_i != imu_j) {
				Vector3d						pts_j = it_per_frame.point;
				ProjectionTwoFrameOneCamFactor *f_td  = new ProjectionTwoFrameOneCamFactor(
					 pts_i, pts_j, it_per_id.feature_per_frame[0].velocity, it_per_frame.velocity,
					 it_per_id.feature_per_frame[0].cur_td, it_per_frame.cur_td);
				// std::vector<double*> param_blocks;
				// param_blocks.push_back(para_Pose[imu_i]);
				// param_blocks.push_back(para_Pose[imu_j]);
				// param_blocks.push_back(para_Ex_Pose[0]);
				// param_blocks.push_back(para_Feature[feature_index]);
				// param_blocks.push_back(para_Td[0]);
				// ROS_INFO("Check ProjectionTwoFrameOneCamFactor");
				// f_td->check(param_blocks.data());
				// exit(-1);
				problem.AddResidualBlock(f_td, loss_function, para_Pose[imu_i], para_Pose[imu_j],
										 para_Ex_Pose[it_per_id.main_cam], para_Feature[feature_index], para_Td[0]);
			}

			if (STEREO && it_per_frame.is_stereo) {
				// For stereo point; main cam must be 0 now
				Vector3d pts_j_right = it_per_frame.pointRight;
				if (imu_i != imu_j) {
					ProjectionTwoFrameTwoCamFactor *f = new ProjectionTwoFrameTwoCamFactor(
						pts_i, pts_j_right, it_per_id.feature_per_frame[0].velocity, it_per_frame.velocityRight,
						it_per_id.feature_per_frame[0].cur_td, it_per_frame.cur_td);

					// std::vector<double*> param_blocks;
					// param_blocks.push_back(para_Pose[imu_i]);
					// param_blocks.push_back(para_Pose[imu_j]);
					// param_blocks.push_back(para_Ex_Pose[0]);
					// param_blocks.push_back(para_Ex_Pose[1]);
					// param_blocks.push_back(para_Feature[feature_index]);
					// param_blocks.push_back(para_Td[0]);
					// ROS_INFO("Check ProjectionTwoFrameTwoCamFactor");
					// f->check(param_blocks.data());
					problem.AddResidualBlock(f, loss_function, para_Pose[imu_i], para_Pose[imu_j], para_Ex_Pose[0],
											 para_Ex_Pose[1], para_Feature[feature_index], para_Td[0]);
				} else {
					ProjectionOneFrameTwoCamFactor *f = new ProjectionOneFrameTwoCamFactor(
						pts_i, pts_j_right, it_per_id.feature_per_frame[0].velocity, it_per_frame.velocityRight,
						it_per_id.feature_per_frame[0].cur_td, it_per_frame.cur_td);

					std::vector<double *> param_blocks;
					param_blocks.push_back(para_Ex_Pose[0]);
					param_blocks.push_back(para_Ex_Pose[1]);
					param_blocks.push_back(para_Feature[feature_index]);
					param_blocks.push_back(para_Td[0]);
					// ROS_INFO("Check ProjectionOneFrameTwoCamFactor ID: %d, index %d depth init %f Velocity L %f %f %f
					// R %f %f %f", it_per_id.feature_id, feature_index,
					//     para_Feature[feature_index][0],
					//     it_per_id.feature_per_frame[0].velocity.x(), it_per_id.feature_per_frame[0].velocity.y(),
					//     it_per_id.feature_per_frame[0].velocity.z(), it_per_frame.velocityRight.x(),
					//     it_per_frame.velocityRight.y(), it_per_frame.velocityRight.z()
					//     );
					// f->check(param_blocks.data());
					// exit(-1);

					problem.AddResidualBlock(f, loss_function, para_Ex_Pose[0], para_Ex_Pose[1],
											 para_Feature[feature_index], para_Td[0]);
				}
			}
			f_m_cnt++;
		}
	}

	ROS_DEBUG("visual measurement count: %d", f_m_cnt);
	// printf("prepare for ceres: %f \n", t_prepare.toc());

	ceres::Solver::Options options;

	options.linear_solver_type = ceres::DENSE_SCHUR;
	// options.use_explicit_schur_complement = true;
	options.num_threads				   = 4;
	options.trust_region_strategy_type = ceres::DOGLEG;
	options.max_num_iterations		   = NUM_ITERATIONS;
	// options.check_gradients = true;
	// options.use_explicit_schur_complement = true;
	// options.minimizer_progress_to_stdout = true;
	// options.use_nonmonotonic_steps = true;
	if (marginalization_flag == MARGIN_OLD)
		options.max_solver_time_in_seconds = SOLVER_TIME * 4.0 / 5.0;
	else
		options.max_solver_time_in_seconds = SOLVER_TIME;
	TicToc				   t_solver;
	ceres::Solver::Summary summary;
	ceres::Solve(options, &problem, &summary);
	std::cout << summary.BriefReport() << endl;
	// std::cout << summary.FullReport() << endl;
	static double sum_iterations = 0;
	static double sum_solve_time = 0;
	static int	  solve_count	 = 0;
	sum_iterations				 = sum_iterations + summary.iterations.size();
	sum_solve_time				 = sum_solve_time + summary.total_time_in_seconds;
	solve_count += 1;

	if (ENABLE_PERF_OUTPUT) {
		ROS_INFO("AVG Iter %f time %fms Iterations : %d solver costs: %f ", sum_iterations / solve_count,
				 sum_solve_time * 1000 / solve_count, static_cast<int>(summary.iterations.size()), t_solver.toc());
	}

	double2vector();
	// printf("frame_count: %d \n", frame_count);

	if (frame_count < WINDOW_SIZE)
		return;

	TicToc t_whole_marginalization;
	if (marginalization_flag == MARGIN_OLD) {
		MarginalizationInfo *marginalization_info = new MarginalizationInfo();
		vector2double();

		if (last_marginalization_info && last_marginalization_info->valid) {
			vector<int> drop_set;
			for (int i = 0; i < static_cast<int>(last_marginalization_parameter_blocks.size()); i++) {
				if (last_marginalization_parameter_blocks[i] == para_Pose[0] ||
					last_marginalization_parameter_blocks[i] == para_SpeedBias[0])
					drop_set.push_back(i);
			}
			// construct new marginlization_factor
			MarginalizationFactor *marginalization_factor = new MarginalizationFactor(last_marginalization_info);
			ResidualBlockInfo	  *residual_block_info =
				new ResidualBlockInfo(marginalization_factor, NULL, last_marginalization_parameter_blocks, drop_set);
			marginalization_info->addResidualBlockInfo(residual_block_info);
		}

		if (USE_IMU) {
			if (pre_integrations[1]->sum_dt < 10.0) {
				IMUFactor		  *imu_factor		   = new IMUFactor(pre_integrations[1]);
				ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(
					imu_factor, NULL,
					vector<double *>{para_Pose[0], para_SpeedBias[0], para_Pose[1], para_SpeedBias[1]},
					vector<int>{0, 1});
				marginalization_info->addResidualBlockInfo(residual_block_info);
			}
		}

		for (int _id : param_feature_id) {
			auto &it_per_id = f_manager.feature[_id];

			int feature_index = param_feature_id_to_index[it_per_id.feature_id];

			int imu_i = it_per_id.start_frame, imu_j = imu_i - 1;
			if (imu_i != 0)
				continue;

			Vector3d pts_i = it_per_id.feature_per_frame[0].point;

			for (auto &it_per_frame : it_per_id.feature_per_frame) {
				imu_j++;
				if (imu_i != imu_j) {
					Vector3d						pts_j = it_per_frame.point;
					ProjectionTwoFrameOneCamFactor *f_td  = new ProjectionTwoFrameOneCamFactor(
						 pts_i, pts_j, it_per_id.feature_per_frame[0].velocity, it_per_frame.velocity,
						 it_per_id.feature_per_frame[0].cur_td, it_per_frame.cur_td);
					ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(
						f_td, loss_function,
						vector<double *>{para_Pose[imu_i], para_Pose[imu_j], para_Ex_Pose[it_per_id.main_cam],
										 para_Feature[feature_index], para_Td[0]},
						vector<int>{0, 3});
					marginalization_info->addResidualBlockInfo(residual_block_info);
				}
				if (STEREO && it_per_frame.is_stereo) {
					Vector3d pts_j_right = it_per_frame.pointRight;
					if (imu_i != imu_j) {
						ProjectionTwoFrameTwoCamFactor *f = new ProjectionTwoFrameTwoCamFactor(
							pts_i, pts_j_right, it_per_id.feature_per_frame[0].velocity, it_per_frame.velocityRight,
							it_per_id.feature_per_frame[0].cur_td, it_per_frame.cur_td);
						ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(
							f, loss_function,
							vector<double *>{para_Pose[imu_i], para_Pose[imu_j], para_Ex_Pose[it_per_id.main_cam],
											 para_Ex_Pose[1], para_Feature[feature_index], para_Td[0]},
							vector<int>{0, 4});
						marginalization_info->addResidualBlockInfo(residual_block_info);
					} else {
						ProjectionOneFrameTwoCamFactor *f = new ProjectionOneFrameTwoCamFactor(
							pts_i, pts_j_right, it_per_id.feature_per_frame[0].velocity, it_per_frame.velocityRight,
							it_per_id.feature_per_frame[0].cur_td, it_per_frame.cur_td);
						ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(
							f, loss_function,
							vector<double *>{para_Ex_Pose[0], para_Ex_Pose[1], para_Feature[feature_index], para_Td[0]},
							vector<int>{2});
						marginalization_info->addResidualBlockInfo(residual_block_info);
					}
				}
			}
		}

		TicToc t_pre_margin;
		marginalization_info->preMarginalize();

		// ROS_INFO("pre marginalization %f ms", t_pre_margin.toc());

		TicToc t_margin;
		marginalization_info->marginalize();
		// ROS_INFO("marginalization %f ms", t_margin.toc());

		std::unordered_map<long, double *> addr_shift;
		for (int i = 1; i <= WINDOW_SIZE; i++) {
			addr_shift[reinterpret_cast<long>(para_Pose[i])] = para_Pose[i - 1];
			if (USE_IMU)
				addr_shift[reinterpret_cast<long>(para_SpeedBias[i])] = para_SpeedBias[i - 1];
		}
		for (int i = 0; i < NUM_OF_CAM; i++)
			addr_shift[reinterpret_cast<long>(para_Ex_Pose[i])] = para_Ex_Pose[i];

		addr_shift[reinterpret_cast<long>(para_Td[0])] = para_Td[0];

		vector<double *> parameter_blocks = marginalization_info->getParameterBlocks(addr_shift);

		if (last_marginalization_info)
			delete last_marginalization_info;
		last_marginalization_info			  = marginalization_info;
		last_marginalization_parameter_blocks = parameter_blocks;

	} else {
		if (last_marginalization_info &&
			std::count(std::begin(last_marginalization_parameter_blocks),
					   std::end(last_marginalization_parameter_blocks), para_Pose[WINDOW_SIZE - 1])) {

			MarginalizationInfo *marginalization_info = new MarginalizationInfo();
			vector2double();
			if (last_marginalization_info && last_marginalization_info->valid) {
				vector<int> drop_set;
				for (int i = 0; i < static_cast<int>(last_marginalization_parameter_blocks.size()); i++) {
					ROS_ASSERT(last_marginalization_parameter_blocks[i] != para_SpeedBias[WINDOW_SIZE - 1]);
					if (last_marginalization_parameter_blocks[i] == para_Pose[WINDOW_SIZE - 1])
						drop_set.push_back(i);
				}
				// construct new marginlization_factor
				MarginalizationFactor *marginalization_factor = new MarginalizationFactor(last_marginalization_info);
				ResidualBlockInfo	  *residual_block_info	  = new ResidualBlockInfo(
					   marginalization_factor, NULL, last_marginalization_parameter_blocks, drop_set);

				marginalization_info->addResidualBlockInfo(residual_block_info);
			}

			TicToc t_pre_margin;
			// ROS_INFO("begin marginalization");
			marginalization_info->preMarginalize();
			ROS_INFO("end pre marginalization, %f ms", t_pre_margin.toc());

			TicToc t_margin;
			// ROS_INFO("begin marginalization");
			marginalization_info->marginalize();
			ROS_INFO("end marginalization, %f ms", t_margin.toc());

			std::unordered_map<long, double *> addr_shift;
			for (int i = 0; i <= WINDOW_SIZE; i++) {
				if (i == WINDOW_SIZE - 1)
					continue;
				else if (i == WINDOW_SIZE) {
					addr_shift[reinterpret_cast<long>(para_Pose[i])] = para_Pose[i - 1];
					if (USE_IMU)
						addr_shift[reinterpret_cast<long>(para_SpeedBias[i])] = para_SpeedBias[i - 1];
				} else {
					addr_shift[reinterpret_cast<long>(para_Pose[i])] = para_Pose[i];
					if (USE_IMU)
						addr_shift[reinterpret_cast<long>(para_SpeedBias[i])] = para_SpeedBias[i];
				}
			}
			for (int i = 0; i < NUM_OF_CAM; i++)
				addr_shift[reinterpret_cast<long>(para_Ex_Pose[i])] = para_Ex_Pose[i];

			addr_shift[reinterpret_cast<long>(para_Td[0])] = para_Td[0];


			vector<double *> parameter_blocks = marginalization_info->getParameterBlocks(addr_shift);
			if (last_marginalization_info)
				delete last_marginalization_info;
			last_marginalization_info			  = marginalization_info;
			last_marginalization_parameter_blocks = parameter_blocks;
		}
	}
	if (ENABLE_PERF_OUTPUT) {
		ROS_INFO("whole marginalization costs: %fms \n", t_whole_marginalization.toc());
	}
	// printf("whole time for ceres: %f \n", t_whole.toc());
}

void Estimator::slideWindow() {
	TicToc t_margin;
	if (marginalization_flag == MARGIN_OLD) {
		double t_0 = Headers[0];
		back_R0	   = Rs[0];
		back_P0	   = Ps[0];
		if (frame_count == WINDOW_SIZE) {
			for (int i = 0; i < WINDOW_SIZE; i++) {
				Headers[i] = Headers[i + 1];
				Rs[i].swap(Rs[i + 1]);
				Ps[i].swap(Ps[i + 1]);
				if (USE_IMU) {
					std::swap(pre_integrations[i], pre_integrations[i + 1]);

					dt_buf[i].swap(dt_buf[i + 1]);
					linear_acceleration_buf[i].swap(linear_acceleration_buf[i + 1]);
					angular_velocity_buf[i].swap(angular_velocity_buf[i + 1]);

					Vs[i].swap(Vs[i + 1]);
					Bas[i].swap(Bas[i + 1]);
					Bgs[i].swap(Bgs[i + 1]);
				}
				if (USE_MAG) {
					Mw[i].swap(Mw[i + 1]);
					Bms[i].swap(Bms[i + 1]);
					mag_measure[i].swap(mag_measure[i + 1]);
				}
			}
			Headers[WINDOW_SIZE] = Headers[WINDOW_SIZE - 1];
			Ps[WINDOW_SIZE]		 = Ps[WINDOW_SIZE - 1];
			Rs[WINDOW_SIZE]		 = Rs[WINDOW_SIZE - 1];

			if (USE_IMU) {
				Vs[WINDOW_SIZE]	 = Vs[WINDOW_SIZE - 1];
				Bas[WINDOW_SIZE] = Bas[WINDOW_SIZE - 1];
				Bgs[WINDOW_SIZE] = Bgs[WINDOW_SIZE - 1];

				delete pre_integrations[WINDOW_SIZE];
				pre_integrations[WINDOW_SIZE] = new IntegrationBase{acc_0, gyr_0, Bas[WINDOW_SIZE], Bgs[WINDOW_SIZE]};

				dt_buf[WINDOW_SIZE].clear();
				linear_acceleration_buf[WINDOW_SIZE].clear();
				angular_velocity_buf[WINDOW_SIZE].clear();
			}
			if (USE_MAG) {
				Mw[WINDOW_SIZE]	 = Mw[WINDOW_SIZE - 1];
				Bms[WINDOW_SIZE] = Bms[WINDOW_SIZE - 1];
			}

			if (true || solver_flag == INITIAL) {
				map<double, ImageFrame>::iterator it_0;
				it_0 = all_image_frame.find(t_0);
				delete it_0->second.pre_integration;
				all_image_frame.erase(all_image_frame.begin(), it_0);
			}
			slideWindowOld();
		}
	} else {
		if (frame_count == WINDOW_SIZE) {
			Headers[frame_count - 1] = Headers[frame_count];
			Ps[frame_count - 1]		 = Ps[frame_count];
			Rs[frame_count - 1]		 = Rs[frame_count];

			if (USE_IMU) {
				for (unsigned int i = 0; i < dt_buf[frame_count].size(); i++) {
					double	 tmp_dt					 = dt_buf[frame_count][i];
					Vector3d tmp_linear_acceleration = linear_acceleration_buf[frame_count][i];
					Vector3d tmp_angular_velocity	 = angular_velocity_buf[frame_count][i];

					pre_integrations[frame_count - 1]->push_back(tmp_dt, tmp_linear_acceleration, tmp_angular_velocity);

					dt_buf[frame_count - 1].push_back(tmp_dt);
					linear_acceleration_buf[frame_count - 1].push_back(tmp_linear_acceleration);
					angular_velocity_buf[frame_count - 1].push_back(tmp_angular_velocity);
				}

				Vs[frame_count - 1]	 = Vs[frame_count];
				Bas[frame_count - 1] = Bas[frame_count];
				Bgs[frame_count - 1] = Bgs[frame_count];

				delete pre_integrations[WINDOW_SIZE];
				pre_integrations[WINDOW_SIZE] = new IntegrationBase{acc_0, gyr_0, Bas[WINDOW_SIZE], Bgs[WINDOW_SIZE]};

				dt_buf[WINDOW_SIZE].clear();
				linear_acceleration_buf[WINDOW_SIZE].clear();
				angular_velocity_buf[WINDOW_SIZE].clear();
			}
			if (USE_MAG) {
				Mw[frame_count - 1]			 = Mw[frame_count];
				Bms[frame_count - 1]		 = Bms[frame_count];
				mag_measure[frame_count - 1] = mag_measure[frame_count];
			}
			slideWindowNew();
		}
	}
}

void Estimator::slideWindowNew() {
	sum_of_front++;
	f_manager.removeFront(frame_count);
}

void Estimator::slideWindowOld() {
	sum_of_back++;

	bool shift_depth = solver_flag == NON_LINEAR ? true : false;
	if (shift_depth) {
		Matrix3d R0, R1;
		Vector3d P0, P1;
		R0 = back_R0 * ric[0];
		R1 = Rs[0] * ric[0];
		P0 = back_P0 + back_R0 * tic[0];
		P1 = Ps[0] + Rs[0] * tic[0];
		f_manager.removeBackShiftDepth(R0, P0, R1, P1);
	} else
		f_manager.removeBack();
}


void Estimator::getPoseInWorldFrame(Eigen::Matrix4d &T) {
	T					= Eigen::Matrix4d::Identity();
	T.block<3, 3>(0, 0) = Rs[frame_count];
	T.block<3, 1>(0, 3) = Ps[frame_count];
}

void Estimator::getPoseInWorldFrame(int index, Eigen::Matrix4d &T) {
	T					= Eigen::Matrix4d::Identity();
	T.block<3, 3>(0, 0) = Rs[index];
	T.block<3, 1>(0, 3) = Ps[index];
}

void Estimator::predictPtsInNextFrame() {
	// printf("predict pts in next frame\n");
	if (frame_count < 2)
		return;
	// predict next pose. Assume constant velocity motion
	Eigen::Matrix4d curT, prevT, nextT;
	getPoseInWorldFrame(curT);
	getPoseInWorldFrame(frame_count - 1, prevT);
	nextT = curT * (prevT.inverse() * curT);
	map<int, Eigen::Vector3d> predictPts;
	map<int, Eigen::Vector3d> predictPts1;

	for (auto &_it : f_manager.feature) {
		auto &it_per_id = _it.second;
		if (it_per_id.estimated_depth > 0) {
			int firstIndex = it_per_id.start_frame;
			int lastIndex  = it_per_id.start_frame + it_per_id.feature_per_frame.size() - 1;
			// printf("cur frame index  %d last frame index %d\n", frame_count, lastIndex);
			if ((int)it_per_id.feature_per_frame.size() >= 2 && lastIndex == frame_count) {
				double	 depth		  = it_per_id.estimated_depth;
				Vector3d pts_j		  = ric[0] * (depth * it_per_id.feature_per_frame[0].point) + tic[0];
				Vector3d pts_w		  = Rs[firstIndex] * pts_j + Ps[firstIndex];
				Vector3d pts_local	  = nextT.block<3, 3>(0, 0).transpose() * (pts_w - nextT.block<3, 1>(0, 3));
				Vector3d pts_cam	  = ric[0].transpose() * (pts_local - tic[0]);
				Vector3d pts_cam1	  = ric[1].transpose() * (pts_local - tic[1]);
				int		 ptsIndex	  = it_per_id.feature_id;
				predictPts[ptsIndex]  = pts_cam;
				predictPts1[ptsIndex] = pts_cam1;
			}
		}
	}
	featureTracker->setPrediction(predictPts, predictPts1);
}

double Estimator::reprojectionError(Matrix3d &Ri, Vector3d &Pi, Matrix3d &rici, Vector3d &tici, Matrix3d &Rj,
									Vector3d &Pj, Matrix3d &ricj, Vector3d &ticj, double depth, Vector3d &uvi,
									Vector3d &uvj) {
	Vector3d pts_w	= Ri * (rici * (depth * uvi) + tici) + Pi;
	Vector3d pts_cj = ricj.transpose() * (Rj.transpose() * (pts_w - Pj) - ticj);
	return (pts_cj.normalized() - uvj).norm();
}

void Estimator::outliersRejection(set<int> &removeIndex) {
	for (int _id : param_feature_id) {
		auto  &it_per_id   = f_manager.feature[_id];
		double err		   = 0;
		int	   errCnt	   = 0;
		it_per_id.used_num = it_per_id.feature_per_frame.size();

		int		 imu_i = it_per_id.start_frame, imu_j = imu_i - 1;
		Vector3d pts_i = it_per_id.feature_per_frame[0].point;
		double	 depth = it_per_id.estimated_depth;

		Vector3d pts_w = Rs[imu_i] * (ric[it_per_id.main_cam] * (depth * pts_i) + tic[it_per_id.main_cam]) + Ps[imu_i];
		// ROS_INFO("PT %d, STEREO %d w %3.2f %3.2f %3.2f drone %3.2f %3.2f %3.2f ptun %3.2f %3.2f %3.2f, depth %f",
		//     it_per_id.feature_id,
		//     it_per_id.feature_per_frame.front().is_stereo,
		//     pts_w.x(), pts_w.y(), pts_w.z(),
		//     Ps[imu_i].x(), Ps[imu_i].y(), Ps[imu_i].z(),
		//     pts_i.x(), pts_i.y(), pts_i.z(),
		//     depth
		// );

		for (auto &it_per_frame : it_per_id.feature_per_frame) {

			imu_j++;

			if (imu_i != imu_j) {
				Vector3d pts_j = it_per_frame.point;

				double tmp_error =
					reprojectionError(Rs[imu_i], Ps[imu_i], ric[it_per_id.main_cam], tic[it_per_id.main_cam], Rs[imu_j],
									  Ps[imu_j], ric[it_per_id.main_cam], tic[it_per_id.main_cam], depth, pts_i, pts_j);
				// printf("ptun   %3.2f %3.2f %3.2f: %3.2f\n", pts_j.x(), pts_j.y(), pts_j.z(), tmp_error);

				err += tmp_error;
				errCnt++;
				// printf("tmp_error %f\n", FOCAL_LENGTH / 1.5 * tmp_error);
			}
			// need to rewrite projecton factor.........
			if (STEREO && it_per_frame.is_stereo) {

				Vector3d pts_j_right = it_per_frame.pointRight;

				if (imu_i != imu_j) {
					double tmp_error = reprojectionError(Rs[imu_i], Ps[imu_i], ric[0], tic[0], Rs[imu_j], Ps[imu_j],
														 ric[1], tic[1], depth, pts_i, pts_j_right);
					err += tmp_error;
					errCnt++;
					// printf("ptright %3.2f %3.2f %3.2f: %3.2f\n", pts_j_right.x(), pts_j_right.y(), pts_j_right.z(),
					// tmp_error*FOCAL_LENGTH);
				} else {
					double tmp_error = reprojectionError(Rs[imu_i], Ps[imu_i], ric[0], tic[0], Rs[imu_j], Ps[imu_j],
														 ric[1], tic[1], depth, pts_i, pts_j_right);
					err += tmp_error;
					errCnt++;
					// printf("ptright %3.2f %3.2f %3.2f: %3.2f\n", pts_j_right.x(), pts_j_right.y(), pts_j_right.z(),
					// tmp_error*FOCAL_LENGTH);
				}
			}
		}

		// printf("\n");
		double ave_err = err / errCnt;
		// Looks we have some bugs on outlier rejection!
		if (ave_err * FOCAL_LENGTH > THRES_OUTLIER) {
			// ROS_INFO("Removing feature %d on cam %d...  error %f", it_per_id.feature_id, it_per_id.main_cam, ave_err
			// * FOCAL_LENGTH);
			removeIndex.insert(it_per_id.feature_id);
		}
	}
}

void Estimator::fastPredictIMU(double t, Eigen::Vector3d linear_acceleration, Eigen::Vector3d angular_velocity) {
	// ROS_INFO
	if (latest_time < 10) {
		return;
	}

	double dt = t - latest_time;
	if (WARN_IMU_DURATION && dt > (1.5 / IMU_FREQ)) {
		ROS_ERROR("[FastPredictIMU] dt %4.1fms t %f lt %f", dt * 1000, (t - base) * 1000, (latest_time - base) * 1000);
	}

	latest_time = t;

	Eigen::Vector3d un_acc_0 = latest_Q * (latest_acc_0 - latest_Ba) - g;
	Eigen::Vector3d un_gyr	 = 0.5 * (latest_gyr_0 + angular_velocity) - latest_Bg;
	latest_Q				 = latest_Q * Utility::deltaQ(un_gyr * dt);
	latest_Q.normalize();
	Eigen::Vector3d un_acc_1 = latest_Q * (linear_acceleration - latest_Ba) - g;
	Eigen::Vector3d un_acc	 = 0.5 * (un_acc_0 + un_acc_1);
	latest_P				 = latest_P + dt * latest_V + 0.5 * dt * dt * un_acc;
	latest_V				 = latest_V + dt * un_acc;
	latest_acc_0			 = linear_acceleration;
	latest_gyr_0			 = angular_velocity;
}

void Estimator::updateLatestStates() {
	mBuf.lock();

	latest_time = Headers[frame_count] + td;
	latest_P	= Ps[frame_count];
	// std::cout << "Ps[frame_count] is " << Ps[frame_count].transpose();
	latest_Q										= Rs[frame_count];
	latest_V										= Vs[frame_count];
	latest_Ba										= Bas[frame_count];
	latest_Bg										= Bgs[frame_count];
	latest_Mw										= Mw[frame_count];
	latest_Bm										= Bms[frame_count];
	latest_mag_measure								= mag_measure[frame_count];
	fast_prop_inited								= true;
	latest_acc_0									= acc_0;
	latest_gyr_0									= gyr_0;
	queue<pair<double, Eigen::Vector3d>> tmp_accBuf = accBuf;
	queue<pair<double, Eigen::Vector3d>> tmp_gyrBuf = gyrBuf;

	double re_propagate_dt = accBuf.back().first - latest_time;

	if (re_propagate_dt > 3.0 / IMAGE_FREQ) {
		ROS_WARN("[updateLatestStates] Reprogate dt too high %4.1fms ", re_propagate_dt * 1000);
	}

	while (!tmp_accBuf.empty()) {
		double			t	= tmp_accBuf.front().first;
		Eigen::Vector3d acc = tmp_accBuf.front().second;
		Eigen::Vector3d gyr = tmp_gyrBuf.front().second;
		double			dt	= t - latest_time;
		if (WARN_IMU_DURATION && dt > 1.5 / IMU_FREQ) {
			ROS_ERROR("[updateLatestStates]IMU sample duration too high %4.2fms. Check your IMU and system performance",
					  dt * 1000);
			// exit(-1);
		}

		fastPredictIMU(t, acc, gyr);
		tmp_accBuf.pop();
		tmp_gyrBuf.pop();
	}
	mBuf.unlock();
}

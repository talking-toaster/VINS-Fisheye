#include "vins_base.hpp"
#include "estimator/estimator.h"
#include "estimator/parameters.h"

using namespace FeatureTracker;

void VinsNodeBaseClass::imgs_callback(const sensor_msgs::ImageConstPtr &img1_msg,
									  const sensor_msgs::ImageConstPtr &img2_msg) {
	auto img1 = getImageFromMsg(img1_msg);
	auto img2 = getImageFromMsg(img2_msg);
	// cv::imwrite("/swarm/fisheye_ws/output/img_left1.png", img1->image);
	// cv::imwrite("/swarm/fisheye_ws/output/img_right1.png", img2->image);
	estimator.inputImage(img1_msg->header.stamp.toSec(), img1->image, img2->image);
	// stero_buf.push(std::make_pair(img1->image, img2->image));
}

void VinsNodeBaseClass::imu_callback(const sensor_msgs::ImuConstPtr &imu_msg) {
	double	 t	= imu_msg->header.stamp.toSec();
	double	 dx = imu_msg->linear_acceleration.x;
	double	 dy = imu_msg->linear_acceleration.y;
	double	 dz = imu_msg->linear_acceleration.z;
	double	 rx = imu_msg->angular_velocity.x;
	double	 ry = imu_msg->angular_velocity.y;
	double	 rz = imu_msg->angular_velocity.z;
	Vector3d acc(dx, dy, dz);
	Vector3d gyr(rx, ry, rz);
	estimator.inputIMU(t, acc, gyr);
}

void VinsNodeBaseClass::restart_callback(const std_msgs::BoolConstPtr &restart_msg) {
	if (restart_msg->data == true) {
		ROS_WARN("restart the estimator!");
		estimator.clearState();
		estimator.setParameter();
	}
	return;
}


void VinsNodeBaseClass::Init(ros::NodeHandle &n) {
	std::string config_file;
	n.getParam("config_file", config_file);

	ROS_INFO_STREAM("config file is " << config_file);
	readParameters(config_file);

	estimator.setParameter();

#ifdef EIGEN_DONT_PARALLELIZE
	ROS_DEBUG("EIGEN_DONT_PARALLELIZE");
#endif

	ROS_WARN("waiting for image and imu...");

	registerPub(n);

	// We use blank images to initialize cuda before every thing
	// if (USE_GPU) {
	// 	TicToc	t_gpu;
	// 	cv::Mat mat(WIDTH, HEIGHT, CV_8UC1);
	// 	estimator.inputImage(0, mat, mat);
	// 	ROS_INFO_STREAM("Initialize with blank cost " << t_gpu.toc() << " ms.");
	// }

	sub_imu		= n.subscribe(IMU_TOPIC, 20, &VinsNodeBaseClass::imu_callback, (VinsNodeBaseClass *)this,
							  ros::TransportHints().tcpNoDelay(true));
	sub_restart = n.subscribe("/vins_restart", 10, &VinsNodeBaseClass::restart_callback, (VinsNodeBaseClass *)this,
							  ros::TransportHints().tcpNoDelay(true));

	ROS_INFO("Will directly receive raw images %s and %s", IMAGE0_TOPIC.c_str(), IMAGE1_TOPIC.c_str());
	image_sub_l = new message_filters::Subscriber<sensor_msgs::Image>(n, IMAGE0_TOPIC, 10,
																	  ros::TransportHints().tcpNoDelay(true));
	image_sub_r = new message_filters::Subscriber<sensor_msgs::Image>(n, IMAGE1_TOPIC, 10,
																	  ros::TransportHints().tcpNoDelay(true));
	sync =
		new message_filters::TimeSynchronizer<sensor_msgs::Image, sensor_msgs::Image>(*image_sub_l, *image_sub_r, 1000);
	sync->registerCallback(boost::bind(&VinsNodeBaseClass::imgs_callback, (VinsNodeBaseClass *)this, _1, _2));

	if (SHOW_TRACK) {
		show_track_thread = std::thread([&]() {
			while (1) {
				cv::Mat img_show;
				while (estimator.image_show_buf.try_pop(img_show)) {
					if (img_show.cols > 0)
						cv::imshow("track", img_show);
				}
				cv::waitKey(1);
			}
		});
	}

	// depth_estimator_thread = std::thread([&]() {
	// 	while (1) {
	// 		std::pair<cv::Mat, cv::Mat> stero_pair;
	// 		if (stero_buf.try_pop(stero_pair)) {
	// 			while (stero_buf.try_pop(stero_pair))
	// 				;
	// 			TicToc t_depth;
	// 			depth_estimator.calculate_depth(stero_pair.first, stero_pair.second);
	// 			ROS_INFO_STREAM("depth used: " << t_depth.toc() << " ms.");
	// 			cv::imshow("depth", depth_estimator.depth_img);
	// 			cv::waitKey(1);
	// 		}
	// 	}
	// });
}
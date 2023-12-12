#include <stdio.h>
#include <queue>
#include <map>
#include <mutex>
#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include "utility/visualization.h"
#include "utility/tic_toc.h"

#include <boost/thread.hpp>
#include "vins/FlattenImages.h"

#include "utility/opencv_cuda.h"
#include "utility/ros_utility.h"
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <sensor_msgs/CompressedImage.h>
#include <sensor_msgs/MagneticField.h>
#include "estimator/depth_estimator.hpp"
#include "utility/queue_wrapper.hpp"

class Estimator;

class VinsNodeBaseClass {
	message_filters::Subscriber<sensor_msgs::Image>							  *image_sub_l;
	message_filters::Subscriber<sensor_msgs::Image>							  *image_sub_r;
	message_filters::TimeSynchronizer<sensor_msgs::Image, sensor_msgs::Image> *sync;
	std::thread																   show_track_thread;
	std::thread																   depth_estimator_thread;
	RW_Queue<std::pair<cv::Mat, cv::Mat>>									   stero_buf;

  public:
	Estimator			estimator;
	vpi::DepthEstimator depth_estimator; // nvidia vpi depth estimator
	ros::Subscriber		sub_imu;
	ros::Subscriber		sub_feature;
	ros::Subscriber		sub_restart;
	ros::Subscriber		sub_mag; // 订阅磁力计数据

  protected:
	void imgs_callback(const sensor_msgs::ImageConstPtr &img1_msg, const sensor_msgs::ImageConstPtr &img2_msg);
	void imu_callback(const sensor_msgs::ImuConstPtr &imu_msg);
	void mag_callback(const sensor_msgs::MagneticFieldConstPtr &mag_msg);
	void restart_callback(const std_msgs::BoolConstPtr &restart_msg);

	virtual void Init(ros::NodeHandle &n);
};
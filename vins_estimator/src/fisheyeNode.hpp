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

class Estimator;
class FisheyeUndist;
class DepthCamManager;



class VinsNodeBaseClass {
        message_filters::Subscriber<sensor_msgs::Image> * image_sub_l;
        message_filters::Subscriber<sensor_msgs::Image> * image_sub_r;
        message_filters::TimeSynchronizer<sensor_msgs::Image, sensor_msgs::Image> * sync;

        message_filters::Subscriber<sensor_msgs::CompressedImage> * comp_image_sub_l;
        message_filters::Subscriber<sensor_msgs::CompressedImage> * comp_image_sub_r;
        message_filters::TimeSynchronizer<sensor_msgs::CompressedImage, sensor_msgs::CompressedImage> * comp_sync;

        DepthCamManager * cam_manager = nullptr;


        double t_last = 0;

        double last_time;
        
        bool is_color = true;

        double t_last_send = 0;
        std::mutex pack_and_send_mtx;
        bool need_to_pack_and_send = false;

        CvCudaImages cur_up_color_cuda, cur_down_color_cuda;
        CvCudaImages cur_up_gray_cuda, cur_down_gray_cuda;

        CvImages cur_up_color, cur_down_color;
        CvImages cur_up_gray, cur_down_gray;

        double cur_frame_t;

        Estimator estimator;
        ros::Subscriber sub_imu;
        ros::Subscriber sub_feature;
        ros::Subscriber sub_restart;
        ros::Subscriber flatten_sub;

    protected:
        void imgs_callback(const sensor_msgs::ImageConstPtr &img1_msg, const sensor_msgs::ImageConstPtr &img2_msg);
        
        void comp_imgs_callback(const sensor_msgs::CompressedImageConstPtr &img1_msg, const sensor_msgs::CompressedImageConstPtr &img2_msg);

        void imu_callback(const sensor_msgs::ImuConstPtr &imu_msg);

        void restart_callback(const std_msgs::BoolConstPtr &restart_msg);

        virtual void Init(ros::NodeHandle & n);
};
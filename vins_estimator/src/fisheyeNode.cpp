#include "fisheyeNode.hpp"
#include "featureTracker/feature_tracker_fisheye.hpp"
#include "featureTracker/fisheye_undist.hpp"
#include "estimator/estimator.h"
#include "estimator/parameters.h"
#include "depth_generation/depth_camera_manager.h"

using namespace FeatureTracker;     


void VinsNodeBaseClass::fisheye_comp_imgs_callback(const sensor_msgs::CompressedImageConstPtr &img1_msg, const sensor_msgs::CompressedImageConstPtr &img2_msg) {
    TicToc tic_input;
    auto img1 = getImageFromMsg(img1_msg);
    auto img2 = getImageFromMsg(img2_msg);

    fisheye_handler->imgs_callback(img1_msg->header.stamp.toSec(), img1, img2);

    if (img1_msg->header.stamp.toSec() - t_last > 0.11) {
        ROS_WARN("Duration between two images is %fms", img1_msg->header.stamp.toSec() - t_last);
    }
    t_last = img1_msg->header.stamp.toSec();
}

void VinsNodeBaseClass::imgs_callback(const sensor_msgs::ImageConstPtr &img1_msg, const sensor_msgs::ImageConstPtr &img2_msg)
{
    auto img1 = getImageFromMsg(img1_msg);
    auto img2 = getImageFromMsg(img2_msg);
    estimator.inputImage(img1_msg->header.stamp.toSec(), img1->image, img2->image);
}


void VinsNodeBaseClass::comp_imgs_callback(const sensor_msgs::CompressedImageConstPtr &img1_msg, const sensor_msgs::CompressedImageConstPtr &img2_msg)
{
    auto img1 = getImageFromMsg(img1_msg, cv::IMREAD_GRAYSCALE);
    auto img2 = getImageFromMsg(img2_msg, cv::IMREAD_GRAYSCALE);
    estimator.inputImage(img1_msg->header.stamp.toSec(), img1, img2);
}

void VinsNodeBaseClass::imu_callback(const sensor_msgs::ImuConstPtr &imu_msg)
{
    double t = imu_msg->header.stamp.toSec();
    double dx = imu_msg->linear_acceleration.x;
    double dy = imu_msg->linear_acceleration.y;
    double dz = imu_msg->linear_acceleration.z;
    double rx = imu_msg->angular_velocity.x;
    double ry = imu_msg->angular_velocity.y;
    double rz = imu_msg->angular_velocity.z;
    Vector3d acc(dx, dy, dz);
    Vector3d gyr(rx, ry, rz);
    estimator.inputIMU(t, acc, gyr);
}

void VinsNodeBaseClass::restart_callback(const std_msgs::BoolConstPtr &restart_msg)
{
    if (restart_msg->data == true)
    {
        ROS_WARN("restart the estimator!");
        estimator.clearState();
        estimator.setParameter();
    }
    return;
}


void VinsNodeBaseClass::Init(ros::NodeHandle & n)
{
    std::string config_file;
    n.getParam("config_file", config_file);
    
    std::cout << "config file is " << config_file << '\n';
    readParameters(config_file);

    estimator.setParameter();

    ROS_INFO("Will %d GPU", USE_GPU);
    if (ENABLE_DEPTH) {
        FisheyeUndist *fun = nullptr;
        if (USE_GPU) {
            auto ft = (BaseFisheyeFeatureTracker<cv::cuda::GpuMat> *)
                estimator.featureTracker;
            fun = ft->get_fisheye_undist(0);
        } else {
            auto ft = (BaseFisheyeFeatureTracker<cv::Mat> *)
                estimator.featureTracker;
            fun = ft->get_fisheye_undist(0);
        }

        cam_manager = new DepthCamManager(n, fun);
        cam_manager -> init_with_extrinsic(estimator.ric[0], estimator.tic[0], estimator.ric[1], estimator.tic[1]);
        estimator.depth_cam_manager = cam_manager;
    }
#ifdef EIGEN_DONT_PARALLELIZE
    ROS_DEBUG("EIGEN_DONT_PARALLELIZE");
#endif

    ROS_WARN("waiting for image and imu...");

    registerPub(n);

    //We use blank images to initialize cuda before every thing
    if (USE_GPU) {
        // TODO : 生成 宽*高图像，estimator.inputImage()
        /*
        cv::Mat mat(fisheye_handler->raw_width(), fisheye_handler->raw_height(), CV_8UC3);
        fisheye_handler->imgs_callback(0, mat, mat, true);
            estimator.inputFisheyeImage(0, 
            fisheye_handler->fisheye_up_imgs_cuda_gray, fisheye_handler->fisheye_down_imgs_cuda_gray, true);
        */   
        std::cout<< "Initialize with blank cost" << blank.toc() << std::endl;
    }

    sub_imu = n.subscribe(IMU_TOPIC, 2000, &VinsNodeBaseClass::imu_callback, (VinsNodeBaseClass*)this, ros::TransportHints().tcpNoDelay(true));
    sub_restart = n.subscribe("/vins_restart", 100, &VinsNodeBaseClass::restart_callback, (VinsNodeBaseClass*)this, ros::TransportHints().tcpNoDelay(true));

    if (IS_COMP_IMAGES) {
        ROS_INFO("Will directly receive compressed images %s and %s", COMP_IMAGE0_TOPIC.c_str(), COMP_IMAGE1_TOPIC.c_str());
        comp_image_sub_l = new message_filters::Subscriber<sensor_msgs::CompressedImage> (n, COMP_IMAGE0_TOPIC, 1000, ros::TransportHints().tcpNoDelay(true));
        comp_image_sub_r = new message_filters::Subscriber<sensor_msgs::CompressedImage> (n, COMP_IMAGE1_TOPIC, 1000, ros::TransportHints().tcpNoDelay(true));
        comp_sync = new message_filters::TimeSynchronizer<sensor_msgs::CompressedImage, sensor_msgs::CompressedImage> (*comp_image_sub_l, *comp_image_sub_r, 1000);
        comp_sync->registerCallback(boost::bind(&VinsNodeBaseClass::comp_imgs_callback, (VinsNodeBaseClass*)this, _1, _2));
    } else {
        ROS_INFO("Will directly receive raw images %s and %s", IMAGE0_TOPIC.c_str(), IMAGE1_TOPIC.c_str());
        image_sub_l = new message_filters::Subscriber<sensor_msgs::Image> (n, IMAGE0_TOPIC, 1000, ros::TransportHints().tcpNoDelay(true));
        image_sub_r = new message_filters::Subscriber<sensor_msgs::Image> (n, IMAGE1_TOPIC, 1000, ros::TransportHints().tcpNoDelay(true));
        sync = new message_filters::TimeSynchronizer<sensor_msgs::Image, sensor_msgs::Image> (*image_sub_l, *image_sub_r, 1000);
        sync->registerCallback(boost::bind(&VinsNodeBaseClass::imgs_callback, (VinsNodeBaseClass*)this, _1, _2));
    }
}
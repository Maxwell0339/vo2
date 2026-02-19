#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/TransformStamped.h>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2/LinearMath/Quaternion.h>
#include <cv_bridge/cv_bridge.h>
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/synchronizer.h>

#include <opencv2/core.hpp>
#include <Eigen/Dense>
#include <Eigen/Geometry>

#include <fstream>
#include <iomanip>

#include "vo2/frontend.h"
#include "vo2/backend.h"
#include "vo2/map.h"
#include "vo2/cam_model.h"
#include "vo2/frame.h"

typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::Image> SyncPolicy;

class VO2Node {
public:
    VO2Node(ros::NodeHandle& nh, const std::string& config_file) : nh_(nh) {
        readParameters(config_file);

        // Setup publishers
        odom_pub_ = nh_.advertise<nav_msgs::Odometry>("/vo2/odometry", 100);
        path_pub_ = nh_.advertise<nav_msgs::Path>("/vo2/path", 10);
        tracking_img_pub_ = nh_.advertise<sensor_msgs::Image>("/vo2/tracking_image", 10);

        // Setup stereo image synchronizer
        sub_img0_.subscribe(nh_, image0_topic_, 100);
        sub_img1_.subscribe(nh_, image1_topic_, 100);
        sync_ = std::make_shared<message_filters::Synchronizer<SyncPolicy>>(SyncPolicy(10), sub_img0_, sub_img1_);
        sync_->registerCallback(boost::bind(&VO2Node::imageCallback, this, _1, _2));

        // Open trajectory file
        std::string traj_path;
        nh_.param<std::string>("trajectory_file", traj_path, "vo2_trajectory.txt");
        trajectory_file_.open(traj_path, std::ios::out);
        if (trajectory_file_.is_open()) {
            ROS_INFO("Saving trajectory to: %s", traj_path.c_str());
        }

        path_msg_.header.frame_id = "world";

        ROS_INFO("VO2 node initialized. Waiting for images on %s and %s",
                 image0_topic_.c_str(), image1_topic_.c_str());
    }

    ~VO2Node() {
        if (trajectory_file_.is_open()) {
            trajectory_file_.close();
        }
    }

private:
    void readParameters(const std::string& config_file) {
        cv::FileStorage fs(config_file, cv::FileStorage::READ);
        if (!fs.isOpened()) {
            ROS_ERROR("Failed to open config file: %s", config_file.c_str());
            ros::shutdown();
            return;
        }

        fs["image0_topic"] >> image0_topic_;
        fs["image1_topic"] >> image1_topic_;

        int num_features = static_cast<int>(fs["num_features"]);
        int num_features_tracking_threshold = static_cast<int>(fs["num_features_tracking_threshold"]);
        int num_features_init = static_cast<int>(fs["num_features_init"]);
        float stereo_max_y_diff = static_cast<float>(fs["stereo_max_y_diff"]);
        float stereo_fb_max_error = static_cast<float>(fs["stereo_fb_max_error"]);
        int stereo_lk_win_size = static_cast<int>(fs["stereo_lk_win_size"]);
        int stereo_lk_max_level = static_cast<int>(fs["stereo_lk_max_level"]);

        if (num_features <= 0) num_features = 150;
        if (num_features_tracking_threshold <= 0) num_features_tracking_threshold = 50;
        if (num_features_init <= 0) num_features_init = 40;
        if (stereo_max_y_diff <= 0.0f) stereo_max_y_diff = 20.0f;
        if (stereo_fb_max_error <= 0.0f) stereo_fb_max_error = 2.0f;
        if (stereo_lk_win_size <= 0) stereo_lk_win_size = 31;
        if (stereo_lk_max_level <= 0) stereo_lk_max_level = 4;

        // Read body-camera extrinsics
        cv::Mat body_T_cam0_cv, body_T_cam1_cv;
        fs["body_T_cam0"] >> body_T_cam0_cv;
        fs["body_T_cam1"] >> body_T_cam1_cv;

        Eigen::Matrix4d body_T_cam0 = cvMatToEigen4d(body_T_cam0_cv);
        Eigen::Matrix4d body_T_cam1 = cvMatToEigen4d(body_T_cam1_cv);

        // T_cam0_cam1: transforms cam1 points to cam0 frame
        Eigen::Matrix4d T_cam0_cam1 = body_T_cam0.inverse() * body_T_cam1;
        body_T_cam0_ = body_T_cam0;

        // Read camera calibration paths
        std::string cam0_calib, cam1_calib;
        fs["cam0_calib"] >> cam0_calib;
        fs["cam1_calib"] >> cam1_calib;
        fs.release();

        std::string config_dir = config_file.substr(0, config_file.find_last_of("/"));
        std::string cam0_path = config_dir + "/" + cam0_calib;
        std::string cam1_path = config_dir + "/" + cam1_calib;

        // Load camera models
        CamModel cam0, cam1;
        if (!cam0.loadFromYaml(cam0_path)) {
            ROS_ERROR("Failed to load cam0 calibration: %s", cam0_path.c_str());
            ros::shutdown();
            return;
        }
        if (!cam1.loadFromYaml(cam1_path)) {
            ROS_ERROR("Failed to load cam1 calibration: %s", cam1_path.c_str());
            ros::shutdown();
            return;
        }

        // Create VO components
        map_ = std::make_shared<Map>();
        backend_ = std::make_shared<Backend>();
        frontend_ = std::make_shared<Frontend>();

        frontend_->SetCameras(cam0, cam1);
        frontend_->SetCamExtrinsic(T_cam0_cam1);
        frontend_->SetParams(num_features,
                     num_features_tracking_threshold,
                     num_features_init,
                     stereo_max_y_diff,
                     stereo_fb_max_error,
                     stereo_lk_win_size,
                     stereo_lk_max_level);
        frontend_->SetMap(map_);
        frontend_->SetBackend(backend_);

        cam0_ = cam0;

        ROS_INFO("Stereo extrinsic T_cam0_cam1 computed. Baseline: %.4f m",
                 T_cam0_cam1.block<3, 1>(0, 3).norm());
    }

    void imageCallback(const sensor_msgs::ImageConstPtr& img0_msg,
                       const sensor_msgs::ImageConstPtr& img1_msg) {
        cv::Mat img0, img1;
        try {
            img0 = cv_bridge::toCvCopy(img0_msg, "mono8")->image;
            img1 = cv_bridge::toCvCopy(img1_msg, "mono8")->image;
        } catch (cv_bridge::Exception& e) {
            ROS_ERROR("cv_bridge exception: %s", e.what());
            return;
        }

        // Create frame
        auto frame = Frame::Create();
        frame->img0_ = img0;
        frame->img1_ = img1;
        frame->timestamp_ = img0_msg->header.stamp.toSec();

        // Process frame
        bool ok = frontend_->process(frame);

        if (ok) {
            publishPose(frame->T_w_c_, img0_msg->header.stamp);
            saveTrajectory(frame->T_w_c_, frame->timestamp_);
        }

        // Publish tracking image
        if (tracking_img_pub_.getNumSubscribers() > 0) {
            cv::Mat tracking_img = frontend_->GetTrackingImage();
            if (!tracking_img.empty()) {
                sensor_msgs::ImagePtr msg =
                    cv_bridge::CvImage(img0_msg->header, "bgr8", tracking_img).toImageMsg();
                tracking_img_pub_.publish(msg);
            }
        }
    }

    void publishPose(const Eigen::Matrix4d& T_w_cam0, const ros::Time& stamp) {
        // Convert camera pose to body pose: T_w_body = T_w_cam0 * inv(body_T_cam0)
        Eigen::Matrix4d T_w_body = T_w_cam0 * body_T_cam0_.inverse();

        Eigen::Matrix3d R = T_w_body.block<3, 3>(0, 0);
        Eigen::Vector3d t = T_w_body.block<3, 1>(0, 3);
        Eigen::Quaterniond q(R);

        // Publish Odometry
        nav_msgs::Odometry odom;
        odom.header.stamp = stamp;
        odom.header.frame_id = "world";
        odom.child_frame_id = "body";
        odom.pose.pose.position.x = t.x();
        odom.pose.pose.position.y = t.y();
        odom.pose.pose.position.z = t.z();
        odom.pose.pose.orientation.x = q.x();
        odom.pose.pose.orientation.y = q.y();
        odom.pose.pose.orientation.z = q.z();
        odom.pose.pose.orientation.w = q.w();
        odom_pub_.publish(odom);

        // Publish TF
        geometry_msgs::TransformStamped tf;
        tf.header.stamp = stamp;
        tf.header.frame_id = "world";
        tf.child_frame_id = "body";
        tf.transform.translation.x = t.x();
        tf.transform.translation.y = t.y();
        tf.transform.translation.z = t.z();
        tf.transform.rotation.x = q.x();
        tf.transform.rotation.y = q.y();
        tf.transform.rotation.z = q.z();
        tf.transform.rotation.w = q.w();
        tf_broadcaster_.sendTransform(tf);

        // Publish Path
        geometry_msgs::PoseStamped pose_stamped;
        pose_stamped.header = odom.header;
        pose_stamped.pose = odom.pose.pose;
        path_msg_.header.stamp = stamp;
        path_msg_.poses.push_back(pose_stamped);
        path_pub_.publish(path_msg_);
    }

    void saveTrajectory(const Eigen::Matrix4d& T_w_cam0, double timestamp) {
        if (!trajectory_file_.is_open()) return;

        // Output body pose in TUM format: timestamp tx ty tz qx qy qz qw
        Eigen::Matrix4d T_w_body = T_w_cam0 * body_T_cam0_.inverse();
        Eigen::Matrix3d R = T_w_body.block<3, 3>(0, 0);
        Eigen::Vector3d t = T_w_body.block<3, 1>(0, 3);
        Eigen::Quaterniond q(R);

        trajectory_file_ << std::fixed << std::setprecision(9)
                         << timestamp << " "
                         << t.x() << " " << t.y() << " " << t.z() << " "
                         << q.x() << " " << q.y() << " " << q.z() << " " << q.w()
                         << std::endl;
    }

    static Eigen::Matrix4d cvMatToEigen4d(const cv::Mat& mat) {
        Eigen::Matrix4d eigen_mat = Eigen::Matrix4d::Identity();
        if (mat.rows != 4 || mat.cols != 4) return eigen_mat;
        for (int i = 0; i < 4; i++)
            for (int j = 0; j < 4; j++)
                eigen_mat(i, j) = mat.at<double>(i, j);
        return eigen_mat;
    }

    ros::NodeHandle& nh_;

    // Subscribers & sync
    message_filters::Subscriber<sensor_msgs::Image> sub_img0_, sub_img1_;
    std::shared_ptr<message_filters::Synchronizer<SyncPolicy>> sync_;

    // Publishers
    ros::Publisher odom_pub_;
    ros::Publisher path_pub_;
    ros::Publisher tracking_img_pub_;
    tf2_ros::TransformBroadcaster tf_broadcaster_;

    // VO components
    Frontend::Ptr frontend_;
    Backend::Ptr backend_;
    Map::Ptr map_;
    CamModel cam0_;

    // Extrinsics
    Eigen::Matrix4d body_T_cam0_ = Eigen::Matrix4d::Identity();

    // Trajectory output
    nav_msgs::Path path_msg_;
    std::ofstream trajectory_file_;

    // Config
    std::string image0_topic_, image1_topic_;
};

int main(int argc, char** argv) {
    ros::init(argc, argv, "vo2_node");
    ros::NodeHandle nh("~");

    if (argc < 2) {
        ROS_ERROR("Usage: vo2_node <config_file>");
        return 1;
    }

    std::string config_file = argv[1];
    ROS_INFO("Starting VO2 with config: %s", config_file.c_str());

    VO2Node node(nh, config_file);

    ros::spin();
    return 0;
}
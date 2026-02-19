#ifndef VO2_FRAME_H
#define VO2_FRAME_H

#include <memory>
#include <vector>
#include <mutex>
#include <Eigen/Dense>
#include <opencv2/core.hpp>
#include "vo2/feature.h"

class Frame {
public:
    using Ptr = std::shared_ptr<Frame>;

    Frame() = default;

    static Frame::Ptr Create();
    void SetKeyFrame();

    Eigen::Matrix3d R() const { return T_w_c_.block<3, 3>(0, 0); }
    Eigen::Vector3d t() const { return T_w_c_.block<3, 1>(0, 3); }

    unsigned long id_ = 0;
    unsigned long kf_id_ = 0;
    bool is_kf_ = false;
    double timestamp_ = 0;
    Eigen::Matrix4d T_w_c_ = Eigen::Matrix4d::Identity();
    cv::Mat img0_, img1_;

    std::vector<Feature::Ptr> features_left_;
    std::vector<Feature::Ptr> features_right_;

    std::mutex pose_mutex_;

private:
    static unsigned long next_id_;
    static unsigned long next_kf_id_;
};

#endif // VO2_FRAME_H
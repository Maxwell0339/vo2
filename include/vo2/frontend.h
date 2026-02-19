#ifndef VO2_FRONTEND_H
#define VO2_FRONTEND_H

#include <memory>
#include <vector>
#include <string>
#include <Eigen/Dense>
#include "vo2/frame.h"
#include "vo2/cam_model.h"

class Map;
class Backend;

class Frontend {
public:
    using Ptr = std::shared_ptr<Frontend>;

    enum Status { INIT, TRACKING, LOST };

    Frontend() = default;

    /// Main processing entry: feed a stereo frame
    bool process(Frame::Ptr frame);

    void SetMap(std::shared_ptr<Map> map) { map_ = map; }
    void SetBackend(std::shared_ptr<Backend> backend) { backend_ = backend; }
    void SetCameras(const CamModel& cam0, const CamModel& cam1);
    void SetCamExtrinsic(const Eigen::Matrix4d& T_cam0_cam1);
    void SetParams(int num_features,
                   int num_features_tracking_threshold,
                   int num_features_init,
                   float stereo_max_y_diff,
                   float stereo_fb_max_error,
                   int stereo_lk_win_size,
                   int stereo_lk_max_level);

    Status GetStatus() const { return status_; }

    /// Get the latest tracking image with features drawn (for visualization)
    cv::Mat GetTrackingImage() const;

private:
    bool Init();
    int Track();
    void Reset();

    int DetectFeatures();
    int TrackLastFrame();
    int StereoMatch();
    int TriangulateNewPoints();
    int EstimatePosePnP();
    bool IsKeyFrame();
    void InsertKeyFrame();

    /// DLT triangulation from two bearing vectors
    Eigen::Vector3d Triangulate(const Eigen::Vector3d& b_left,
                                const Eigen::Vector3d& b_right,
                                const Eigen::Matrix3d& R,
                                const Eigen::Vector3d& t);

    Status status_ = INIT;
    Frame::Ptr current_frame_ = nullptr;
    Frame::Ptr last_frame_ = nullptr;
    Frame::Ptr ref_kf_ = nullptr;

    std::shared_ptr<Map> map_ = nullptr;
    std::shared_ptr<Backend> backend_ = nullptr;

    CamModel cam0_, cam1_;
    Eigen::Matrix4d T_cam0_cam1_ = Eigen::Matrix4d::Identity();
    Eigen::Matrix4d relative_motion_ = Eigen::Matrix4d::Identity();

    // Parameters
    int num_features_ = 150;
    int num_features_tracking_threshold_ = 50;
    int num_features_init_ = 50;

    float stereo_max_y_diff_ = 20.0f;
    float stereo_fb_max_error_ = 2.0f;
    int stereo_lk_win_size_ = 31;
    int stereo_lk_max_level_ = 4;
};

#endif // VO2_FRONTEND_H
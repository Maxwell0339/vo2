#include "vo2/frontend.h"
#include "vo2/backend.h"
#include "vo2/map.h"
#include "vo2/mappoint.h"
#include "vo2/feature.h"

#include <opencv2/video/tracking.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <ros/ros.h>

// ---------------------------------------------------------------------------
void Frontend::SetCameras(const CamModel& cam0, const CamModel& cam1) {
    cam0_ = cam0;
    cam1_ = cam1;
}

void Frontend::SetCamExtrinsic(const Eigen::Matrix4d& T_cam0_cam1) {
    T_cam0_cam1_ = T_cam0_cam1;
}

void Frontend::SetParams(int num_features,
                         int num_features_tracking_threshold,
                         int num_features_init,
                         float stereo_max_y_diff,
                         float stereo_fb_max_error,
                         int stereo_lk_win_size,
                         int stereo_lk_max_level) {
    num_features_ = std::max(20, num_features);
    num_features_tracking_threshold_ = std::max(10, num_features_tracking_threshold);
    num_features_init_ = std::max(10, num_features_init);

    stereo_max_y_diff_ = std::max(0.0f, stereo_max_y_diff);
    stereo_fb_max_error_ = std::max(0.0f, stereo_fb_max_error);
    stereo_lk_win_size_ = std::max(11, stereo_lk_win_size);
    if ((stereo_lk_win_size_ % 2) == 0) stereo_lk_win_size_ += 1;
    stereo_lk_max_level_ = std::max(1, stereo_lk_max_level);

    ROS_INFO("Frontend params: num_features=%d track_thr=%d init_thr=%d stereo_max_y_diff=%.1f stereo_fb_max_error=%.2f lk_win=%d lk_level=%d",
             num_features_, num_features_tracking_threshold_, num_features_init_,
             stereo_max_y_diff_, stereo_fb_max_error_, stereo_lk_win_size_, stereo_lk_max_level_);
}

// ---------------------------------------------------------------------------
bool Frontend::process(Frame::Ptr frame) {
    current_frame_ = frame;

    switch (status_) {
        case INIT:
            if (Init()) {
                status_ = TRACKING;
                last_frame_ = current_frame_;
                return true;
            }
            return false;
        case TRACKING: {
            int inliers = Track();
            if (inliers > 0) {
                last_frame_ = current_frame_;
                return true;
            } else {
                status_ = LOST;
                return false;
            }
        }
        case LOST:
            Reset();
            return false;
    }
    return false;
}

// ---------------------------------------------------------------------------
bool Frontend::Init() {
    // Detect features in left image
    int num_detected = DetectFeatures();
    if (num_detected < num_features_init_) {
        ROS_WARN("Init: not enough features detected: %d", num_detected);
        return false;
    }
    //ROS_INFO("Init: detected %d features", num_detected);

    // Stereo match left -> right
    int num_stereo = StereoMatch();
    if (num_stereo < num_features_init_) {
        ROS_WARN("Init: not enough stereo matches: %d", num_stereo);
        return false;
    }
    //ROS_INFO("Init: %d stereo matches", num_stereo);

    // First frame pose is identity (world = first camera frame)
    current_frame_->T_w_c_ = Eigen::Matrix4d::Identity();

    // Triangulate initial 3D map points
    int num_tri = TriangulateNewPoints();
    if (num_tri < num_features_init_ / 2) {
        ROS_WARN("Init: not enough triangulated points: %d", num_tri);
        return false;
    }

    // Insert as first keyframe
    InsertKeyFrame();

    ROS_INFO("VO2 Initialization success: %d features, %d stereo, %d map points",
             num_detected, num_stereo, num_tri);
    return true;
}

// ---------------------------------------------------------------------------
int Frontend::Track() {
    // Motion model prediction
    current_frame_->T_w_c_ = relative_motion_ * last_frame_->T_w_c_;

    // Track features from last frame via LK optical flow
    int num_tracked = TrackLastFrame();
    if (num_tracked < 5) {
        ROS_WARN("Track: too few tracked features: %d", num_tracked);
        return -1;
    }

    // Frontend PnP: initial pose estimate
    int num_pnp = EstimatePosePnP();
    if (num_pnp < 4) {
        ROS_WARN("Track: PnP failed, only %d correspondences", num_pnp);
        return -1;
    }

    // Backend: Ceres nonlinear optimization on reprojection error
    int num_inliers = num_pnp;
    if (backend_) {
        num_inliers = backend_->OptimizePose(current_frame_, cam0_);
        if (num_inliers < 4) {
            ROS_WARN("Track: Backend optimization failed, %d inliers", num_inliers);
            return -1;
        }
    }

    // Update motion model
    relative_motion_ = current_frame_->T_w_c_ * last_frame_->T_w_c_.inverse();

    // Keyframe decision
    if (IsKeyFrame()) {
        InsertKeyFrame();
    }

    return num_inliers;
}

// ---------------------------------------------------------------------------
int Frontend::DetectFeatures() {
    cv::Mat mask(current_frame_->img0_.rows, current_frame_->img0_.cols, CV_8UC1, 255);

    // Mask out existing feature locations
    for (auto& feat : current_frame_->features_left_) {
        if (feat) {
            cv::circle(mask, feat->position_, 20, 0, -1);
        }
    }

    std::vector<cv::KeyPoint> keypoints;
    auto fast = cv::FastFeatureDetector::create(20, true, cv::FastFeatureDetector::TYPE_9_16);
    fast->detect(current_frame_->img0_, keypoints, mask);

    // Sort by response (strongest first)
    std::sort(keypoints.begin(), keypoints.end(),
              [](const cv::KeyPoint& a, const cv::KeyPoint& b) {
                  return a.response > b.response;
              });

    int num_new = 0;
    int max_new = num_features_ - static_cast<int>(current_frame_->features_left_.size());
    for (auto& kp : keypoints) {
        if (num_new >= max_new) break;
        auto feat = std::make_shared<Feature>(kp.pt);
        feat->is_on_left_ = true;
        current_frame_->features_left_.push_back(feat);
        num_new++;
    }

    return static_cast<int>(current_frame_->features_left_.size());
}

// ---------------------------------------------------------------------------
int Frontend::TrackLastFrame() {
    if (!last_frame_ || last_frame_->features_left_.empty()) return 0;

    std::vector<cv::Point2f> pts_last, pts_cur;
    for (auto& feat : last_frame_->features_left_) {
        // Use map point projection as initial guess if available
        auto mp = feat->map_point_.lock();
        if (mp && !mp->is_outlier_) {
            Eigen::Vector3d pw = mp->Position();
            Eigen::Matrix4d T_c_w = current_frame_->T_w_c_.inverse();
            Eigen::Vector3d p_cam = T_c_w.block<3, 3>(0, 0) * pw + T_c_w.block<3, 1>(0, 3);
            Eigen::Vector2d px = cam0_.project(p_cam);
            if (px.x() > 0 && px.x() < cam0_.width_ && px.y() > 0 && px.y() < cam0_.height_) {
                pts_cur.push_back(cv::Point2f(static_cast<float>(px.x()), static_cast<float>(px.y())));
            } else {
                pts_cur.push_back(feat->position_);
            }
        } else {
            pts_cur.push_back(feat->position_);
        }
        pts_last.push_back(feat->position_);
    }

    // Forward LK optical flow
    std::vector<uchar> status;
    std::vector<float> err;
    cv::calcOpticalFlowPyrLK(last_frame_->img0_, current_frame_->img0_,
                              pts_last, pts_cur, status, err,
                              cv::Size(21, 21), 3,
                              cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30, 0.01),
                              cv::OPTFLOW_USE_INITIAL_FLOW);

    // Backward LK for consistency check
    std::vector<cv::Point2f> pts_back;
    std::vector<uchar> status_back;
    std::vector<float> err_back;
    cv::calcOpticalFlowPyrLK(current_frame_->img0_, last_frame_->img0_,
                              pts_cur, pts_back, status_back, err_back,
                              cv::Size(21, 21), 3);

    int num_tracked = 0;
    for (size_t i = 0; i < status.size(); i++) {
        if (!status[i] || !status_back[i]) continue;

        // Forward-backward consistency (2 pixel threshold)
        float dx = pts_last[i].x - pts_back[i].x;
        float dy = pts_last[i].y - pts_back[i].y;
        if (dx * dx + dy * dy > 4.0f) continue;

        // Boundary check
        if (pts_cur[i].x < 0 || pts_cur[i].x >= cam0_.width_ ||
            pts_cur[i].y < 0 || pts_cur[i].y >= cam0_.height_) continue;

        auto feat = std::make_shared<Feature>(pts_cur[i]);
        feat->map_point_ = last_frame_->features_left_[i]->map_point_;
        feat->is_on_left_ = true;
        current_frame_->features_left_.push_back(feat);
        num_tracked++;
    }

    ROS_DEBUG("TrackLastFrame: %d tracked from %lu", num_tracked, last_frame_->features_left_.size());
    return num_tracked;
}

// ---------------------------------------------------------------------------
int Frontend::StereoMatch() {
    std::vector<cv::Point2f> pts_left, pts_right;
    for (auto& feat : current_frame_->features_left_) {
        pts_left.push_back(feat->position_);
        pts_right.push_back(feat->position_);  // initial guess
    }

    if (pts_left.empty()) return 0;

    std::vector<uchar> status;
    std::vector<float> err;
    cv::calcOpticalFlowPyrLK(current_frame_->img0_, current_frame_->img1_,
                              pts_left, pts_right, status, err,
                              cv::Size(stereo_lk_win_size_, stereo_lk_win_size_),
                              stereo_lk_max_level_);

    // Backward LK consistency check (right -> left)
    std::vector<cv::Point2f> pts_back;
    std::vector<uchar> status_back;
    std::vector<float> err_back;
    cv::calcOpticalFlowPyrLK(current_frame_->img1_, current_frame_->img0_,
                              pts_right, pts_back, status_back, err_back,
                              cv::Size(stereo_lk_win_size_, stereo_lk_win_size_),
                              stereo_lk_max_level_);

    int num_matched = 0;
    current_frame_->features_right_.resize(current_frame_->features_left_.size(), nullptr);

    for (size_t i = 0; i < status.size(); i++) {
        if (!status[i] || !status_back[i]) continue;

        float dx_fb = pts_left[i].x - pts_back[i].x;
        float dy_fb = pts_left[i].y - pts_back[i].y;
        if (dx_fb * dx_fb + dy_fb * dy_fb > stereo_fb_max_error_ * stereo_fb_max_error_) continue;

        // Epipolar check: y difference should be small (non-rectified, allow some tolerance)
        float dy = std::abs(pts_left[i].y - pts_right[i].y);
        if (dy > stereo_max_y_diff_) continue;

        // Boundary check
        if (pts_right[i].x < 0 || pts_right[i].x >= cam1_.width_ ||
            pts_right[i].y < 0 || pts_right[i].y >= cam1_.height_) continue;

        auto feat_right = std::make_shared<Feature>(pts_right[i]);
        feat_right->is_on_left_ = false;
        current_frame_->features_right_[i] = feat_right;
        num_matched++;
    }

    ROS_DEBUG("StereoMatch: %d matches from %lu features", num_matched, pts_left.size());
    return num_matched;
}

// ---------------------------------------------------------------------------
int Frontend::EstimatePosePnP() {
    // Collect 3D-2D correspondences for features with valid map points
    std::vector<cv::Point3f> pts_3d;
    std::vector<cv::Point2f> pts_2d_norm;  // normalized coordinates

    for (size_t i = 0; i < current_frame_->features_left_.size(); i++) {
        auto& feat = current_frame_->features_left_[i];
        if (!feat || feat->is_outlier_) continue;
        auto mp = feat->map_point_.lock();
        if (!mp || mp->is_outlier_) continue;

        Eigen::Vector3d pw = mp->Position();
        pts_3d.push_back(cv::Point3f(static_cast<float>(pw.x()),
                                      static_cast<float>(pw.y()),
                                      static_cast<float>(pw.z())));

        // Unproject pixel to bearing, then get normalized coords for solvePnP
        Eigen::Vector3d bearing = cam0_.unproject(
            Eigen::Vector2d(feat->position_.x, feat->position_.y));
        pts_2d_norm.push_back(cv::Point2f(
            static_cast<float>(bearing.x() / bearing.z()),
            static_cast<float>(bearing.y() / bearing.z())));
    }

    if (pts_3d.size() < 4) return 0;

    // Identity K since we pass normalized coordinates
    cv::Mat K = cv::Mat::eye(3, 3, CV_64F);
    cv::Mat dist_coeffs = cv::Mat::zeros(4, 1, CV_64F);

    // Initial guess from predicted pose (T_c_w)
    Eigen::Matrix4d T_c_w = current_frame_->T_w_c_.inverse();
    Eigen::Matrix3d R_e = T_c_w.block<3, 3>(0, 0);
    Eigen::Vector3d t_e = T_c_w.block<3, 1>(0, 3);

    Eigen::AngleAxisd aa(R_e);
    Eigen::Vector3d aa_vec = aa.angle() * aa.axis();
    cv::Mat rvec = (cv::Mat_<double>(3, 1) << aa_vec.x(), aa_vec.y(), aa_vec.z());
    cv::Mat tvec = (cv::Mat_<double>(3, 1) << t_e.x(), t_e.y(), t_e.z());

    bool success = cv::solvePnP(pts_3d, pts_2d_norm, K, dist_coeffs, rvec, tvec,
                                 true, cv::SOLVEPNP_ITERATIVE);

    if (!success) {
        ROS_WARN("EstimatePosePnP: solvePnP failed");
        return 0;
    }

    // Convert result back to Eigen T_w_c
    Eigen::Vector3d rvec_e(rvec.at<double>(0), rvec.at<double>(1), rvec.at<double>(2));
    double angle = rvec_e.norm();
    Eigen::Matrix3d R_opt;
    if (angle < 1e-8) {
        R_opt = Eigen::Matrix3d::Identity();
    } else {
        R_opt = Eigen::AngleAxisd(angle, rvec_e / angle).toRotationMatrix();
    }
    Eigen::Vector3d t_opt(tvec.at<double>(0), tvec.at<double>(1), tvec.at<double>(2));

    Eigen::Matrix4d T_c_w_opt = Eigen::Matrix4d::Identity();
    T_c_w_opt.block<3, 3>(0, 0) = R_opt;
    T_c_w_opt.block<3, 1>(0, 3) = t_opt;

    current_frame_->T_w_c_ = T_c_w_opt.inverse();

    ROS_DEBUG("EstimatePosePnP: success with %lu points", pts_3d.size());
    return static_cast<int>(pts_3d.size());
}

// ---------------------------------------------------------------------------
int Frontend::TriangulateNewPoints() {
    // T_c1_c0: transforms points from cam0 frame to cam1 frame
    Eigen::Matrix4d T_c1_c0 = T_cam0_cam1_.inverse();
    Eigen::Matrix3d R_rl = T_c1_c0.block<3, 3>(0, 0);
    Eigen::Vector3d t_rl = T_c1_c0.block<3, 1>(0, 3);

    int num_triangulated = 0;

    for (size_t i = 0; i < current_frame_->features_left_.size(); i++) {
        // Only triangulate features without existing map points
        if (!current_frame_->features_left_[i]->map_point_.expired()) continue;
        if (i >= current_frame_->features_right_.size() ||
            !current_frame_->features_right_[i]) continue;

        Eigen::Vector3d b_left = cam0_.unproject(
            Eigen::Vector2d(current_frame_->features_left_[i]->position_.x,
                            current_frame_->features_left_[i]->position_.y));
        Eigen::Vector3d b_right = cam1_.unproject(
            Eigen::Vector2d(current_frame_->features_right_[i]->position_.x,
                            current_frame_->features_right_[i]->position_.y));

        Eigen::Vector3d pt_cam0 = Triangulate(b_left, b_right, R_rl, t_rl);

        // Validate: positive depth
        if (pt_cam0.z() < 0.1 || pt_cam0.z() > 200.0) continue;

        // Validate: reprojection error in left image
        Eigen::Vector2d reproj = cam0_.project(pt_cam0);
        double err = (reproj - Eigen::Vector2d(current_frame_->features_left_[i]->position_.x,
                                                current_frame_->features_left_[i]->position_.y)).norm();
        if (err > 3.0) continue;

        // Transform to world frame
        Eigen::Vector3d pt_world = current_frame_->R() * pt_cam0 + current_frame_->t();

        auto mp = MapPoint::Create(pt_world);
        mp->AddObservation(current_frame_->features_left_[i]);
        mp->AddObservation(current_frame_->features_right_[i]);
        current_frame_->features_left_[i]->map_point_ = mp;
        current_frame_->features_right_[i]->map_point_ = mp;

        map_->InsertMapPoint(mp);
        num_triangulated++;
    }

    ROS_DEBUG("TriangulateNewPoints: %d new points", num_triangulated);
    return num_triangulated;
}

// ---------------------------------------------------------------------------
Eigen::Vector3d Frontend::Triangulate(const Eigen::Vector3d& b_left,
                                       const Eigen::Vector3d& b_right,
                                       const Eigen::Matrix3d& R,
                                       const Eigen::Vector3d& t) {
    // DLT triangulation
    // Left camera: P1 = [I | 0] (cam0 frame)
    // Right camera: P2 = [R | t] (cam0 → cam1)
    double xl = b_left.x() / b_left.z();
    double yl = b_left.y() / b_left.z();
    double xr = b_right.x() / b_right.z();
    double yr = b_right.y() / b_right.z();

    Eigen::Matrix<double, 3, 4> P1, P2;
    P1 << 1, 0, 0, 0,
          0, 1, 0, 0,
          0, 0, 1, 0;
    P2.block<3, 3>(0, 0) = R;
    P2.col(3) = t;

    Eigen::Matrix4d A;
    A.row(0) = xl * P1.row(2) - P1.row(0);
    A.row(1) = yl * P1.row(2) - P1.row(1);
    A.row(2) = xr * P2.row(2) - P2.row(0);
    A.row(3) = yr * P2.row(2) - P2.row(1);

    Eigen::JacobiSVD<Eigen::Matrix4d> svd(A, Eigen::ComputeFullV);
    Eigen::Vector4d X_h = svd.matrixV().col(3);

    return X_h.head<3>() / X_h(3);
}

// ---------------------------------------------------------------------------
bool Frontend::IsKeyFrame() {
    int num_tracked_mp = 0;
    for (auto& feat : current_frame_->features_left_) {
        if (feat && !feat->is_outlier_ && !feat->map_point_.expired()) {
            num_tracked_mp++;
        }
    }
    return num_tracked_mp < num_features_tracking_threshold_;
}

// ---------------------------------------------------------------------------
void Frontend::InsertKeyFrame() {
    current_frame_->SetKeyFrame();
    map_->InsertKeyFrame(current_frame_);
    ref_kf_ = current_frame_;

    // Detect new features to fill up
    DetectFeatures();

    // Stereo match new features
    StereoMatch();

    // Triangulate new map points
    int num_new = TriangulateNewPoints();
    ROS_INFO("New keyframe %lu with %d new map points (total left features: %lu)",
             current_frame_->kf_id_, num_new, current_frame_->features_left_.size());
}

// ---------------------------------------------------------------------------
void Frontend::Reset() {
    ROS_WARN("VO2 tracking lost, resetting to INIT...");
    status_ = INIT;
    current_frame_ = nullptr;
    last_frame_ = nullptr;
    ref_kf_ = nullptr;
}

// ---------------------------------------------------------------------------
cv::Mat Frontend::GetTrackingImage() const {
    cv::Mat img;
    if (current_frame_ && !current_frame_->img0_.empty()) {
        cv::cvtColor(current_frame_->img0_, img, cv::COLOR_GRAY2BGR);

        for (auto& feat : current_frame_->features_left_) {
            if (!feat) continue;
            auto mp = feat->map_point_.lock();
            if (mp && !feat->is_outlier_) {
                // Tracked feature with map point: green
                cv::circle(img, feat->position_, 3, cv::Scalar(0, 255, 0), -1);
            } else {
                // Feature without map point: red
                cv::circle(img, feat->position_, 3, cv::Scalar(0, 0, 255), -1);
            }
        }
    }
    return img;
}
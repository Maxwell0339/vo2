#ifndef VO2_CAM_MODEL_H
#define VO2_CAM_MODEL_H

#include <string>
#include <cmath>
#include <opencv2/core.hpp>
#include <Eigen/Dense>

class CamModel {
public:
    CamModel() = default;

    bool loadFromYaml(const std::string& calib_file);

    /// Project a 3D point (in camera frame) to pixel coordinates using MEI model
    Eigen::Vector2d project(const Eigen::Vector3d& P_cam) const;

    /// Unproject pixel coordinates to a unit bearing vector (in camera frame)
    Eigen::Vector3d unproject(const Eigen::Vector2d& px) const;

    /// Convenience: project and return cv::Point2f
    cv::Point2f projectCv(const Eigen::Vector3d& P_cam) const;

    int height_ = 0;
    int width_ = 0;
    double k1_ = 0, k2_ = 0, p1_ = 0, p2_ = 0;
    double gamma1_ = 0, gamma2_ = 0;
    double u0_ = 0, v0_ = 0;
    double xi_ = 0;
};

#endif // VO2_CAM_MODEL_H
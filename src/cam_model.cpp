#include "vo2/cam_model.h"
#include <opencv2/core.hpp>
#include <iostream>

bool CamModel::loadFromYaml(const std::string& calib_file) {
    cv::FileStorage fs(calib_file, cv::FileStorage::READ);
    if (!fs.isOpened()) {
        std::cerr << "Failed to open calibration file: " << calib_file << std::endl;
        return false;
    }

    fs["image_height"] >> height_;
    fs["image_width"] >> width_;

    cv::FileNode mirror = fs["mirror_parameters"];
    mirror["xi"] >> xi_;

    cv::FileNode dist = fs["distortion_parameters"];
    dist["k1"] >> k1_;
    dist["k2"] >> k2_;
    dist["p1"] >> p1_;
    dist["p2"] >> p2_;

    cv::FileNode proj = fs["projection_parameters"];
    proj["gamma1"] >> gamma1_;
    proj["gamma2"] >> gamma2_;
    proj["u0"] >> u0_;
    proj["v0"] >> v0_;

    std::cout << "Loaded camera: " << width_ << "x" << height_
              << " xi=" << xi_ << " gamma=(" << gamma1_ << "," << gamma2_ << ")"
              << " pp=(" << u0_ << "," << v0_ << ")" << std::endl;

    fs.release();
    return true;
}

Eigen::Vector2d CamModel::project(const Eigen::Vector3d& P_cam) const {
    double x = P_cam.x(), y = P_cam.y(), z = P_cam.z();
    double norm_p = std::sqrt(x * x + y * y + z * z);
    double d = z + xi_ * norm_p;

    if (std::abs(d) < 1e-8) {
        return Eigen::Vector2d(-1, -1);
    }

    double mx = x / d;
    double my = y / d;

    // Apply distortion (radial + tangential)
    double r2 = mx * mx + my * my;
    double r4 = r2 * r2;
    double radial = 1.0 + k1_ * r2 + k2_ * r4;
    double mx_d = radial * mx + 2.0 * p1_ * mx * my + p2_ * (r2 + 2.0 * mx * mx);
    double my_d = radial * my + p1_ * (r2 + 2.0 * my * my) + 2.0 * p2_ * mx * my;

    double u = gamma1_ * mx_d + u0_;
    double v = gamma2_ * my_d + v0_;

    return Eigen::Vector2d(u, v);
}

Eigen::Vector3d CamModel::unproject(const Eigen::Vector2d& px) const {
    // Step 1: normalize pixel to MEI projection plane
    double mx = (px.x() - u0_) / gamma1_;
    double my = (px.y() - v0_) / gamma2_;

    // Step 2: iterative undistortion (fixed-point iteration, ~10 rounds)
    double mx_u = mx, my_u = my;
    for (int i = 0; i < 10; i++) {
        double r2 = mx_u * mx_u + my_u * my_u;
        double r4 = r2 * r2;
        double radial = 1.0 + k1_ * r2 + k2_ * r4;
        double dx = 2.0 * p1_ * mx_u * my_u + p2_ * (r2 + 2.0 * mx_u * mx_u);
        double dy = p1_ * (r2 + 2.0 * my_u * my_u) + 2.0 * p2_ * mx_u * my_u;
        mx_u = (mx - dx) / radial;
        my_u = (my - dy) / radial;
    }

    // Step 3: lift to unit sphere using MEI model
    double r2 = mx_u * mx_u + my_u * my_u;
    double disc = 1.0 + (1.0 - xi_ * xi_) * r2;
    if (disc < 0) {
        return Eigen::Vector3d(0, 0, 1);  // invalid, return forward direction
    }

    double alpha = (xi_ + std::sqrt(disc)) / (1.0 + r2);

    Eigen::Vector3d P(alpha * mx_u, alpha * my_u, alpha - xi_);
    return P.normalized();
}

cv::Point2f CamModel::projectCv(const Eigen::Vector3d& P_cam) const {
    Eigen::Vector2d px = project(P_cam);
    return cv::Point2f(static_cast<float>(px.x()), static_cast<float>(px.y()));
}
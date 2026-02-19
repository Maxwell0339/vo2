#include "vo2/backend.h"
#include "vo2/mappoint.h"
#include "vo2/feature.h"
#include <ros/ros.h>

int Backend::OptimizePose(Frame::Ptr frame, const CamModel& cam) {
    // Extract T_c_w from frame (camera from world)
    Eigen::Matrix4d T_c_w = frame->T_w_c_.inverse();
    Eigen::Matrix3d R_cw = T_c_w.block<3, 3>(0, 0);
    Eigen::Vector3d t_cw = T_c_w.block<3, 1>(0, 3);

    // Convert rotation to angle-axis
    Eigen::AngleAxisd aa(R_cw);
    double pose[6];
    Eigen::Vector3d axis_angle = aa.angle() * aa.axis();
    pose[0] = axis_angle.x();
    pose[1] = axis_angle.y();
    pose[2] = axis_angle.z();
    pose[3] = t_cw.x();
    pose[4] = t_cw.y();
    pose[5] = t_cw.z();

    // Collect 2D-3D correspondences
    struct Observation {
        int idx;
        double u, v;
        Eigen::Vector3d pw;
    };
    std::vector<Observation> observations;

    for (size_t i = 0; i < frame->features_left_.size(); i++) {
        auto& feat = frame->features_left_[i];
        if (feat == nullptr || feat->is_outlier_) continue;
        auto mp = feat->map_point_.lock();
        if (mp == nullptr || mp->is_outlier_) continue;

        Observation obs;
        obs.idx = i;
        obs.u = feat->position_.x;
        obs.v = feat->position_.y;
        obs.pw = mp->Position();
        observations.push_back(obs);
    }

    if (observations.size() < 4) {
        ROS_WARN("Backend: too few observations (%lu)", observations.size());
        return 0;
    }

    // === Round 1: optimize with all observations ===
    ceres::Problem problem1;
    for (auto& obs : observations) {
        ceres::CostFunction* cost = new ceres::AutoDiffCostFunction<ReprojectionCost, 2, 6>(
            new ReprojectionCost(obs.u, obs.v, obs.pw.x(), obs.pw.y(), obs.pw.z(), cam));
        problem1.AddResidualBlock(cost, new ceres::HuberLoss(1.0), pose);
    }

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    options.max_num_iterations = 10;
    options.minimizer_progress_to_stdout = false;

    ceres::Solver::Summary summary1;
    ceres::Solve(options, &problem1, &summary1);

    // Evaluate reprojection errors and mark outliers
    // Compute T_c_w from current pose estimate
    Eigen::Vector3d aa_vec(pose[0], pose[1], pose[2]);
    double angle = aa_vec.norm();
    Eigen::Matrix3d R_opt;
    if (angle < 1e-8) {
        R_opt = Eigen::Matrix3d::Identity();
    } else {
        R_opt = Eigen::AngleAxisd(angle, aa_vec / angle).toRotationMatrix();
    }
    Eigen::Vector3d t_opt(pose[3], pose[4], pose[5]);

    const double chi2_threshold = 5.991;  // 95% chi-square with 2 DOF
    int num_outliers = 0;
    std::vector<bool> is_outlier(observations.size(), false);

    for (size_t i = 0; i < observations.size(); i++) {
        Eigen::Vector3d p_cam = R_opt * observations[i].pw + t_opt;
        Eigen::Vector2d proj = cam.project(p_cam);
        double err_u = proj.x() - observations[i].u;
        double err_v = proj.y() - observations[i].v;
        double err2 = err_u * err_u + err_v * err_v;

        if (err2 > chi2_threshold) {
            is_outlier[i] = true;
            frame->features_left_[observations[i].idx]->is_outlier_ = true;
            num_outliers++;
        }
    }

    // === Round 2: optimize without outliers ===
    ceres::Problem problem2;
    int num_inliers = 0;
    for (size_t i = 0; i < observations.size(); i++) {
        if (is_outlier[i]) continue;
        auto& obs = observations[i];
        ceres::CostFunction* cost = new ceres::AutoDiffCostFunction<ReprojectionCost, 2, 6>(
            new ReprojectionCost(obs.u, obs.v, obs.pw.x(), obs.pw.y(), obs.pw.z(), cam));
        problem2.AddResidualBlock(cost, new ceres::HuberLoss(1.0), pose);
        num_inliers++;
    }

    if (num_inliers < 4) {
        ROS_WARN("Backend: too few inliers after outlier rejection (%d)", num_inliers);
        return num_inliers;
    }

    ceres::Solver::Summary summary2;
    ceres::Solve(options, &problem2, &summary2);

    // Convert optimized pose back to T_w_c
    Eigen::Vector3d aa_final(pose[0], pose[1], pose[2]);
    double angle_final = aa_final.norm();
    Eigen::Matrix3d R_final;
    if (angle_final < 1e-8) {
        R_final = Eigen::Matrix3d::Identity();
    } else {
        R_final = Eigen::AngleAxisd(angle_final, aa_final / angle_final).toRotationMatrix();
    }
    Eigen::Vector3d t_final(pose[3], pose[4], pose[5]);

    Eigen::Matrix4d T_c_w_final = Eigen::Matrix4d::Identity();
    T_c_w_final.block<3, 3>(0, 0) = R_final;
    T_c_w_final.block<3, 1>(0, 3) = t_final;

    frame->T_w_c_ = T_c_w_final.inverse();

    ROS_DEBUG("Backend: %d inliers, %d outliers, cost %.4f -> %.4f",
              num_inliers, num_outliers,
              summary1.initial_cost, summary2.final_cost);

    return num_inliers;
}

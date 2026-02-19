#ifndef VO2_BACKEND_H
#define VO2_BACKEND_H

#include <memory>
#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include "vo2/frame.h"
#include "vo2/cam_model.h"

class Backend {
public:
    using Ptr = std::shared_ptr<Backend>;

    Backend() = default;

    /// Optimize the pose of the given frame using Ceres (minimize reprojection error)
    /// Returns the number of inlier features after optimization
    int OptimizePose(Frame::Ptr frame, const CamModel& cam);

    /// Ceres AutoDiff cost functor for MEI reprojection error
    struct ReprojectionCost {
        // Observation (pixel coordinates)
        double u_obs, v_obs;
        // 3D map point in world frame
        double X, Y, Z;
        // MEI camera parameters
        double xi, gamma1, gamma2, u0, v0;
        double k1, k2, p1, p2;

        ReprojectionCost(double u, double v,
                         double X, double Y, double Z,
                         const CamModel& cam)
            : u_obs(u), v_obs(v), X(X), Y(Y), Z(Z),
              xi(cam.xi_), gamma1(cam.gamma1_), gamma2(cam.gamma2_),
              u0(cam.u0_), v0(cam.v0_),
              k1(cam.k1_), k2(cam.k2_), p1(cam.p1_), p2(cam.p2_) {}

        template <typename T>
        bool operator()(const T* pose, T* residual) const {
            // pose[0..2]: angle-axis rotation (T_c_w)
            // pose[3..5]: translation (T_c_w)

            // Transform world point to camera frame
            T p_w[3] = {T(X), T(Y), T(Z)};
            T p_cam[3];
            ceres::AngleAxisRotatePoint(pose, p_w, p_cam);
            p_cam[0] += pose[3];
            p_cam[1] += pose[4];
            p_cam[2] += pose[5];

            // MEI projection
            T x = p_cam[0], y = p_cam[1], z = p_cam[2];
            T norm_p = ceres::sqrt(x * x + y * y + z * z);
            T d = z + T(xi) * norm_p;

            // Avoid division by zero
            if (d < T(1e-8)) {
                residual[0] = T(0);
                residual[1] = T(0);
                return true;
            }

            T mx = x / d;
            T my = y / d;

            // Apply distortion
            T r2 = mx * mx + my * my;
            T r4 = r2 * r2;
            T radial = T(1.0) + T(k1) * r2 + T(k2) * r4;
            T mx_d = radial * mx + T(2.0) * T(p1) * mx * my + T(p2) * (r2 + T(2.0) * mx * mx);
            T my_d = radial * my + T(p1) * (r2 + T(2.0) * my * my) + T(2.0) * T(p2) * mx * my;

            // Project to pixel
            T u_proj = T(gamma1) * mx_d + T(u0);
            T v_proj = T(gamma2) * my_d + T(v0);

            residual[0] = u_proj - T(u_obs);
            residual[1] = v_proj - T(v_obs);

            return true;
        }
    };
};

#endif // VO2_BACKEND_H

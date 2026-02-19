#ifndef VO2_FEATURE_H
#define VO2_FEATURE_H

#include <memory>
#include <opencv2/core.hpp>

class Frame;
class MapPoint;

class Feature {
public:
    using Ptr = std::shared_ptr<Feature>;

    Feature() = default;
    Feature(const cv::Point2f& pos) : position_(pos) {}

    cv::Point2f position_;
    std::weak_ptr<Frame> frame_;
    std::weak_ptr<MapPoint> map_point_;
    bool is_on_left_ = true;
    bool is_outlier_ = false;
};

#endif // VO2_FEATURE_H
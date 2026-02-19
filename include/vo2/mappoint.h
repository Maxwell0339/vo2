#ifndef VO2_MAPPOINT_H
#define VO2_MAPPOINT_H

#include <memory>
#include <list>
#include <mutex>
#include <Eigen/Dense>

class Feature;

class MapPoint {
public:
    using Ptr = std::shared_ptr<MapPoint>;

    MapPoint() = default;
    MapPoint(const Eigen::Vector3d& pos) : position_(pos) {}

    static MapPoint::Ptr Create();
    static MapPoint::Ptr Create(const Eigen::Vector3d& pos);

    void AddObservation(std::shared_ptr<Feature> feat);
    void RemoveObservation(std::shared_ptr<Feature> feat);
    std::list<std::weak_ptr<Feature>> GetObservations();

    Eigen::Vector3d Position() {
        std::lock_guard<std::mutex> lock(data_mutex_);
        return position_;
    }
    void SetPosition(const Eigen::Vector3d& pos) {
        std::lock_guard<std::mutex> lock(data_mutex_);
        position_ = pos;
    }

    unsigned long id_ = 0;
    int observed_times_ = 0;
    bool is_outlier_ = false;

private:
    Eigen::Vector3d position_ = Eigen::Vector3d::Zero();
    std::list<std::weak_ptr<Feature>> observations_;
    std::mutex data_mutex_;
    static unsigned long next_id_;
};

#endif // VO2_MAPPOINT_H

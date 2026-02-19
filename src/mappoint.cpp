#include "vo2/mappoint.h"
#include "vo2/feature.h"

unsigned long MapPoint::next_id_ = 0;

MapPoint::Ptr MapPoint::Create() {
    auto mp = std::make_shared<MapPoint>();
    mp->id_ = next_id_++;
    return mp;
}

MapPoint::Ptr MapPoint::Create(const Eigen::Vector3d& pos) {
    auto mp = std::make_shared<MapPoint>(pos);
    mp->id_ = next_id_++;
    return mp;
}

void MapPoint::AddObservation(std::shared_ptr<Feature> feat) {
    std::lock_guard<std::mutex> lock(data_mutex_);
    observations_.push_back(feat);
    observed_times_++;
}

void MapPoint::RemoveObservation(std::shared_ptr<Feature> feat) {
    std::lock_guard<std::mutex> lock(data_mutex_);
    for (auto it = observations_.begin(); it != observations_.end(); ++it) {
        if (it->lock() == feat) {
            observations_.erase(it);
            observed_times_--;
            break;
        }
    }
}

std::list<std::weak_ptr<Feature>> MapPoint::GetObservations() {
    std::lock_guard<std::mutex> lock(data_mutex_);
    return observations_;
}

#include "vo2/map.h"
#include "vo2/feature.h"
#include <ros/ros.h>

void Map::InsertKeyFrame(Frame::Ptr frame) {
    std::lock_guard<std::mutex> lock(data_mutex_);
    current_frame_ = frame;
    if (keyframes_.find(frame->kf_id_) == keyframes_.end()) {
        keyframes_.insert({frame->kf_id_, frame});
        active_keyframes_.insert({frame->kf_id_, frame});
    } else {
        keyframes_[frame->kf_id_] = frame;
        active_keyframes_[frame->kf_id_] = frame;
    }

    if (static_cast<int>(active_keyframes_.size()) > num_active_keyframes_) {
        RemoveOldKeyframe();
    }
}

void Map::InsertMapPoint(MapPoint::Ptr mp) {
    std::lock_guard<std::mutex> lock(data_mutex_);
    if (landmarks_.find(mp->id_) == landmarks_.end()) {
        landmarks_.insert({mp->id_, mp});
        active_landmarks_.insert({mp->id_, mp});
    } else {
        landmarks_[mp->id_] = mp;
        active_landmarks_[mp->id_] = mp;
    }
}

void Map::RemoveOldKeyframe() {
    if (current_frame_ == nullptr) return;

    // find the closest and farthest keyframe from current frame
    double max_dist = 0, min_dist = 9999;
    unsigned long max_kf_id = 0, min_kf_id = 0;

    Eigen::Vector3d current_t = current_frame_->t();
    for (auto& kf : active_keyframes_) {
        if (kf.second == current_frame_) continue;
        double dist = (kf.second->t() - current_t).norm();
        if (dist > max_dist) {
            max_dist = dist;
            max_kf_id = kf.first;
        }
        if (dist < min_dist) {
            min_dist = dist;
            min_kf_id = kf.first;
        }
    }

    // remove the closest if distance < threshold, otherwise remove farthest
    Frame::Ptr frame_to_remove = nullptr;
    if (min_dist < 0.2) {
        frame_to_remove = active_keyframes_.at(min_kf_id);
    } else {
        frame_to_remove = active_keyframes_.at(max_kf_id);
    }

    // remove keyframe and its associated active landmarks
    active_keyframes_.erase(frame_to_remove->kf_id_);
    for (auto& feat : frame_to_remove->features_left_) {
        if (feat == nullptr) continue;
        auto mp = feat->map_point_.lock();
        if (mp) {
            mp->RemoveObservation(feat);
        }
    }
    for (auto& feat : frame_to_remove->features_right_) {
        if (feat == nullptr) continue;
        auto mp = feat->map_point_.lock();
        if (mp) {
            mp->RemoveObservation(feat);
        }
    }

    CleanMap();
}

void Map::CleanMap() {
    int removed = 0;
    for (auto it = active_landmarks_.begin(); it != active_landmarks_.end();) {
        if (it->second->observed_times_ <= 0) {
            it = active_landmarks_.erase(it);
            removed++;
        } else {
            ++it;
        }
    }
    ROS_DEBUG("Removed %d active landmarks", removed);
}

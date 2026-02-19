#ifndef VO2_MAP_H
#define VO2_MAP_H

#include <memory>
#include <unordered_map>
#include <mutex>
#include "vo2/frame.h"
#include "vo2/mappoint.h"

class Map {
public:
    using Ptr = std::shared_ptr<Map>;
    using KeyFramesType = std::unordered_map<unsigned long, Frame::Ptr>;
    using LandmarksType = std::unordered_map<unsigned long, MapPoint::Ptr>;

    Map() = default;

    void InsertKeyFrame(Frame::Ptr frame);
    void InsertMapPoint(MapPoint::Ptr mp);

    KeyFramesType GetAllKeyFrames() {
        std::lock_guard<std::mutex> lock(data_mutex_);
        return keyframes_;
    }
    KeyFramesType GetActiveKeyFrames() {
        std::lock_guard<std::mutex> lock(data_mutex_);
        return active_keyframes_;
    }
    LandmarksType GetAllMapPoints() {
        std::lock_guard<std::mutex> lock(data_mutex_);
        return landmarks_;
    }
    LandmarksType GetActiveMapPoints() {
        std::lock_guard<std::mutex> lock(data_mutex_);
        return active_landmarks_;
    }

    void CleanMap();

private:
    void RemoveOldKeyframe();

    KeyFramesType keyframes_;
    KeyFramesType active_keyframes_;
    LandmarksType landmarks_;
    LandmarksType active_landmarks_;

    Frame::Ptr current_frame_ = nullptr;
    int num_active_keyframes_ = 7;

    std::mutex data_mutex_;
};

#endif // VO2_MAP_H

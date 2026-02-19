#include "vo2/frame.h"

unsigned long Frame::next_id_ = 0;
unsigned long Frame::next_kf_id_ = 0;

Frame::Ptr Frame::Create() {
    auto frame = std::make_shared<Frame>();
    frame->id_ = next_id_++;
    return frame;
}

void Frame::SetKeyFrame() {
    is_kf_ = true;
    kf_id_ = next_kf_id_++;
}

# A simple stereo VO

## Prerequisites
- ROS Noetic
- OpenCV4
- Eigen3
- Ceres Solver 1.14

## Building
```bash
mkdir -p ~/catkin_ws/src
git clone https://github.com/Maxwell0339/vo2.git
cd ~/catkin_ws
catkin_make
source devel/setup.bash
```

## Running
```bash
roslaunch vo2 euroc.launch
```

Now it has some running effienciency issues, with some lagging in the visualization. I will try to optimize it in the future.

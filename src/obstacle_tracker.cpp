#include "obstacle_tracker/obstacle_tracker_node.hpp"
#include <rclcpp/rclcpp.hpp>
#include <memory>

int main(int argc, char ** argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<obstacle_tracker::ObstacleTrackerNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}

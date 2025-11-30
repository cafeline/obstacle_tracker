#ifndef OBSTACLE_TRACKER_NODE_HPP_
#define OBSTACLE_TRACKER_NODE_HPP_

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/laser_scan.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <nav_msgs/msg/occupancy_grid.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <vector>
#include <string>
#include <optional>
#include <array>

namespace obstacle_tracker
{

struct Point2D
{
  double x{0.0};
  double y{0.0};
};

struct Cluster
{
  std::vector<Point2D> points;
  Point2D centroid;
  double length{0.0};
  double width{0.0};
};

class ObstacleTrackerNode : public rclcpp::Node
{
public:
  ObstacleTrackerNode();

  // exposed for tests
  std::vector<Point2D> scanToPoints(const sensor_msgs::msg::LaserScan & scan) const;
  std::vector<Point2D> transformToMap(
    const std::vector<Point2D> & points,
    const std::string & frame, const rclcpp::Time & stamp, bool & success) const;
  std::vector<Cluster> segmentByRangeJump(const std::vector<Point2D> & points) const;
  visualization_msgs::msg::MarkerArray buildCubes(
    const std::vector<Cluster> & clusters,
    bool dynamic, const rclcpp::Time & stamp, const std::string & frame_id) const;
  visualization_msgs::msg::MarkerArray buildOutlines(
    const std::vector<Cluster> & clusters,
    bool dynamic, const rclcpp::Time & stamp, const std::string & frame_id) const;
  nav_msgs::msg::OccupancyGrid buildOccupancyMask(
    const std::vector<Point2D> & points,
    const std::vector<Cluster> & clusters,
    const rclcpp::Time & stamp,
    const std::string & frame_id,
    const std::optional<Point2D> & sensor_origin) const;
  std::vector<Point2D> computeHull(const std::vector<Point2D> & pts) const;
  std::array<Point2D, 4> computeObb(const std::vector<Point2D> & hull) const;

private:
  void laserScanCallback(const sensor_msgs::msg::LaserScan::SharedPtr msg);

  // ROS
  rclcpp::Subscription<sensor_msgs::msg::LaserScan>::SharedPtr scan_sub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr obstacles_pub_;
  rclcpp::Publisher<nav_msgs::msg::OccupancyGrid>::SharedPtr mask_pub_;
  std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
  std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
  mutable std::optional<geometry_msgs::msg::TransformStamped> last_tf_;

  // Parameters
  std::string scan_topic_;
  std::string obstacles_topic_;
  std::string mask_topic_;
  std::string target_frame_;
  double tf_timeout_sec_{};
  double processing_range_{};
  double range_jump_ratio_{};
  double range_jump_min_{};
  int min_cluster_points_{};
  double range_gap_abs_{};
};

}  // namespace obstacle_tracker

#endif  // OBSTACLE_TRACKER_NODE_HPP_

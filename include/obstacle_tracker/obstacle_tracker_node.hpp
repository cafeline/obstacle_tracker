#ifndef OBSTACLE_TRACKER_NODE_HPP_
#define OBSTACLE_TRACKER_NODE_HPP_

#include <memory>
#include <vector>
#include <deque>
#include <string>
#include <Eigen/Dense>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/laser_scan.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

namespace obstacle_tracker
{

struct Point3D
{
  double x;
  double y;
  double z;
  double angle;
  rclcpp::Time stamp;
  std::string frame_id;

  Point3D(
    double px = 0.0, double py = 0.0, double pz = 0.0, double a = 0.0,
    const std::string & frame = "map")
  : x(px), y(py), z(pz), angle(a), stamp(), frame_id(frame) {}
};

struct ClusterShape
{
  double length{0.0};
  double width{0.0};
};

struct Cluster
{
  std::vector<Point3D> points;
  Point3D centroid;
  Point3D velocity;
  Point3D smoothed_velocity;
  ClusterShape shape;
  double speed{0.0};
  double confidence{0.0};
  double average_range{0.0};
  int track_count{0};
  int id{-1};
  bool is_dynamic{false};
  bool is_confirmed{false};
  std::string frame_id{"map"};

  Cluster()
  : centroid(), velocity(), smoothed_velocity(), shape(), speed(0.0), confidence(0.0),
    average_range(0.0), track_count(0), id(-1), is_dynamic(false), is_confirmed(false),
    frame_id("map") {}
};

struct TrackManagerConfig
{
  double process_noise{0.5};
  double measurement_noise{0.1};
  double gating_distance{1.0};
  int max_missed_frames{3};
  int min_consecutive_hits{2};
  double confidence_increment{0.2};
  double confidence_decrement{0.1};
};

class TrackManager
{
public:
  explicit TrackManager(const TrackManagerConfig & config = TrackManagerConfig());

  std::vector<Cluster> update(const std::vector<Cluster> & detections, double dt);
  void reset();
  size_t activeTrackCount() const;

private:
  struct TrackInternal
  {
    int id{0};
    Eigen::Vector4d state{Eigen::Vector4d::Zero()};
    Eigen::Matrix4d covariance{Eigen::Matrix4d::Identity()};
    int total_hits{0};
    int consecutive_misses{0};
    double confidence{0.0};
    Cluster last_cluster;
  };

  TrackManagerConfig config_;
  int next_id_;
  std::vector<TrackInternal> tracks_;
};

class ObstacleTrackerNode : public rclcpp::Node
{
public:
  ObstacleTrackerNode();

private:
  void laserScanCallback(const sensor_msgs::msg::LaserScan::SharedPtr msg);

  std::vector<Point3D> convertScanToPoints(const sensor_msgs::msg::LaserScan & scan) const;
  std::vector<Point3D> filterByRange(const std::vector<Point3D> & points) const;
  std::vector<Point3D> transformPointsToMap(const std::vector<Point3D> & points);
  std::vector<Point3D> voxelize(const std::vector<Point3D> & points) const;
  std::vector<Cluster> clusterPoints(const std::vector<Point3D> & points) const;
  void classifyClusters(std::vector<Cluster> & clusters) const;
  void publishClusters(const std::vector<Cluster> & clusters, const rclcpp::Time & stamp);
  void ensureClustersInMap(std::vector<Cluster> & clusters);

  double computeDeltaTime(const rclcpp::Time & stamp);

  visualization_msgs::msg::MarkerArray buildMarkerArray(
    const std::vector<Cluster> & clusters,
    bool dynamic,
    const rclcpp::Time & stamp);
  Point3D transformPointToMap(const Point3D & point) const;

  // Parameters
  double robot_processing_range_;
  double voxel_size_;
  double clustering_base_radius_;
  double clustering_far_gain_;
  int min_cluster_points_;
  double processing_frequency_;
  double dynamic_cluster_speed_threshold_;
  bool enable_sliding_window_;
  int sliding_window_size_;
  int history_frame_count_;

  TrackManagerConfig tracker_config_;
  TrackManager track_manager_;

  // ROS resources
  rclcpp::Subscription<sensor_msgs::msg::LaserScan>::SharedPtr scan_subscriber_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr dynamic_publisher_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr static_publisher_;
  std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
  std::shared_ptr<tf2_ros::TransformListener> tf_listener_;

  // Internal states
  rclcpp::Time last_scan_stamp_;
  std::deque<std::vector<Point3D>> sliding_points_;
  std::deque<visualization_msgs::msg::MarkerArray> dynamic_history_;
  std::deque<visualization_msgs::msg::MarkerArray> static_history_;
  mutable rclcpp::Time last_tf_warning_time_;
};

}  // namespace obstacle_tracker

#endif  // OBSTACLE_TRACKER_NODE_HPP_

#include "obstacle_tracker/obstacle_tracker_node.hpp"

#include <Eigen/Dense>
#include <algorithm>
#include <functional>
#include <array>
#include <Eigen/Eigenvalues>
#include <cmath>
#include <limits>
#include <unordered_map>
#include <utility>

namespace obstacle_tracker
{

namespace
{
constexpr double kDefaultDt = 0.1;

inline double clampNonNegative(double value)
{
  return value < 0.0 ? 0.0 : value;
}

std::string makeVoxelKey(int vx, int vy, int vz)
{
  return std::to_string(vx) + ":" + std::to_string(vy) + ":" + std::to_string(vz);
}

}  // namespace

TrackManager::TrackManager(const TrackManagerConfig & config)
: config_(config), next_id_(1), tracks_()
{
}

std::vector<Cluster> TrackManager::update(const std::vector<Cluster> & detections, double dt)
{
  double delta_t = dt > 0.0 ? dt : kDefaultDt;
  Eigen::Matrix4d F = Eigen::Matrix4d::Identity();
  F(0, 2) = delta_t;
  F(1, 3) = delta_t;

  Eigen::Matrix4d Q = Eigen::Matrix4d::Zero();
  double q = std::max(1e-4, config_.process_noise);
  Q(0, 0) = q;
  Q(1, 1) = q;
  Q(2, 2) = q;
  Q(3, 3) = q;

  Eigen::Matrix<double, 2, 4> H = Eigen::Matrix<double, 2, 4>::Zero();
  H(0, 0) = 1.0;
  H(1, 1) = 1.0;

  Eigen::Matrix2d R = Eigen::Matrix2d::Identity() * std::max(1e-4, config_.measurement_noise);

  // 1. 予測ステップ
  for (auto & track : tracks_) {
    track.state = F * track.state;
    track.covariance = F * track.covariance * F.transpose() + Q;
  }

  std::vector<int> assignment(tracks_.size(), -1);
  std::vector<bool> detection_used(detections.size(), false);

  // 2. 簡易ゲーティング + 最近傍対応
  for (size_t track_idx = 0; track_idx < tracks_.size(); ++track_idx) {
    auto & track = tracks_[track_idx];
    double best_distance = std::numeric_limits<double>::max();
    int best_detection = -1;

    for (size_t det_idx = 0; det_idx < detections.size(); ++det_idx) {
      if (detection_used[det_idx]) {
        continue;
      }

      const auto & detection = detections[det_idx];
      double dx = detection.centroid.x - track.state(0);
      double dy = detection.centroid.y - track.state(1);
      double distance = std::hypot(dx, dy);

      if (distance <= config_.gating_distance && distance < best_distance) {
        best_distance = distance;
        best_detection = static_cast<int>(det_idx);
      }
    }

    if (best_detection >= 0) {
      assignment[track_idx] = best_detection;
      detection_used[best_detection] = true;
    }
  }

  std::vector<Cluster> output;
  output.reserve(detections.size());

  auto I = Eigen::Matrix4d::Identity();

  // 3. 更新ステップ
  for (size_t track_idx = 0; track_idx < tracks_.size(); ++track_idx) {
    auto & track = tracks_[track_idx];
    int matched_idx = assignment[track_idx];

    if (matched_idx >= 0) {
      const Cluster & detection = detections[matched_idx];
      Eigen::Vector2d z(detection.centroid.x, detection.centroid.y);
      Eigen::Vector2d y = z - H * track.state;
      Eigen::Matrix2d S = H * track.covariance * H.transpose() + R;
      Eigen::Matrix<double, 4, 2> K = track.covariance * H.transpose() * S.inverse();

      track.state = track.state + K * y;
      track.covariance = (I - K * H) * track.covariance;
      track.total_hits++;
      track.consecutive_misses = 0;
      track.confidence = std::min(1.0, track.confidence + config_.confidence_increment);

      track.last_cluster = detection;
      track.last_cluster.id = track.id;
      track.last_cluster.track_count = track.total_hits;
      track.last_cluster.velocity = Point3D(track.state(2), track.state(3), 0.0);
      track.last_cluster.smoothed_velocity = track.last_cluster.velocity;
      track.last_cluster.speed = std::hypot(track.state(2), track.state(3));
      track.last_cluster.confidence = track.confidence;
      track.last_cluster.is_confirmed = (track.total_hits >= config_.min_consecutive_hits);

      output.push_back(track.last_cluster);
    } else {
      track.consecutive_misses++;
      track.confidence = std::max(0.0, track.confidence - config_.confidence_decrement);
    }
  }

  // 4. 未対応検出から新規トラックを生成
  for (size_t det_idx = 0; det_idx < detections.size(); ++det_idx) {
    if (detection_used[det_idx]) {
      continue;
    }
    TrackInternal new_track;
    new_track.id = next_id_++;
    new_track.state << detections[det_idx].centroid.x, detections[det_idx].centroid.y, 0.0, 0.0;
    new_track.covariance = Eigen::Matrix4d::Identity() * config_.measurement_noise;
    new_track.total_hits = 1;
    new_track.consecutive_misses = 0;
    new_track.confidence = config_.confidence_increment;
    new_track.last_cluster = detections[det_idx];
    new_track.last_cluster.id = new_track.id;
    new_track.last_cluster.track_count = 1;
    new_track.last_cluster.velocity = Point3D(0.0, 0.0, 0.0);
    new_track.last_cluster.smoothed_velocity = Point3D(0.0, 0.0, 0.0);
    new_track.last_cluster.speed = 0.0;
    new_track.last_cluster.confidence = new_track.confidence;
    new_track.last_cluster.is_confirmed = (config_.min_consecutive_hits <= 1);

    output.push_back(new_track.last_cluster);
    tracks_.push_back(new_track);
  }

  // 5. 長期間観測されないトラックを削除
  tracks_.erase(
    std::remove_if(
      tracks_.begin(), tracks_.end(), [this](const TrackInternal & track) {
        return track.consecutive_misses > config_.max_missed_frames;
      }),
    tracks_.end());

  return output;
}

void TrackManager::reset()
{
  tracks_.clear();
  next_id_ = 1;
}

size_t TrackManager::activeTrackCount() const
{
  return tracks_.size();
}

ObstacleTrackerNode::ObstacleTrackerNode()
: Node("obstacle_tracker_node"),
  robot_processing_range_(10.0),
  voxel_size_(0.2),
  clustering_base_radius_(0.3),
  clustering_far_gain_(0.03),
  min_cluster_points_(2),
  processing_frequency_(15.0),
  dynamic_cluster_speed_threshold_(0.5),
  enable_sliding_window_(true),
  sliding_window_size_(3),
  history_frame_count_(3),
  tracker_config_(),
  track_manager_(tracker_config_),
  last_scan_stamp_()
{
  this->declare_parameter("robot_processing_range", robot_processing_range_);
  this->declare_parameter("dynamic_cluster_speed_threshold", dynamic_cluster_speed_threshold_);
  this->declare_parameter("voxel_size", voxel_size_);
  this->declare_parameter("clustering_max_distance", clustering_base_radius_);
  this->declare_parameter("clustering_far_distance_gain", clustering_far_gain_);
  this->declare_parameter("min_cluster_points", min_cluster_points_);
  this->declare_parameter("processing_frequency", processing_frequency_);
  this->declare_parameter("enable_sliding_window", enable_sliding_window_);
  this->declare_parameter("sliding_window_size", sliding_window_size_);
  this->declare_parameter("history_frame_count", history_frame_count_);

  this->declare_parameter("tracker_process_noise", tracker_config_.process_noise);
  this->declare_parameter("tracker_measurement_noise", tracker_config_.measurement_noise);
  this->declare_parameter("tracker_gating_distance", tracker_config_.gating_distance);
  this->declare_parameter("tracker_max_missed_frames", tracker_config_.max_missed_frames);
  this->declare_parameter("tracker_min_consecutive_hits", tracker_config_.min_consecutive_hits);
  this->declare_parameter("tracker_confidence_increment", tracker_config_.confidence_increment);
  this->declare_parameter("tracker_confidence_decrement", tracker_config_.confidence_decrement);

  robot_processing_range_ = this->get_parameter("robot_processing_range").as_double();
  dynamic_cluster_speed_threshold_ =
    this->get_parameter("dynamic_cluster_speed_threshold").as_double();
  voxel_size_ = this->get_parameter("voxel_size").as_double();
  clustering_base_radius_ = this->get_parameter("clustering_max_distance").as_double();
  clustering_far_gain_ = this->get_parameter("clustering_far_distance_gain").as_double();
  min_cluster_points_ = this->get_parameter("min_cluster_points").as_int();
  processing_frequency_ = this->get_parameter("processing_frequency").as_double();
  enable_sliding_window_ = this->get_parameter("enable_sliding_window").as_bool();
  sliding_window_size_ = this->get_parameter("sliding_window_size").as_int();
  history_frame_count_ = this->get_parameter("history_frame_count").as_int();

  tracker_config_.process_noise = this->get_parameter("tracker_process_noise").as_double();
  tracker_config_.measurement_noise = this->get_parameter("tracker_measurement_noise").as_double();
  tracker_config_.gating_distance = this->get_parameter("tracker_gating_distance").as_double();
  tracker_config_.max_missed_frames = this->get_parameter("tracker_max_missed_frames").as_int();
  tracker_config_.min_consecutive_hits =
    this->get_parameter("tracker_min_consecutive_hits").as_int();
  tracker_config_.confidence_increment =
    this->get_parameter("tracker_confidence_increment").as_double();
  tracker_config_.confidence_decrement =
    this->get_parameter("tracker_confidence_decrement").as_double();

  track_manager_ = TrackManager(tracker_config_);

  tf_buffer_ = std::make_shared<tf2_ros::Buffer>(this->get_clock());
  tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

  scan_subscriber_ = this->create_subscription<sensor_msgs::msg::LaserScan>(
    "/scan", rclcpp::SensorDataQoS(),
    std::bind(&ObstacleTrackerNode::laserScanCallback, this, std::placeholders::_1));

  dynamic_publisher_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(
    "/dynamic_obstacles", 10);
  static_publisher_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(
    "/static_obstacles", 10);
  debug_dynamic_publisher_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(
    "/dynamic_obstacles_debug", 10);
  debug_static_publisher_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(
    "/static_obstacles_debug", 10);

  RCLCPP_INFO(
    this->get_logger(), "Obstacle tracker initialized: range=%.2f, voxel=%.2f, min_pts=%d",
    robot_processing_range_, voxel_size_, min_cluster_points_);
}

void ObstacleTrackerNode::laserScanCallback(const sensor_msgs::msg::LaserScan::SharedPtr msg)
{
  rclcpp::Time scan_stamp(msg->header.stamp);

  if (processing_frequency_ > 0.0 && last_scan_stamp_.nanoseconds() > 0) {
    double dt = (scan_stamp - last_scan_stamp_).seconds();
    if (dt > 0.0) {
      double min_interval = 1.0 / processing_frequency_;
      if (dt < min_interval * 0.5) {
        return;
      }
    }
  }

  auto raw_points = convertScanToPoints(*msg);
  auto ranged_points = filterByRange(raw_points);
  auto map_points = transformPointsToMap(ranged_points);
  auto voxel_points = voxelize(map_points);

  std::vector<Point3D> aggregated_points;
  if (enable_sliding_window_) {
    sliding_points_.push_back(voxel_points);
    while (static_cast<int>(sliding_points_.size()) > std::max(1, sliding_window_size_)) {
      sliding_points_.pop_front();
    }
    size_t total = 0;
    for (const auto & frame : sliding_points_) {
      total += frame.size();
    }
    aggregated_points.reserve(total);
    for (const auto & frame : sliding_points_) {
      aggregated_points.insert(aggregated_points.end(), frame.begin(), frame.end());
    }
  } else {
    aggregated_points = voxel_points;
  }

  auto clusters = clusterPoints(aggregated_points);
  ensureClustersInMap(clusters);
  double dt = computeDeltaTime(scan_stamp);
  auto tracked_clusters = track_manager_.update(clusters, dt);
  classifyClusters(tracked_clusters);
  publishClusters(tracked_clusters, scan_stamp);
}

std::vector<Point3D> ObstacleTrackerNode::convertScanToPoints(
  const sensor_msgs::msg::LaserScan & scan) const
{
  std::vector<Point3D> points;
  if (scan.ranges.empty()) {
    return points;
  }

  points.reserve(scan.ranges.size());
  rclcpp::Time stamp(scan.header.stamp);
  double angle = scan.angle_min;

  for (size_t i = 0; i < scan.ranges.size(); ++i) {
    double range = scan.ranges[i];
    if (!std::isfinite(range) || range <= scan.range_min || range >= scan.range_max) {
      angle += scan.angle_increment;
      continue;
    }
    Point3D point(range * std::cos(angle), range * std::sin(angle), 0.0, angle,
      scan.header.frame_id);
    point.stamp = stamp;
    points.push_back(point);
    angle += scan.angle_increment;
  }

  return points;
}

std::vector<Point3D> ObstacleTrackerNode::filterByRange(const std::vector<Point3D> & points) const
{
  if (robot_processing_range_ <= 0.0) {
    return points;
  }
  std::vector<Point3D> filtered;
  filtered.reserve(points.size());
  double max_distance = robot_processing_range_;

  for (const auto & point : points) {
    double distance = std::hypot(point.x, point.y);
    if (distance <= max_distance) {
      filtered.push_back(point);
    }
  }
  return filtered;
}

std::vector<Point3D> ObstacleTrackerNode::transformPointsToMap(
  const std::vector<Point3D> & points)
{
  if (points.empty()) {
    return points;
  }

  std::vector<Point3D> transformed;
  transformed.reserve(points.size());
  for (const auto & point : points) {
    transformed.push_back(transformPointToMap(point));
  }

  return transformed;
}

std::vector<Point3D> ObstacleTrackerNode::voxelize(const std::vector<Point3D> & points) const
{
  if (points.empty() || voxel_size_ <= 0.0) {
    return points;
  }

  std::unordered_map<std::string, std::pair<Point3D, int>> voxel_map;
  voxel_map.reserve(points.size());
  double inv_size = 1.0 / voxel_size_;

  for (const auto & point : points) {
    int vx = static_cast<int>(std::floor(point.x * inv_size));
    int vy = static_cast<int>(std::floor(point.y * inv_size));
    int vz = static_cast<int>(std::floor(point.z * inv_size));
    auto key = makeVoxelKey(vx, vy, vz);
    auto it = voxel_map.find(key);
    if (it == voxel_map.end()) {
      voxel_map.emplace(key, std::make_pair(point, 1));
    } else {
      auto & entry = it->second;
      entry.second += 1;
      double n = static_cast<double>(entry.second);
      entry.first.x = entry.first.x + (point.x - entry.first.x) / n;
      entry.first.y = entry.first.y + (point.y - entry.first.y) / n;
      entry.first.z = entry.first.z + (point.z - entry.first.z) / n;
      entry.first.angle = point.angle;
      entry.first.stamp = point.stamp;
      entry.first.frame_id = point.frame_id;
    }
  }

  std::vector<Point3D> voxelized;
  voxelized.reserve(voxel_map.size());
  for (const auto & [key, value] : voxel_map) {
    voxelized.push_back(value.first);
  }

  return voxelized;
}

std::vector<Cluster> ObstacleTrackerNode::clusterPoints(const std::vector<Point3D> & points) const
{
  std::vector<Cluster> clusters;
  if (points.empty()) {
    return clusters;
  }

  const size_t n = points.size();
  std::vector<int> labels(n, -1);
  int cluster_label = 0;

  auto adaptiveRadius = [this](const Point3D & pt) {
      double range = std::hypot(pt.x, pt.y);
      return std::max(0.05, clustering_base_radius_ + clustering_far_gain_ * range);
    };

  auto adaptiveMinPoints = [this](const Point3D & pt) {
      double range = std::hypot(pt.x, pt.y);
      double factor = robot_processing_range_ > 0.0 ? range / robot_processing_range_ : 0.0;
      factor = std::clamp(factor, 0.0, 1.0);
      int dynamic_min = min_cluster_points_ + static_cast<int>(std::round(2.0 * factor));
      return std::max(min_cluster_points_, dynamic_min);
    };

  auto regionQuery = [&](size_t idx) {
      std::vector<size_t> neighbors;
      double radius = adaptiveRadius(points[idx]);
      double radius_sq = radius * radius;
      for (size_t j = 0; j < n; ++j) {
        if (j == idx) {
          continue;
        }
        double dx = points[idx].x - points[j].x;
        double dy = points[idx].y - points[j].y;
        double dz = points[idx].z - points[j].z;
        double distance_sq = dx * dx + dy * dy + dz * dz;
        if (distance_sq <= radius_sq) {
          neighbors.push_back(j);
        }
      }
      return neighbors;
    };

  std::vector<bool> visited(n, false);

  for (size_t i = 0; i < n; ++i) {
    if (visited[i]) {
      continue;
    }
    visited[i] = true;
    auto neighbors = regionQuery(i);
    if (static_cast<int>(neighbors.size()) + 1 < adaptiveMinPoints(points[i])) {
      continue;
    }

    Cluster cluster;
    std::vector<size_t> seeds = neighbors;
    cluster.points.push_back(points[i]);
    cluster.frame_id = cluster.points.front().frame_id;
    labels[i] = cluster_label;

    while (!seeds.empty()) {
      size_t current_idx = seeds.back();
      seeds.pop_back();

      if (!visited[current_idx]) {
        visited[current_idx] = true;
        auto current_neighbors = regionQuery(current_idx);
        if (static_cast<int>(current_neighbors.size()) + 1 >=
          adaptiveMinPoints(points[current_idx]))
        {
          seeds.insert(seeds.end(), current_neighbors.begin(), current_neighbors.end());
        }
      }

      if (labels[current_idx] == -1) {
        labels[current_idx] = cluster_label;
        cluster.points.push_back(points[current_idx]);
      }
    }

    if (static_cast<int>(cluster.points.size()) >= min_cluster_points_) {
      double sum_x = 0.0;
      double sum_y = 0.0;
      double sum_z = 0.0;
      double sum_range = 0.0;
      double min_x = std::numeric_limits<double>::max();
      double max_x = -std::numeric_limits<double>::max();
      double min_y = std::numeric_limits<double>::max();
      double max_y = -std::numeric_limits<double>::max();

      for (const auto & pt : cluster.points) {
        sum_x += pt.x;
        sum_y += pt.y;
        sum_z += pt.z;
        sum_range += std::hypot(pt.x, pt.y);
        min_x = std::min(min_x, pt.x);
        max_x = std::max(max_x, pt.x);
        min_y = std::min(min_y, pt.y);
        max_y = std::max(max_y, pt.y);
      }

      double inv = 1.0 / static_cast<double>(cluster.points.size());
      cluster.centroid.x = sum_x * inv;
      cluster.centroid.y = sum_y * inv;
      cluster.centroid.z = sum_z * inv;
      cluster.centroid.stamp = cluster.points.front().stamp;
      cluster.centroid.frame_id = cluster.frame_id;
      cluster.average_range = sum_range * inv;
      cluster.shape.length = std::max(voxel_size_, max_x - min_x);
      cluster.shape.width = std::max(voxel_size_, max_y - min_y);
      cluster.id = cluster_label;
      clusters.push_back(cluster);
      cluster_label++;
    }
  }

  return clusters;
}

void ObstacleTrackerNode::ensureClustersInMap(std::vector<Cluster> & clusters)
{
  for (auto & cluster : clusters) {
    for (auto & point : cluster.points) {
      point = transformPointToMap(point);
    }
    cluster.centroid = transformPointToMap(cluster.centroid);
    cluster.frame_id = cluster.centroid.frame_id;
  }
}

void ObstacleTrackerNode::classifyClusters(std::vector<Cluster> & clusters) const
{
  for (auto & cluster : clusters) {
    double speed = std::hypot(cluster.smoothed_velocity.x, cluster.smoothed_velocity.y);
    cluster.speed = speed;
    cluster.is_dynamic = speed > dynamic_cluster_speed_threshold_;
  }
}

visualization_msgs::msg::MarkerArray ObstacleTrackerNode::buildMarkerArray(
  const std::vector<Cluster> & clusters,
  bool dynamic,
  const rclcpp::Time & stamp)
{
  visualization_msgs::msg::MarkerArray array;
  int marker_id = 1;
  for (const auto & cluster : clusters) {
    Point3D centroid_map = transformPointToMap(cluster.centroid);
    if (centroid_map.frame_id != "map") {
      RCLCPP_WARN(
        this->get_logger(),
        "クラスタ重心をmap座標へ変換できませんでした (frame=%s)",
        cluster.centroid.frame_id.c_str());
      continue;
    }

    visualization_msgs::msg::Marker marker;
    marker.header.frame_id = "map";
    marker.header.stamp = stamp;
    marker.ns = dynamic ? "dynamic_obstacles" : "static_obstacles";
    marker.id = marker_id++;
    marker.type = visualization_msgs::msg::Marker::LINE_LIST;
    marker.action = visualization_msgs::msg::Marker::ADD;
    marker.pose.orientation.w = 1.0;
    marker.scale.x = 0.05;
    if (dynamic) {
      marker.color.r = 1.0;
      marker.color.g = 0.2;
      marker.color.b = 0.2;
    } else {
      marker.color.r = 0.2;
      marker.color.g = 0.2;
      marker.color.b = 1.0;
    }
    marker.color.a = 1.0;
    marker.lifetime = rclcpp::Duration::from_seconds(0.5);

    if (cluster.points.empty()) {
      continue;
    }

    Eigen::Vector2d axis_long(1.0, 0.0);
    Eigen::Vector2d axis_short(0.0, 1.0);
    if (cluster.points.size() >= 2) {
      Eigen::Matrix2d cov = Eigen::Matrix2d::Zero();
      for (const auto & pt : cluster.points) {
        double dx = pt.x - centroid_map.x;
        double dy = pt.y - centroid_map.y;
        cov(0, 0) += dx * dx;
        cov(0, 1) += dx * dy;
        cov(1, 0) += dx * dy;
        cov(1, 1) += dy * dy;
      }
      cov /= static_cast<double>(cluster.points.size());
      Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> solver(cov);
      if (solver.info() == Eigen::Success) {
        axis_long = solver.eigenvectors().col(1).normalized();
        axis_short = Eigen::Vector2d(-axis_long.y(), axis_long.x());
      }
    }

    double min_long = std::numeric_limits<double>::max();
    double max_long = -std::numeric_limits<double>::max();
    double min_short = std::numeric_limits<double>::max();
    double max_short = -std::numeric_limits<double>::max();
    for (const auto & pt : cluster.points) {
      Eigen::Vector2d diff(pt.x - centroid_map.x, pt.y - centroid_map.y);
      double proj_long = diff.dot(axis_long);
      double proj_short = diff.dot(axis_short);
      min_long = std::min(min_long, proj_long);
      max_long = std::max(max_long, proj_long);
      min_short = std::min(min_short, proj_short);
      max_short = std::max(max_short, proj_short);
    }

    if (!std::isfinite(min_long) || !std::isfinite(max_long) ||
      !std::isfinite(min_short) || !std::isfinite(max_short))
    {
      continue;
    }

    double length = max_long - min_long;
    if (length < 2.0 * voxel_size_) {
      double delta = (2.0 * voxel_size_ - length) * 0.5;
      min_long -= delta;
      max_long += delta;
    }

    double width = max_short - min_short;
    if (width < 2.0 * voxel_size_) {
      double delta = (2.0 * voxel_size_ - width) * 0.5;
      min_short -= delta;
      max_short += delta;
    }

    std::array<geometry_msgs::msg::Point, 4> corners{};
    std::array<Eigen::Vector2d, 4> projections = {
      Eigen::Vector2d(min_long, min_short),
      Eigen::Vector2d(max_long, min_short),
      Eigen::Vector2d(max_long, max_short),
      Eigen::Vector2d(min_long, max_short)};

    for (size_t i = 0; i < 4; ++i) {
      Eigen::Vector2d world =
        Eigen::Vector2d(centroid_map.x, centroid_map.y) +
        axis_long * projections[i].x() +
        axis_short * projections[i].y();
      corners[i].x = world.x();
      corners[i].y = world.y();
      corners[i].z = centroid_map.z;
    }

    for (int i = 0; i < 4; ++i) {
      marker.points.push_back(corners[i]);
      marker.points.push_back(corners[(i + 1) % 4]);
    }

    array.markers.push_back(marker);
  }
  return array;
}

visualization_msgs::msg::MarkerArray ObstacleTrackerNode::buildCubeListMarkerArray(
  const std::vector<Cluster> & clusters,
  bool dynamic,
  const rclcpp::Time & stamp)
{
  visualization_msgs::msg::MarkerArray array;
  int marker_id = 1;

  for (const auto & cluster : clusters) {
    Point3D centroid_map = transformPointToMap(cluster.centroid);
    if (centroid_map.frame_id != "map") {
      continue;
    }

    visualization_msgs::msg::Marker marker;
    marker.header.frame_id = "map";
    marker.header.stamp = stamp;
    marker.ns = dynamic ? "dynamic_obstacles" : "static_obstacles";
    marker.id = marker_id++;
    marker.type = visualization_msgs::msg::Marker::CUBE;
    marker.action = visualization_msgs::msg::Marker::ADD;
    marker.pose.position.x = centroid_map.x;
    marker.pose.position.y = centroid_map.y;
    marker.pose.position.z = centroid_map.z;
    marker.pose.orientation.w = 1.0;
    marker.scale.x = std::max(voxel_size_, cluster.shape.length);
    marker.scale.y = std::max(voxel_size_, cluster.shape.width);
    marker.scale.z = voxel_size_;
    if (dynamic) {
      marker.color.r = 1.0;
      marker.color.g = 0.2;
      marker.color.b = 0.2;
    } else {
      marker.color.r = 0.2;
      marker.color.g = 0.2;
      marker.color.b = 1.0;
    }
    marker.color.a = 1.0;
    marker.lifetime = rclcpp::Duration::from_seconds(0.5);
    array.markers.push_back(marker);
  }

  return array;
}

Point3D ObstacleTrackerNode::transformPointToMap(const Point3D & point) const
{
  if (point.frame_id == "map") {
    return point;
  }

  Point3D transformed_point = point;
  rclcpp::Time lookup_stamp = point.stamp;
  if (lookup_stamp.nanoseconds() == 0 && last_scan_stamp_.nanoseconds() > 0) {
    lookup_stamp = last_scan_stamp_;
  }

  if (!tf_buffer_->canTransform(
      "map", point.frame_id, tf2::TimePointZero, tf2::durationFromSec(0.0)))
  {
    std::string frames = tf_buffer_->allFramesAsString();
    if (frames.find("map") == std::string::npos) {
      transformed_point.frame_id = "map";
      return transformed_point;
    }

    auto now_time = this->now();
    if (last_tf_warning_time_.nanoseconds() == 0 ||
      (now_time - last_tf_warning_time_).seconds() > 1.0)
    {
      RCLCPP_WARN(
        this->get_logger(),
        "TF変換に失敗しました (%s -> map): 変換が利用できません",
        point.frame_id.c_str());
      last_tf_warning_time_ = now_time;
    }
    return point;
  }

  try {
    auto transform = tf_buffer_->lookupTransform(
      "map", point.frame_id, tf2::TimePointZero, tf2::durationFromSec(0.2));
    geometry_msgs::msg::PointStamped in, out;
    in.header.frame_id = point.frame_id;
    in.header.stamp = lookup_stamp;
    in.point.x = point.x;
    in.point.y = point.y;
    in.point.z = point.z;
    tf2::doTransform(in, out, transform);

    transformed_point.x = out.point.x;
    transformed_point.y = out.point.y;
    transformed_point.z = out.point.z;
    transformed_point.frame_id = "map";
    transformed_point.stamp = lookup_stamp;
  } catch (const tf2::TransformException & ex) {
    auto now_time = this->now();
    if (last_tf_warning_time_.nanoseconds() == 0 ||
      (now_time - last_tf_warning_time_).seconds() > 1.0)
    {
      RCLCPP_WARN(
        this->get_logger(),
        "TF変換に失敗しました (%s -> map): %s", point.frame_id.c_str(), ex.what());
      last_tf_warning_time_ = now_time;
    }
    transformed_point.frame_id = point.frame_id;
  }

  return transformed_point;
}

void ObstacleTrackerNode::publishClusters(
  const std::vector<Cluster> & clusters,
  const rclcpp::Time & stamp)
{
  std::vector<Cluster> dynamic_clusters;
  std::vector<Cluster> static_clusters;

  for (const auto & cluster : clusters) {
    if (cluster.is_dynamic) {
      dynamic_clusters.push_back(cluster);
    } else {
      static_clusters.push_back(cluster);
    }
  }

  auto dynamic_markers = buildCubeListMarkerArray(dynamic_clusters, true, stamp);
  auto static_markers = buildCubeListMarkerArray(static_clusters, false, stamp);

  dynamic_history_.push_back(dynamic_markers);
  static_history_.push_back(static_markers);
  while (static_cast<int>(dynamic_history_.size()) > std::max(1, history_frame_count_)) {
    dynamic_history_.pop_front();
  }
  while (static_cast<int>(static_history_.size()) > std::max(1, history_frame_count_)) {
    static_history_.pop_front();
  }

  auto buildOutput = [&](const std::deque<visualization_msgs::msg::MarkerArray> & history,
      const std::string & ns) {
      visualization_msgs::msg::MarkerArray output;
      visualization_msgs::msg::Marker clear;
      clear.header.frame_id = "map";
      clear.header.stamp = stamp;
      clear.ns = ns;
      clear.id = 0;
      clear.action = visualization_msgs::msg::Marker::DELETEALL;
      output.markers.push_back(clear);

      int id = 1;
      for (const auto & frame : history) {
        for (const auto & marker : frame.markers) {
          visualization_msgs::msg::Marker copy = marker;
          copy.id = id++;
          output.markers.push_back(copy);
        }
      }
      return output;
    };

  dynamic_publisher_->publish(buildOutput(dynamic_history_, "dynamic_obstacles"));
  static_publisher_->publish(buildOutput(static_history_, "static_obstacles"));

  auto debug_dynamic = buildMarkerArray(dynamic_clusters, true, stamp);
  auto debug_static = buildMarkerArray(static_clusters, false, stamp);
  debug_dynamic_publisher_->publish(debug_dynamic);
  debug_static_publisher_->publish(debug_static);
}

double ObstacleTrackerNode::computeDeltaTime(const rclcpp::Time & stamp)
{
  double dt = kDefaultDt;
  if (last_scan_stamp_.nanoseconds() > 0) {
    double diff = (stamp - last_scan_stamp_).seconds();
    if (diff > 0.0) {
      dt = diff;
    }
  }
  last_scan_stamp_ = stamp;
  return dt;
}

}  // namespace obstacle_tracker

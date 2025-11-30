#include "obstacle_tracker/obstacle_tracker_node.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace obstacle_tracker
{

ObstacleTrackerNode::ObstacleTrackerNode()
: Node("obstacle_tracker_node")
{
  // Parameters (mimic pointcloud2_cutter style: declare and log)
  scan_topic_ = declare_parameter<std::string>("scan_topic", "/scan");
  obstacles_topic_ = declare_parameter<std::string>("obstacles_topic", "/obstacles");
  mask_topic_ = declare_parameter<std::string>("mask_topic", "/obstacle_mask");
  target_frame_ = declare_parameter<std::string>("target_frame", "map");
  tf_timeout_sec_ = declare_parameter<double>("tf_timeout_sec", 0.05);
  processing_range_ = declare_parameter<double>("processing_range", 10.0);
  range_jump_ratio_ = declare_parameter<double>("range_jump_ratio", 0.1);
  range_jump_min_ = declare_parameter<double>("range_jump_min", 0.05);
  min_cluster_points_ = declare_parameter<int>("min_cluster_points", 3);
  range_gap_abs_ = declare_parameter<double>("range_gap_abs", 0.5);
  declare_parameter<double>("mask_resolution", 0.05);
  declare_parameter<double>("mask_inflation_radius", 0.1);

  tf_buffer_ = std::make_shared<tf2_ros::Buffer>(this->get_clock());
  tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

  scan_sub_ = this->create_subscription<sensor_msgs::msg::LaserScan>(
    scan_topic_, rclcpp::SensorDataQoS(),
    std::bind(&ObstacleTrackerNode::laserScanCallback, this, std::placeholders::_1));

  auto marker_qos = rclcpp::QoS(rclcpp::KeepLast(10)).reliable();
  obstacles_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(
    obstacles_topic_, marker_qos);
  auto mask_qos = rclcpp::QoS(rclcpp::KeepLast(1)).reliable().transient_local();
  mask_pub_ = this->create_publisher<nav_msgs::msg::OccupancyGrid>(mask_topic_, mask_qos);

  RCLCPP_INFO(
    this->get_logger(),
    "obstacle_tracker 起動: scan=%s, obstacles=%s, mask=%s, target_frame=%s",
    scan_topic_.c_str(), obstacles_topic_.c_str(), mask_topic_.c_str(), target_frame_.c_str());
}

void ObstacleTrackerNode::laserScanCallback(const sensor_msgs::msg::LaserScan::SharedPtr msg)
{
  auto points_lidar = scanToPoints(*msg);
  bool transformed = false;
  auto points_map = transformToMap(
    points_lidar, msg->header.frame_id, msg->header.stamp, transformed);
  if (points_map.empty()) {
    return;
  }
  const std::string marker_frame = transformed ? target_frame_ : msg->header.frame_id;
  auto clusters = segmentByRangeJump(points_map);

  rclcpp::Time stamp(msg->header.stamp);
  std::optional<Point2D> sensor_origin;
  if (last_tf_) {
    sensor_origin = Point2D{
      last_tf_->transform.translation.x,
      last_tf_->transform.translation.y};
  }
  auto mask = buildOccupancyMask(points_map, clusters, stamp, marker_frame, sensor_origin);
  mask_pub_->publish(mask);

  auto cubes = buildCubes(clusters, true, stamp, marker_frame);
  auto outlines = buildOutlines(clusters, true, stamp, marker_frame);

  visualization_msgs::msg::MarkerArray merged;
  merged.markers.reserve(cubes.markers.size() + outlines.markers.size());
  merged.markers.insert(merged.markers.end(), cubes.markers.begin(), cubes.markers.end());
  merged.markers.insert(merged.markers.end(), outlines.markers.begin(), outlines.markers.end());

  obstacles_pub_->publish(merged);
}

std::vector<Point2D> ObstacleTrackerNode::scanToPoints(const sensor_msgs::msg::LaserScan & scan)
const
{
  std::vector<Point2D> pts;
  pts.reserve(scan.ranges.size());
  double angle = scan.angle_min;
  for (double range : scan.ranges) {
    if (!std::isfinite(range) || range < scan.range_min || range > scan.range_max) {
      angle += scan.angle_increment;
      continue;
    }
    if (range > processing_range_) {
      angle += scan.angle_increment;
      continue;
    }
    Point2D p{range * std::cos(angle), range * std::sin(angle)};
    pts.push_back(p);
    angle += scan.angle_increment;
  }
  return pts;
}

std::vector<Point2D> ObstacleTrackerNode::transformToMap(
  const std::vector<Point2D> & points, const std::string & frame, const rclcpp::Time & stamp,
  bool & success) const
{
  success = false;
  if (points.empty()) {
    return points;
  }

  std::vector<Point2D> out;
  out.reserve(points.size());

  if (frame == target_frame_) {
    success = true;
    return points;
  }

  geometry_msgs::msg::TransformStamped tf;
  try {
    tf = tf_buffer_->lookupTransform(
      target_frame_, frame, stamp, rclcpp::Duration::from_seconds(tf_timeout_sec_));
    last_tf_ = tf;
  } catch (const tf2::TransformException & ex) {
    bool recovered = false;
    // fallback to latest transform
    try {
      tf = tf_buffer_->lookupTransform(
        target_frame_, frame, rclcpp::Time(0),
        rclcpp::Duration::from_seconds(tf_timeout_sec_));
      last_tf_ = tf;
      recovered = true;
      RCLCPP_WARN(
        this->get_logger(),
        "TF %s<- %s unavailable at stamp %.3f: %s. Using latest available transform at %.3f.",
        target_frame_.c_str(), frame.c_str(), stamp.seconds(), ex.what(),
        rclcpp::Time(tf.header.stamp).seconds());
    } catch (const tf2::TransformException & ex_latest) {
      if (last_tf_.has_value() && last_tf_->header.frame_id == target_frame_ &&
        last_tf_->child_frame_id == frame)
      {
        tf = *last_tf_;
        recovered = true;
        RCLCPP_WARN(
          this->get_logger(),
          "TF %s<- %s unavailable at stamp %.3f: %s. Using last known transform at %.3f.",
          target_frame_.c_str(), frame.c_str(), stamp.seconds(), ex.what(),
          rclcpp::Time(last_tf_->header.stamp).seconds());
      } else {
        (void)ex_latest;
      }
    }
    if (!recovered) {
      RCLCPP_WARN(
        this->get_logger(),
        "TF %s<- %s unavailable at stamp %.3f and no fallback: %s. Using raw points.",
        target_frame_.c_str(), frame.c_str(), stamp.seconds(), ex.what());
      success = false;
      return points;
    }
  }

  for (const auto & p : points) {
    geometry_msgs::msg::PointStamped in, out_pt;
    in.header.frame_id = frame;
    in.header.stamp = stamp;
    in.point.x = p.x;
    in.point.y = p.y;
    in.point.z = 0.0;
    tf2::doTransform(in, out_pt, tf);
    out.push_back({out_pt.point.x, out_pt.point.y});
  }
  success = true;
  return out;
}

std::vector<Cluster> ObstacleTrackerNode::segmentByRangeJump(const std::vector<Point2D> & points)
const
{
  std::vector<Cluster> clusters;
  if (points.empty()) {
    return clusters;
  }

  Cluster current;
  auto push_cluster = [&]() {
      if (static_cast<int>(current.points.size()) >= min_cluster_points_) {
        double min_x = std::numeric_limits<double>::max();
        double max_x = -std::numeric_limits<double>::max();
        double min_y = std::numeric_limits<double>::max();
        double max_y = -std::numeric_limits<double>::max();
        double sum_x = 0.0, sum_y = 0.0;
        for (const auto & pt : current.points) {
          sum_x += pt.x;
          sum_y += pt.y;
          min_x = std::min(min_x, pt.x);
          max_x = std::max(max_x, pt.x);
          min_y = std::min(min_y, pt.y);
          max_y = std::max(max_y, pt.y);
        }
        current.centroid.x = sum_x / static_cast<double>(current.points.size());
        current.centroid.y = sum_y / static_cast<double>(current.points.size());
        current.length = std::max(0.1, max_x - min_x);
        current.width = std::max(0.1, max_y - min_y);
        clusters.push_back(current);
      }
      current = Cluster();
    };

  current.points.push_back(points.front());
  for (size_t i = 1; i < points.size(); ++i) {
    const auto & prev = points[i - 1];
    const auto & curr = points[i];
    double prev_r = std::hypot(prev.x, prev.y);
    double curr_r = std::hypot(curr.x, curr.y);
    double jump = std::hypot(curr.x - prev.x, curr.y - prev.y);
    double thresh = std::max(range_jump_min_, range_jump_ratio_ * std::min(prev_r, curr_r));
    if (jump > thresh || std::abs(curr_r - prev_r) > range_gap_abs_) {
      push_cluster();
    }
    current.points.push_back(curr);
  }
  push_cluster();

  return clusters;
}

visualization_msgs::msg::MarkerArray ObstacleTrackerNode::buildCubes(
  const std::vector<Cluster> & clusters, bool dynamic, const rclcpp::Time & stamp,
  const std::string & frame_id) const
{
  visualization_msgs::msg::MarkerArray arr;
  int id = 1;
  for (const auto & c : clusters) {
    visualization_msgs::msg::Marker m;
    m.header.frame_id = frame_id;
    m.header.stamp = stamp;
    m.ns = dynamic ? "dynamic_obstacles" : "static_obstacles";
    m.id = id++;
    m.type = visualization_msgs::msg::Marker::CUBE;
    m.action = visualization_msgs::msg::Marker::ADD;
    m.pose.position.x = c.centroid.x;
    m.pose.position.y = c.centroid.y;
    m.pose.position.z = 0.0;
    m.pose.orientation.w = 1.0;
    m.scale.x = std::max(0.1, c.length);
    m.scale.y = std::max(0.1, c.width);
    m.scale.z = 0.1;
    if (dynamic) {
      m.color.r = 1.0; m.color.g = 0.2; m.color.b = 0.2;
    } else {
      m.color.r = 0.2; m.color.g = 0.2; m.color.b = 1.0;
    }
    m.color.a = 1.0;
    m.lifetime = rclcpp::Duration::from_seconds(0.5);
    arr.markers.push_back(m);
  }
  return arr;
}

visualization_msgs::msg::MarkerArray ObstacleTrackerNode::buildOutlines(
  const std::vector<Cluster> & clusters, bool dynamic, const rclcpp::Time & stamp,
  const std::string & frame_id) const
{
  visualization_msgs::msg::MarkerArray arr;
  int id = 1;
  for (const auto & c : clusters) {
    visualization_msgs::msg::Marker m;
    m.header.frame_id = frame_id;
    m.header.stamp = stamp;
    m.ns = dynamic ? "dynamic_obstacles" : "static_obstacles";
    m.id = id++;
    m.type = visualization_msgs::msg::Marker::LINE_LIST;
    m.action = visualization_msgs::msg::Marker::ADD;
    m.pose.orientation.w = 1.0;
    m.scale.x = 0.05;
    if (dynamic) {
      m.color.r = 1.0; m.color.g = 0.2; m.color.b = 0.2;
    } else {
      m.color.r = 0.2; m.color.g = 0.2; m.color.b = 1.0;
    }
    m.color.a = 1.0;
    m.lifetime = rclcpp::Duration::from_seconds(0.5);

    // Convex hull only
    auto hull = computeHull(c.points);
    if (hull.size() >= 2) {
      for (size_t i = 0; i < hull.size(); ++i) {
        geometry_msgs::msg::Point p1, p2;
        const auto & a = hull[i];
        const auto & b = hull[(i + 1) % hull.size()];
        p1.x = a.x; p1.y = a.y; p1.z = 0.0;
        p2.x = b.x; p2.y = b.y; p2.z = 0.0;
        m.points.push_back(p1);
        m.points.push_back(p2);
      }
    }
    arr.markers.push_back(m);
  }
  return arr;
}

nav_msgs::msg::OccupancyGrid ObstacleTrackerNode::buildOccupancyMask(
  const std::vector<Point2D> & points,
  const std::vector<Cluster> & clusters,
  const rclcpp::Time & stamp,
  const std::string & frame_id,
  const std::optional<Point2D> & sensor_origin) const
{
  (void)points;  // 未使用だがインターフェース維持のため
  nav_msgs::msg::OccupancyGrid grid;
  grid.header.frame_id = frame_id;
  grid.header.stamp = stamp;

  const double resolution_param = this->get_parameter("mask_resolution").as_double();
  const double range_param = this->get_parameter("processing_range").as_double();
  const double inflation_param = this->get_parameter("mask_inflation_radius").as_double();
  const double resolution = std::max(0.01, resolution_param);
  const double range = std::max(resolution, range_param);
  const double inflation = std::max(0.0, inflation_param);
  const uint32_t width = static_cast<uint32_t>(std::ceil((range * 2.0) / resolution)) + 1;
  grid.info.resolution = resolution;
  grid.info.width = width;
  grid.info.height = width;

  const Point2D origin_pt = sensor_origin.value_or(Point2D{0.0, 0.0});
  const double origin_x = origin_pt.x - range;
  const double origin_y = origin_pt.y - range;

  grid.info.origin.position.x = origin_x;
  grid.info.origin.position.y = origin_y;
  grid.info.origin.position.z = 0.0;
  grid.info.origin.orientation.w = 1.0;

  grid.data.assign(static_cast<size_t>(grid.info.width) * grid.info.height, -1);

  auto toIndex = [&](double x, double y) -> std::optional<size_t> {
      int col = static_cast<int>(std::floor((x - origin_x) / resolution));
      int row = static_cast<int>(std::floor((y - origin_y) / resolution));
      if (col < 0 || row < 0 ||
        col >= static_cast<int>(grid.info.width) || row >= static_cast<int>(grid.info.height))
      {
        return std::nullopt;
      }
      return static_cast<size_t>(row) * grid.info.width + static_cast<size_t>(col);
    };

  auto pointInside = [](const std::vector<Point2D> & poly, const Point2D & pt) {
      bool inside = false;
      const size_t n = poly.size();
      for (size_t i = 0, j = n - 1; i < n; j = i++) {
        const bool intersect = ((poly[i].y > pt.y) != (poly[j].y > pt.y)) &&
          (pt.x < (poly[j].x - poly[i].x) * (pt.y - poly[i].y) /
          (poly[j].y - poly[i].y + 1e-12) + poly[i].x);
        if (intersect) {
          inside = !inside;
        }
      }
      return inside;
    };

  auto distanceToSegment = [](const Point2D & a, const Point2D & b, const Point2D & p) {
      const double vx = b.x - a.x;
      const double vy = b.y - a.y;
      const double wx = p.x - a.x;
      const double wy = p.y - a.y;
      const double c1 = vx * wx + vy * wy;
      if (c1 <= 0.0) {
        return std::hypot(wx, wy);
      }
      const double c2 = vx * vx + vy * vy;
      if (c2 <= c1) {
        return std::hypot(p.x - b.x, p.y - b.y);
      }
      const double t = c1 / c2;
      const double proj_x = a.x + t * vx;
      const double proj_y = a.y + t * vy;
      return std::hypot(p.x - proj_x, p.y - proj_y);
    };

  auto distanceToPolygon = [&](const std::vector<Point2D> & poly, const Point2D & p) {
      double best = std::numeric_limits<double>::infinity();
      if (pointInside(poly, p)) {
        return 0.0;
      }
      for (size_t i = 0; i < poly.size(); ++i) {
        const auto & a = poly[i];
        const auto & b = poly[(i + 1) % poly.size()];
        best = std::min(best, distanceToSegment(a, b, p));
      }
      return best;
    };

  for (const auto & c : clusters) {
    if (c.points.empty()) {
      continue;
    }
    const auto hull = computeHull(c.points);
    if (hull.empty()) {
      continue;
    }
    std::vector<Point2D> poly;
    if (hull.size() >= 3) {
      poly = hull;  // 凸包でより点群に沿う形に塗る
    } else {
      const auto obb = computeObb(hull);
      poly.assign(obb.begin(), obb.end());
    }

    double min_x = poly.front().x;
    double max_x = poly.front().x;
    double min_y = poly.front().y;
    double max_y = poly.front().y;
    for (const auto & pt : poly) {
      min_x = std::min(min_x, pt.x);
      max_x = std::max(max_x, pt.x);
      min_y = std::min(min_y, pt.y);
      max_y = std::max(max_y, pt.y);
    }
    const double margin = inflation;
    const int min_col = std::max(
      0, static_cast<int>(std::floor((min_x - margin - origin_x) / resolution)));
    const int max_col = std::min(
      static_cast<int>(grid.info.width) - 1,
      static_cast<int>(std::floor((max_x + margin - origin_x) / resolution)));
    const int min_row = std::max(
      0, static_cast<int>(std::floor((min_y - margin - origin_y) / resolution)));
    const int max_row = std::min(
      static_cast<int>(grid.info.height) - 1,
      static_cast<int>(std::floor((max_y + margin - origin_y) / resolution)));

    if (min_col > max_col || min_row > max_row) {
      continue;
    }

    for (int row = min_row; row <= max_row; ++row) {
      for (int col = min_col; col <= max_col; ++col) {
        const double cx = origin_x + (static_cast<double>(col) + 0.5) * resolution;
        const double cy = origin_y + (static_cast<double>(row) + 0.5) * resolution;
        Point2D cell_pt{cx, cy};
        const bool inside = pointInside(poly, cell_pt);
        const double dist = distanceToPolygon(poly, cell_pt);
        if (inside || dist <= inflation) {
          const size_t idx = static_cast<size_t>(row) * grid.info.width + static_cast<size_t>(col);
          grid.data[idx] = 100;
        }
      }
    }

    for (const auto & pt : c.points) {
      auto idx = toIndex(pt.x, pt.y);
      if (idx.has_value()) {
        grid.data[*idx] = 100;
      }
    }
  }

  return grid;
}

std::vector<Point2D> ObstacleTrackerNode::computeHull(const std::vector<Point2D> & pts) const
{
  std::vector<Point2D> p = pts;
  if (p.size() < 2) {
    return p;
  }
  std::sort(
    p.begin(), p.end(), [](const Point2D & a, const Point2D & b) {
      if (a.x == b.x) {
        return a.y < b.y;
      }
      return a.x < b.x;
    });
  auto cross = [](const Point2D & o, const Point2D & a, const Point2D & b) {
      return (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x);
    };

  std::vector<Point2D> lower;
  for (const auto & pt : p) {
    while (lower.size() >= 2 && cross(lower[lower.size() - 2], lower.back(), pt) <= 0.0) {
      lower.pop_back();
    }
    lower.push_back(pt);
  }
  std::vector<Point2D> upper;
  for (auto it = p.rbegin(); it != p.rend(); ++it) {
    while (upper.size() >= 2 && cross(upper[upper.size() - 2], upper.back(), *it) <= 0.0) {
      upper.pop_back();
    }
    upper.push_back(*it);
  }
  lower.pop_back();
  upper.pop_back();
  lower.insert(lower.end(), upper.begin(), upper.end());
  return lower;
}

std::array<Point2D, 4> ObstacleTrackerNode::computeObb(const std::vector<Point2D> & hull) const
{
  std::array<Point2D, 4> best{{{0, 0}, {0, 0}, {0, 0}, {0, 0}}};
  if (hull.empty()) {
    return best;
  }
  if (hull.size() == 1) {
    double half = 0.05;
    best = {{{hull[0].x - half, hull[0].y - half},
      {hull[0].x + half, hull[0].y - half},
      {hull[0].x + half, hull[0].y + half},
      {hull[0].x - half, hull[0].y + half}}};
    return best;
  }

  auto addCorner = [](std::array<Point2D, 4> & arr_corner,
      double min_x, double max_x, double min_y, double max_y, double c, double s) {
      arr_corner = {{
        {c * min_x - s * min_y, s * min_x + c * min_y},
        {c * max_x - s * min_y, s * max_x + c * min_y},
        {c * max_x - s * max_y, s * max_x + c * max_y},
        {c * min_x - s * max_y, s * min_x + c * max_y}
      }};
    };

  double best_area = std::numeric_limits<double>::max();
  for (size_t i = 0; i < hull.size(); ++i) {
    const auto & p0 = hull[i];
    const auto & p1 = hull[(i + 1) % hull.size()];
    double dx = p1.x - p0.x;
    double dy = p1.y - p0.y;
    double len = std::hypot(dx, dy);
    if (len < 1e-6) {
      continue;
    }
    double c = dx / len;
    double s = dy / len;
    double min_x = std::numeric_limits<double>::max();
    double max_x = -std::numeric_limits<double>::max();
    double min_y = std::numeric_limits<double>::max();
    double max_y = -std::numeric_limits<double>::max();
    for (const auto & p : hull) {
      double rx = c * p.x + s * p.y;
      double ry = -s * p.x + c * p.y;
      min_x = std::min(min_x, rx);
      max_x = std::max(max_x, rx);
      min_y = std::min(min_y, ry);
      max_y = std::max(max_y, ry);
    }
    double area = (max_x - min_x) * (max_y - min_y);
    if (area < best_area) {
      best_area = area;
      addCorner(best, min_x, max_x, min_y, max_y, c, s);
    }
  }
  return best;
}

}  // namespace obstacle_tracker

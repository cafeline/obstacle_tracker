#include <gtest/gtest.h>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/laser_scan.hpp>
#include <nav_msgs/msg/occupancy_grid.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <chrono>
#include <thread>
#include "obstacle_tracker/obstacle_tracker_node.hpp"

using obstacle_tracker::ObstacleTrackerNode;
using obstacle_tracker::Cluster;
using obstacle_tracker::Point2D;

sensor_msgs::msg::LaserScan makeScan()
{
  sensor_msgs::msg::LaserScan scan;
  scan.header.frame_id = "map";
  scan.angle_min = -1.0;
  scan.angle_max = 1.0;
  scan.angle_increment = 0.05;
  scan.range_min = 0.1;
  scan.range_max = 10.0;
  size_t n = static_cast<size_t>((scan.angle_max - scan.angle_min) / scan.angle_increment) + 1;
  scan.ranges.assign(n, std::numeric_limits<float>::quiet_NaN());
  // cluster 1 around 2m straight ahead for small angles
  for (size_t i = 8; i <= 12; ++i) {
    scan.ranges[i] = 2.0;
  }
  // cluster 2 to the left
  for (size_t i = 2; i <= 4; ++i) {
    scan.ranges[i] = 3.0;
  }
  return scan;
}

TEST(ObstacleTrackerNodeTest, SegmentationSplitsClusters)
{
  auto node = std::make_shared<ObstacleTrackerNode>();

  auto scan = makeScan();
  auto pts = node->scanToPoints(scan);
  auto clusters = node->segmentByRangeJump(pts);

  ASSERT_GE(clusters.size(), 2u);
  for (const auto & c : clusters) {
    EXPECT_TRUE(std::isfinite(c.centroid.x));
    EXPECT_TRUE(std::isfinite(c.centroid.y));
  }

}

TEST(ObstacleTrackerNodeTest, IgnoresSparseClusters)
{
  auto node = std::make_shared<ObstacleTrackerNode>();

  sensor_msgs::msg::LaserScan scan;
  scan.header.frame_id = "map";
  scan.angle_min = 0.0;
  scan.angle_max = 0.5;
  scan.angle_increment = 0.1;
  scan.range_min = 0.1;
  scan.range_max = 10.0;
  scan.ranges = {1.0, 1.0}; // only two points

  auto pts = node->scanToPoints(scan);
  auto clusters = node->segmentByRangeJump(pts);
  EXPECT_TRUE(clusters.empty());

}

TEST(ObstacleTrackerNodeTest, TransformFailureSkipsPublish)
{
  auto node = std::make_shared<ObstacleTrackerNode>();
  std::vector<Point2D> pts{{1.0, 0.0}};
  bool ok = true;
  auto out = node->transformToMap(pts, "lidar_link", rclcpp::Time(0), ok);
  EXPECT_FALSE(ok);
  EXPECT_EQ(out.size(), pts.size());
}

TEST(ObstacleTrackerNodeTest, ConvexHullReturnsOrdered)
{
  auto node = std::make_shared<ObstacleTrackerNode>();
  std::vector<Point2D> pts{{0, 0}, {1, 0}, {1, 1}, {0, 1}, {0.5, 0.5}};
  auto hull = node->computeHull(pts);
  EXPECT_GE(hull.size(), 4u);
}

TEST(ObstacleTrackerNodeTest, MaskUsesResolutionAndRange)
{
  auto node = std::make_shared<ObstacleTrackerNode>();
  node->set_parameter(rclcpp::Parameter("mask_resolution", 0.5));
  node->set_parameter(rclcpp::Parameter("processing_range", 1.0));

  std::vector<Point2D> points{{0.2, 0.0}};
  Cluster c;
  c.points = points;
  c.centroid = points[0];
  c.length = 0.2;
  c.width = 0.2;
  std::vector<Cluster> clusters{c};

  auto grid = node->buildOccupancyMask(
    points, clusters, rclcpp::Time(1, 0), "map", std::nullopt);

  EXPECT_DOUBLE_EQ(grid.info.resolution, 0.5);
  EXPECT_EQ(grid.info.width, 5u);
  EXPECT_EQ(grid.info.height, 5u);
  EXPECT_EQ(grid.header.frame_id, "map");
  EXPECT_EQ(grid.data.size(), grid.info.width * grid.info.height);
}

TEST(ObstacleTrackerNodeTest, MaskMarksObstacleAndFreeCells)
{
  auto node = std::make_shared<ObstacleTrackerNode>();
  node->set_parameter(rclcpp::Parameter("mask_resolution", 0.5));
  node->set_parameter(rclcpp::Parameter("processing_range", 1.0));
  node->set_parameter(rclcpp::Parameter("mask_inflation_radius", 0.0));

  std::vector<Point2D> points{{0.8, 0.0}};
  Cluster c;
  c.points = points;
  c.centroid = points[0];
  c.length = 0.2;
  c.width = 0.2;
  std::vector<Cluster> clusters{c};

  auto grid = node->buildOccupancyMask(
    points, clusters, rclcpp::Time(1, 0), "map", Point2D{0.0, 0.0});

  // Cell for the obstacle point should be 100
  auto idx = [w = grid.info.width](uint32_t x, uint32_t y) {return static_cast<size_t>(y) * w + x;};
  EXPECT_EQ(grid.data[idx(3, 2)], 100);

  // Unknown cells remain -1
  EXPECT_EQ(grid.data[idx(0, 0)], -1);
}

TEST(ObstacleTrackerNodeTest, MaskFollowsHullNotBoundingBox)
{
  auto node = std::make_shared<ObstacleTrackerNode>();
  node->set_parameter(rclcpp::Parameter("mask_resolution", 0.5));
  node->set_parameter(rclcpp::Parameter("processing_range", 2.0));
  node->set_parameter(rclcpp::Parameter("mask_inflation_radius", 0.0));

  std::vector<Point2D> points{{0.0, 0.0}, {1.0, 0.0}, {0.0, 1.0}};  // 三角形
  Cluster c;
  c.points = points;
  c.centroid = {0.33, 0.33};
  c.length = 1.0;
  c.width = 1.0;
  std::vector<Cluster> clusters{c};

  auto grid = node->buildOccupancyMask(
    points, clusters, rclcpp::Time(1, 0), "map", Point2D{0.0, 0.0});

  auto idx = [w = grid.info.width](uint32_t x, uint32_t y) {return static_cast<size_t>(y) * w + x;};

  // (1,1) は OBB なら塗られるが凸包三角形では外側 -> 未観測のまま
  EXPECT_EQ(grid.data[idx(6, 6)], -1);

  // 三角形辺近傍は占有
  EXPECT_EQ(grid.data[idx(4, 4)], 100);
}

TEST(ObstacleTrackerNodeTest, ClustersAreSplitInSensorFrameBeforeTransform)
{
  auto node = std::make_shared<ObstacleTrackerNode>();

  geometry_msgs::msg::TransformStamped tf;
  tf.header.stamp = node->now();
  tf.header.frame_id = "map";
  tf.child_frame_id = "lidar";
  tf.transform.translation.x = 50.0;
  tf.transform.translation.y = 0.0;
  tf.transform.rotation.w = 1.0;
  node->setTestTransform(tf);

  rclcpp::executors::SingleThreadedExecutor exec;
  exec.add_node(node);

  auto receiver = std::make_shared<rclcpp::Node>("obstacles_receiver");
  visualization_msgs::msg::MarkerArray::SharedPtr latest;
  bool got = false;
  auto sub = receiver->create_subscription<visualization_msgs::msg::MarkerArray>(
    "/obstacles", 10,
    [&](const visualization_msgs::msg::MarkerArray & msg) {
      latest = std::make_shared<visualization_msgs::msg::MarkerArray>(msg);
      got = true;
    });
  (void)sub;
  exec.add_node(receiver);

  auto pub_node = std::make_shared<rclcpp::Node>("scan_publisher");
  auto scan_pub = pub_node->create_publisher<sensor_msgs::msg::LaserScan>(
    "/scan", rclcpp::SensorDataQoS());
  exec.add_node(pub_node);

  sensor_msgs::msg::LaserScan scan;
  scan.header.frame_id = "lidar";
  scan.header.stamp = node->now();
  scan.angle_min = 1.5708;
  scan.angle_max = 1.8208;
  scan.angle_increment = 0.05;
  scan.range_min = 0.1;
  scan.range_max = 100.0;
  scan.ranges = {1.0, 1.0, 1.0, 2.0, 2.0, 2.0};
  scan_pub->publish(scan);

  const auto start = std::chrono::steady_clock::now();
  while (!got && (std::chrono::steady_clock::now() - start) < std::chrono::seconds(1)) {
    exec.spin_some();
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }

  ASSERT_TRUE(got);
  ASSERT_TRUE(latest);
  EXPECT_EQ(latest->markers.size(), 4u);
}

int main(int argc, char ** argv)
{
  testing::InitGoogleTest(&argc, argv);
  if (rclcpp::ok()) {
    rclcpp::shutdown();
  }
  rclcpp::init(argc, argv);
  int ret = RUN_ALL_TESTS();
  rclcpp::shutdown();
  return ret;
}

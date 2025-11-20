#include <gtest/gtest.h>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/laser_scan.hpp>
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

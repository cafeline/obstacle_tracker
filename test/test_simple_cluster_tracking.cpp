#include <gtest/gtest.h>
#include <rclcpp/rclcpp.hpp>
#include "obstacle_tracker/obstacle_tracker_node.hpp"
#include <cmath>

using obstacle_tracker::Cluster;
using obstacle_tracker::Point3D;
using obstacle_tracker::TrackManager;
using obstacle_tracker::TrackManagerConfig;

namespace
{

Cluster makeCluster(double x, double y, double spread = 0.2, int count = 6)
{
  Cluster cluster;
  cluster.centroid = Point3D(x, y, 0.0);
  cluster.points.reserve(count);
  for (int i = 0; i < count; ++i) {
    double angle = 2.0 * M_PI * static_cast<double>(i) / static_cast<double>(count);
    cluster.points.push_back(
      Point3D(
        x + std::cos(angle) * spread,
        y + std::sin(angle) * spread,
        0.0,
        angle));
  }
  return cluster;
}

} // namespace

class TrackManagerTest : public ::testing::Test
{
protected:
  void SetUp() override
  {
    rclcpp::init(0, nullptr);
    config_.process_noise = 0.5;
    config_.measurement_noise = 0.1;
    config_.gating_distance = 1.0;
    config_.max_missed_frames = 2;
    config_.min_consecutive_hits = 1;
  }

  void TearDown() override
  {
    rclcpp::shutdown();
  }

  TrackManagerConfig config_;
};

TEST_F(TrackManagerTest, MaintainsStableIdsAcrossFrames)
{
  TrackManager manager(config_);
  std::vector<Cluster> first;
  first.push_back(makeCluster(1.0, 0.0));
  first.push_back(makeCluster(-2.0, 0.5));

  auto tracked_first = manager.update(first, 0.1);
  ASSERT_EQ(tracked_first.size(), 2u);

  std::vector<Cluster> second;
  second.push_back(makeCluster(1.1, -0.05));
  second.push_back(makeCluster(-2.2, 0.45));

  auto tracked_second = manager.update(second, 0.1);
  ASSERT_EQ(tracked_second.size(), 2u);

  EXPECT_EQ(tracked_first[0].id, tracked_second[0].id);
  EXPECT_EQ(tracked_first[1].id, tracked_second[1].id);
  EXPECT_GT(tracked_second[0].speed, 0.0);
  EXPECT_GT(tracked_second[1].speed, 0.0);
}

TEST_F(TrackManagerTest, RemovesTracksAfterMissThreshold)
{
  TrackManager manager(config_);
  std::vector<Cluster> frame = {makeCluster(0.0, 0.0)};
  auto tracked = manager.update(frame, 0.1);
  ASSERT_EQ(tracked.size(), 1u);
  EXPECT_EQ(manager.activeTrackCount(), 1u);

  // 2フレーム連続で観測しない
  manager.update({}, 0.1);
  auto after_drop = manager.update({}, 0.1);
  EXPECT_TRUE(after_drop.empty());
  EXPECT_EQ(manager.activeTrackCount(), 0u);
}

TEST_F(TrackManagerTest, CreatesNewTracksForUnmatchedClusters)
{
  TrackManager manager(config_);
  auto tracked_initial = manager.update({}, 0.1);
  EXPECT_TRUE(tracked_initial.empty());

  std::vector<Cluster> far_cluster = {makeCluster(5.0, -5.0)};
  auto tracked_second = manager.update(far_cluster, 0.1);
  ASSERT_EQ(tracked_second.size(), 1u);
  EXPECT_GE(tracked_second[0].confidence, 0.4);
  EXPECT_EQ(tracked_second[0].track_count, 1);
}

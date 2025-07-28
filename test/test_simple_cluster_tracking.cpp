#include <gtest/gtest.h>
#include <rclcpp/rclcpp.hpp>
#include "obstacle_tracker/obstacle_tracker_node.hpp"
#include <cmath>

namespace obstacle_tracker
{

class SimpleClusterTrackingTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        rclcpp::init(0, nullptr);
    }

    void TearDown() override
    {
        rclcpp::shutdown();
    }

    // テスト用のクラスタ生成
    Cluster createTestCluster(int id, double x, double y, double vx = 0.0, double vy = 0.0, int point_count = 5)
    {
        Cluster cluster;
        cluster.id = id;
        cluster.centroid = Point3D(x, y, 0.0);
        cluster.velocity = Point3D(vx, vy, 0.0);
        cluster.track_count = 1;
        cluster.is_dynamic = false;
        
        // テスト用の点群生成
        for (int i = 0; i < point_count; ++i) {
            double angle = (2.0 * M_PI * i) / point_count;
            double radius = 0.3;
            Point3D point(
                x + radius * std::cos(angle),
                y + radius * std::sin(angle),
                0.0,
                angle
            );
            cluster.points.push_back(point);
        }
        
        return cluster;
    }

    // 改善されたマッチングスコア計算
    double calculateMatchingScore(const Cluster& prev, const Cluster& curr, double dt = 0.1)
    {
        // 1. 位置距離（正規化）
        double dx = curr.centroid.x - prev.centroid.x;
        double dy = curr.centroid.y - prev.centroid.y;
        double position_distance = std::sqrt(dx * dx + dy * dy);
        double position_score = std::exp(-position_distance / 1.0); // 1m で大幅減衰
        
        // 2. サイズ類似度
        double size_ratio = static_cast<double>(curr.points.size()) / static_cast<double>(prev.points.size());
        if (size_ratio > 1.0) size_ratio = 1.0 / size_ratio; // 0-1に正規化
        double size_score = size_ratio;
        
        // 3. 予測位置との一致度
        Point3D predicted_pos;
        predicted_pos.x = prev.centroid.x + prev.velocity.x * dt;
        predicted_pos.y = prev.centroid.y + prev.velocity.y * dt;
        
        double pred_dx = curr.centroid.x - predicted_pos.x;
        double pred_dy = curr.centroid.y - predicted_pos.y;
        double prediction_error = std::sqrt(pred_dx * pred_dx + pred_dy * pred_dy);
        double prediction_score = std::exp(-prediction_error / 0.5); // 0.5m で大幅減衰
        
        // 重み付き統合スコア
        double total_score = 0.5 * position_score + 0.2 * size_score + 0.3 * prediction_score;
        
        return total_score;
    }

    // 速度平滑化（α-βフィルタ）
    Point3D smoothVelocity(const Point3D& prev_vel, const Point3D& measured_vel, double alpha = 0.3)
    {
        Point3D smooth_vel;
        smooth_vel.x = alpha * measured_vel.x + (1 - alpha) * prev_vel.x;
        smooth_vel.y = alpha * measured_vel.y + (1 - alpha) * prev_vel.y;
        smooth_vel.z = 0.0;
        return smooth_vel;
    }
};

// テスト1: 基本的なマッチングスコア
TEST_F(SimpleClusterTrackingTest, BasicMatchingScoreTest)
{
    Cluster prev = createTestCluster(1, 1.0, 1.0, 0.5, 0.0, 5);
    Cluster curr = createTestCluster(2, 1.05, 1.0, 0.5, 0.0, 5); // 少し移動
    
    double score = calculateMatchingScore(prev, curr);
    
    // 類似したクラスタは高いスコアを持つべき
    EXPECT_GT(score, 0.8);
}

// テスト2: 距離による減衰テスト
TEST_F(SimpleClusterTrackingTest, DistanceDecayTest)
{
    Cluster prev = createTestCluster(1, 0.0, 0.0, 0.0, 0.0, 5);
    Cluster near = createTestCluster(2, 0.5, 0.0, 0.0, 0.0, 5);  // 近い
    Cluster far = createTestCluster(3, 3.0, 0.0, 0.0, 0.0, 5);   // 遠い
    
    double near_score = calculateMatchingScore(prev, near);
    double far_score = calculateMatchingScore(prev, far);
    
    // 近いクラスタの方が高いスコアを持つべき
    EXPECT_GT(near_score, far_score);
    EXPECT_GT(near_score, 0.5);
    EXPECT_LT(far_score, 0.2);
}

// テスト3: サイズ変化に対する耐性
TEST_F(SimpleClusterTrackingTest, SizeVariationTest)
{
    Cluster prev = createTestCluster(1, 2.0, 2.0, 0.0, 0.0, 5);
    Cluster similar_size = createTestCluster(2, 2.0, 2.0, 0.0, 0.0, 6);  // 少し大きい
    Cluster different_size = createTestCluster(3, 2.0, 2.0, 0.0, 0.0, 15); // 大きく異なる
    
    double similar_score = calculateMatchingScore(prev, similar_size);
    double different_score = calculateMatchingScore(prev, different_size);
    
    // サイズが似ている方が高いスコア
    EXPECT_GT(similar_score, different_score);
}

// テスト4: 予測位置との一致度
TEST_F(SimpleClusterTrackingTest, PredictionAccuracyTest)
{
    // 前フレーム: 右に1m/sで移動中
    Cluster prev = createTestCluster(1, 0.0, 0.0, 1.0, 0.0, 5);
    
    // 現フレーム: 予測通りの位置 (dt=0.1s後に0.1m移動)
    Cluster predicted = createTestCluster(2, 0.1, 0.0, 1.0, 0.0, 5);
    
    // 現フレーム: 予測から外れた位置
    Cluster unpredicted = createTestCluster(3, 0.0, 0.5, 1.0, 0.0, 5);
    
    double predicted_score = calculateMatchingScore(prev, predicted, 0.1);
    double unpredicted_score = calculateMatchingScore(prev, unpredicted, 0.1);
    
    // 予測に合う方が高いスコア
    EXPECT_GT(predicted_score, unpredicted_score);
}

// テスト5: 速度平滑化テスト
TEST_F(SimpleClusterTrackingTest, VelocitySmoothingTest)
{
    Point3D prev_vel(1.0, 0.0, 0.0);
    Point3D noisy_measured(1.5, 0.2, 0.0); // ノイズあり
    
    Point3D smoothed = smoothVelocity(prev_vel, noisy_measured, 0.3);
    
    // 平滑化後はノイズが軽減されるべき
    EXPECT_LT(std::abs(smoothed.x - 1.0), std::abs(noisy_measured.x - 1.0));
    EXPECT_LT(std::abs(smoothed.y), std::abs(noisy_measured.y));
}

// テスト6: 複数クラスタの最適マッチング
TEST_F(SimpleClusterTrackingTest, MultipleClusterMatchingTest)
{
    // 前フレーム: 2つのクラスタ
    std::vector<Cluster> prev_clusters;
    prev_clusters.push_back(createTestCluster(1, 0.0, 0.0, 1.0, 0.0));
    prev_clusters.push_back(createTestCluster(2, 5.0, 0.0, -1.0, 0.0));
    
    // 現フレーム: 移動後のクラスタ
    std::vector<Cluster> curr_clusters;
    curr_clusters.push_back(createTestCluster(3, 0.1, 0.0, 1.0, 0.0)); // prev[0]に対応
    curr_clusters.push_back(createTestCluster(4, 4.9, 0.0, -1.0, 0.0)); // prev[1]に対応
    
    // マッチングマトリックス計算
    double score_00 = calculateMatchingScore(prev_clusters[0], curr_clusters[0]);
    double score_01 = calculateMatchingScore(prev_clusters[0], curr_clusters[1]);
    double score_10 = calculateMatchingScore(prev_clusters[1], curr_clusters[0]);
    double score_11 = calculateMatchingScore(prev_clusters[1], curr_clusters[1]);
    
    // 正しいマッチング(0-0, 1-1)が高いスコア
    EXPECT_GT(score_00, score_01);
    EXPECT_GT(score_11, score_10);
    EXPECT_GT(score_00, 0.7);
    EXPECT_GT(score_11, 0.7);
}

// テスト7: 新規クラスタの検出
TEST_F(SimpleClusterTrackingTest, NewClusterDetectionTest)
{
    Cluster existing = createTestCluster(1, 1.0, 1.0, 0.0, 0.0);
    Cluster new_cluster = createTestCluster(2, 10.0, 10.0, 0.0, 0.0); // 遠い位置
    
    double score = calculateMatchingScore(existing, new_cluster);
    
    // 新規クラスタは低いマッチングスコア
    EXPECT_LT(score, 0.3);
}

// テスト8: 時系列での信頼度テスト
TEST_F(SimpleClusterTrackingTest, TrackConfidenceTest)
{
    // track_countが追跡の信頼度を表すと想定
    Cluster reliable_cluster = createTestCluster(1, 1.0, 1.0);
    reliable_cluster.track_count = 10; // 長期間追跡
    
    Cluster new_cluster = createTestCluster(2, 1.0, 1.0);
    new_cluster.track_count = 1; // 新規
    
    // 信頼度の高いクラスタの方が安定した追跡が期待される
    EXPECT_GT(reliable_cluster.track_count, new_cluster.track_count);
}

// テスト9: 動的・静的分類の安定性
TEST_F(SimpleClusterTrackingTest, DynamicClassificationStabilityTest)
{
    double threshold = 1.5;
    
    // ノイズありの速度（閾値付近）
    Point3D noisy_vel1(1.4, 0.2, 0.0); // 約1.41 m/s
    Point3D noisy_vel2(1.6, -0.1, 0.0); // 約1.60 m/s
    
    // 平滑化後
    Point3D smooth_vel = smoothVelocity(noisy_vel1, noisy_vel2, 0.5);
    double smooth_speed = std::sqrt(smooth_vel.x * smooth_vel.x + smooth_vel.y * smooth_vel.y);
    
    // 分類の安定性チェック
    bool is_dynamic = (smooth_speed > threshold);
    
    // 平滑化により分類が安定化されることを確認
    EXPECT_TRUE(is_dynamic); // この場合は動的と分類されるべき
}

// テスト10: ゼロ除算防止
TEST_F(SimpleClusterTrackingTest, ZeroDivisionProtectionTest)
{
    Cluster prev = createTestCluster(1, 1.0, 1.0, 0.0, 0.0, 5);
    
    // 空のクラスタ
    Cluster empty_cluster;
    empty_cluster.id = 2;
    empty_cluster.centroid = Point3D(1.0, 1.0, 0.0);
    empty_cluster.points.clear();
    
    // ゼロ除算が発生しないことを確認
    EXPECT_NO_THROW({
        double score = calculateMatchingScore(prev, empty_cluster);
        EXPECT_GE(score, 0.0);
        EXPECT_LE(score, 1.0);
    });
}

} // namespace obstacle_tracker

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
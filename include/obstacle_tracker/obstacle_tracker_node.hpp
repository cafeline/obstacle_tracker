#ifndef OBSTACLE_TRACKER_NODE_HPP_
#define OBSTACLE_TRACKER_NODE_HPP_

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/laser_scan.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <geometry_msgs/msg/point.hpp>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/buffer.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <tf2/time.h>
#include <vector>
#include <memory>
#include <map>
#include <deque>

namespace obstacle_tracker
{

struct Point3D
{
    double x, y, z;
    double angle;  // LiDARからの角度情報（ラジアン）
    Point3D(double x = 0.0, double y = 0.0, double z = 0.0, double angle = 0.0) 
        : x(x), y(y), z(z), angle(angle) {}
};

struct Cluster
{
    std::vector<Point3D> points;
    Point3D centroid;
    Point3D velocity;
    Point3D smoothed_velocity;  // 平滑化された速度
    int id;
    bool is_dynamic;
    int track_count;
    double confidence;  // 追跡信頼度 (0.0-1.0)
    int missing_count;  // 検出されなかった連続フレーム数
    
    Cluster() : centroid(0, 0, 0), velocity(0, 0, 0), smoothed_velocity(0, 0, 0), 
                id(0), is_dynamic(false), track_count(0), confidence(0.0), missing_count(0) {}
};

class ObstacleTrackerNode : public rclcpp::Node
{
public:
    ObstacleTrackerNode();

private:
    // コールバック関数
    void laserScanCallback(const sensor_msgs::msg::LaserScan::SharedPtr msg);
    
    // データ処理関数
    std::vector<Point3D> convertScanToPoints(const sensor_msgs::msg::LaserScan::SharedPtr scan);
    std::vector<Point3D> voxelizePoints(const std::vector<Point3D>& points);
    std::vector<Cluster> clusterPoints(const std::vector<Point3D>& voxelized_points, const rclcpp::Time& timestamp);
    std::vector<Cluster> adaptiveDBSCANCluster(const std::vector<Point3D>& points, const rclcpp::Time& timestamp);
    void trackClusters(std::vector<Cluster>& clusters);
    void enhancedTrackClusters(std::vector<Cluster>& clusters);  // 改善された追跡
    void classifyClusters(std::vector<Cluster>& clusters);
    
    // 出力関数
    void publishObstacles(const std::vector<Cluster>& clusters, const rclcpp::Time& timestamp);
    visualization_msgs::msg::MarkerArray createMarkerArray(
        const std::vector<Cluster>& clusters, bool is_dynamic, const rclcpp::Time& timestamp);
    visualization_msgs::msg::MarkerArray createConnectedMarkerArray(
        const std::vector<Cluster>& clusters, bool is_dynamic, const rclcpp::Time& timestamp);
    void addClearMarkers(visualization_msgs::msg::MarkerArray& marker_array, 
                        const std::string& namespace_name, const rclcpp::Time& timestamp);
    std::vector<Point3D> fillGapsInCluster(const Cluster& cluster);
    
    // ユーティリティ関数
    double calculateDistance(const Point3D& p1, const Point3D& p2);
    Point3D calculateCentroid(const std::vector<Point3D>& points);
    bool isWithinProcessingRange(const Point3D& point);
    Point3D transformPointToMap(const Point3D& point, const std::string& source_frame, 
                               const rclcpp::Time& stamp);
    Point3D getRobotVelocity();
    
    // パラメータ検証
    void validateParameters();
    
    // 改善された追跡関連
    double calculateMatchingScore(const Cluster& prev, const Cluster& curr, double dt);
    Point3D smoothVelocity(const Point3D& prev_vel, const Point3D& measured_vel, double alpha);
    void updateClusterConfidence(Cluster& cluster, bool matched);
    
    // Adaptive DBSCAN関連
    double getAdaptiveEps(const Point3D& point);
    double calculateAngularDifference(const Point3D& p1, const Point3D& p2);
    std::vector<int> getNeighbors(int point_idx, const std::vector<Point3D>& points);
    bool isCore(int point_idx, const std::vector<Point3D>& points);
    
    // スライディングウィンドウ関連メソッド
    void updateSlidingWindow(const std::vector<Point3D>& points, const rclcpp::Time& timestamp);
    std::vector<Point3D> getAggregatedPoints() const;
    void cleanupOldObservations(const rclcpp::Time& current_time);
    void trimMarkerHistory(std::deque<std::vector<visualization_msgs::msg::Marker>>& history);
    void appendHistoryMarkers(
        const std::deque<std::vector<visualization_msgs::msg::Marker>>& history,
        visualization_msgs::msg::MarkerArray& marker_array);
    
    
    // ROS2要素
    rclcpp::Subscription<sensor_msgs::msg::LaserScan>::SharedPtr scan_subscriber_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr dynamic_obstacles_publisher_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr static_obstacles_publisher_;
    
    // パラメータ
    double robot_processing_range_;
    double dynamic_cluster_speed_threshold_;
    double voxel_size_;
    double clustering_max_distance_;
    int min_cluster_points_;
    double processing_frequency_;
    double max_computation_time_;
    
    // Adaptive DBSCAN パラメータ
    double dbscan_base_eps_;         // 基本近傍半径
    int dbscan_min_points_;          // 最小点数
    double distance_adaptive_factor_; // 距離適応係数
    double max_angular_difference_;   // 最大角度差 [rad]
    bool enable_adaptive_dbscan_;     // Adaptive DBSCAN有効/無効
    
    
    // 改善された追跡パラメータ
    bool enable_enhanced_tracking_;   // 改善された追跡の有効/無効
    double velocity_smoothing_alpha_; // 速度平滑化係数 (0.0-1.0)
    double confidence_decay_rate_;    // 信頼度減衰率
    int max_missing_frames_;          // 最大欠測フレーム数
    
    // スライディングウィンドウパラメータ
    bool enable_sliding_window_;      // スライディングウィンドウの有効/無効
    int sliding_window_size_;         // 保持するフレーム数
    int min_voxel_observations_;      // ボクセルが有効と判定される最小観測回数
    
    // パラメータ検証用定数
    static constexpr double MIN_PROCESSING_RANGE = 0.1;
    static constexpr double MAX_PROCESSING_RANGE = 50.0;
    static constexpr double MIN_VOXEL_SIZE = 0.01;
    static constexpr double MAX_VOXEL_SIZE = 2.0;
    static constexpr double MIN_CLUSTERING_DISTANCE = 0.01;
    static constexpr double MAX_CLUSTERING_DISTANCE = 5.0;
    static constexpr double MAX_SPEED_THRESHOLD = 10.0;
    static constexpr int MIN_CLUSTER_POINTS = 1;
    static constexpr int MAX_CLUSTER_POINTS = 100;
    static constexpr double MIN_PROCESSING_FREQUENCY = 0.1;
    static constexpr double MAX_PROCESSING_FREQUENCY = 100.0;
    static constexpr double MIN_COMPUTATION_TIME = 0.001;
    static constexpr double MAX_COMPUTATION_TIME = 1.0;
    static constexpr double MIN_DBSCAN_EPS = 0.01;
    static constexpr double MAX_DBSCAN_EPS = 2.0;
    static constexpr int MIN_DBSCAN_POINTS = 1;
    static constexpr int MAX_DBSCAN_POINTS = 50;
    static constexpr double MAX_ADAPTIVE_FACTOR = 1.0;
    
    // TF関連
    std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
    std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
    
    // 内部状態
    std::map<int, Cluster> previous_clusters_;
    int next_cluster_id_;
    rclcpp::Time last_scan_time_;
    
    // スライディングウィンドウ用データ構造
    struct VoxelObservation {
        Point3D position;
        rclcpp::Time timestamp;
        double angle;  // LiDAR角度情報
    };
    
    std::deque<std::vector<Point3D>> point_history_;  // 過去数フレームの生点群
    std::deque<rclcpp::Time> frame_timestamps_;       // 各フレームのタイムスタンプ
    std::map<std::tuple<int, int, int>, std::vector<VoxelObservation>> voxel_history_;  // ボクセルごとの観測履歴
    
    // ロボット速度追跡
    Point3D previous_robot_position_;
    rclcpp::Time previous_robot_time_;
    Point3D current_robot_velocity_;
    bool robot_velocity_initialized_;
    
    // マーカー履歴（最新sliding_window_sizeフレーム分を保持）
    std::deque<std::vector<visualization_msgs::msg::Marker>> dynamic_marker_history_;
    std::deque<std::vector<visualization_msgs::msg::Marker>> static_marker_history_;
};

} // namespace obstacle_tracker

#endif // OBSTACLE_TRACKER_NODE_HPP_

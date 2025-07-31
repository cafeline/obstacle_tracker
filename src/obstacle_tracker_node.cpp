#include "obstacle_tracker/obstacle_tracker_node.hpp"
#include <cmath>
#include <algorithm>
#include <chrono>
#include <unordered_map>
#include <string>
#include <set>
#include <tuple>

namespace obstacle_tracker
{

ObstacleTrackerNode::ObstacleTrackerNode() : Node("obstacle_tracker_node"), next_cluster_id_(1), 
                                           robot_velocity_initialized_(false)
{
    // TFバッファとリスナーの初期化
    tf_buffer_ = std::make_shared<tf2_ros::Buffer>(this->get_clock());
    tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);
    
    // ロボット速度追跡の初期化
    current_robot_velocity_ = Point3D(0, 0, 0);
    
    // パラメータの宣言とデフォルト値設定
    this->declare_parameter("robot_processing_range", 10.0);
    this->declare_parameter("dynamic_cluster_speed_threshold", 1.5);
    this->declare_parameter("voxel_size", 0.2);
    this->declare_parameter("clustering_max_distance", 0.3);
    this->declare_parameter("min_cluster_points", 2);
    this->declare_parameter("processing_frequency", 20.0);
    this->declare_parameter("max_computation_time", 0.05);
    
    // Adaptive DBSCAN パラメータ
    this->declare_parameter("enable_adaptive_dbscan", true);
    this->declare_parameter("dbscan_base_eps", 0.3);
    this->declare_parameter("dbscan_min_points", 3);
    this->declare_parameter("distance_adaptive_factor", 0.05);
    this->declare_parameter("max_angular_difference", 0.2);
    
    
    // 改善された追跡パラメータ
    this->declare_parameter("enable_enhanced_tracking", true);
    this->declare_parameter("velocity_smoothing_alpha", 0.3);
    this->declare_parameter("confidence_decay_rate", 0.1);
    this->declare_parameter("max_missing_frames", 5);
    
    // スライディングウィンドウパラメータ
    this->declare_parameter("enable_sliding_window", true);
    this->declare_parameter("sliding_window_size", 3);  // 保持するフレーム数
    this->declare_parameter("min_voxel_observations", 2);  // ボクセルが有効と判定される最小観測回数
    
    // パラメータの取得
    robot_processing_range_ = this->get_parameter("robot_processing_range").as_double();
    dynamic_cluster_speed_threshold_ = this->get_parameter("dynamic_cluster_speed_threshold").as_double();
    voxel_size_ = this->get_parameter("voxel_size").as_double();
    clustering_max_distance_ = this->get_parameter("clustering_max_distance").as_double();
    min_cluster_points_ = this->get_parameter("min_cluster_points").as_int();
    processing_frequency_ = this->get_parameter("processing_frequency").as_double();
    max_computation_time_ = this->get_parameter("max_computation_time").as_double();
    
    // Adaptive DBSCAN パラメータ取得
    enable_adaptive_dbscan_ = this->get_parameter("enable_adaptive_dbscan").as_bool();
    dbscan_base_eps_ = this->get_parameter("dbscan_base_eps").as_double();
    dbscan_min_points_ = this->get_parameter("dbscan_min_points").as_int();
    distance_adaptive_factor_ = this->get_parameter("distance_adaptive_factor").as_double();
    max_angular_difference_ = this->get_parameter("max_angular_difference").as_double();
    
    
    // 改善された追跡パラメータ取得
    enable_enhanced_tracking_ = this->get_parameter("enable_enhanced_tracking").as_bool();
    velocity_smoothing_alpha_ = this->get_parameter("velocity_smoothing_alpha").as_double();
    confidence_decay_rate_ = this->get_parameter("confidence_decay_rate").as_double();
    max_missing_frames_ = this->get_parameter("max_missing_frames").as_int();
    
    // スライディングウィンドウパラメータ取得
    enable_sliding_window_ = this->get_parameter("enable_sliding_window").as_bool();
    sliding_window_size_ = this->get_parameter("sliding_window_size").as_int();
    min_voxel_observations_ = this->get_parameter("min_voxel_observations").as_int();
    
    // パラメータ検証
    validateParameters();
    
    // サブスクライバーとパブリッシャーの作成
    scan_subscriber_ = this->create_subscription<sensor_msgs::msg::LaserScan>(
        "/scan", 10, std::bind(&ObstacleTrackerNode::laserScanCallback, this, std::placeholders::_1));
    
    dynamic_obstacles_publisher_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(
        "/dynamic_obstacles", 10);
    
    static_obstacles_publisher_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(
        "/static_obstacles", 10);
    
    
    RCLCPP_INFO(this->get_logger(), "ObstacleTracker node initialized with validated parameters");
    RCLCPP_INFO(this->get_logger(), "Processing range: %.2f m", robot_processing_range_);
    RCLCPP_INFO(this->get_logger(), "Dynamic threshold: %.2f m/s", dynamic_cluster_speed_threshold_);
    RCLCPP_INFO(this->get_logger(), "Processing frequency: %.1f Hz", processing_frequency_);
    RCLCPP_INFO(this->get_logger(), "Max computation time: %.3f s", max_computation_time_);
    RCLCPP_INFO(this->get_logger(), "Adaptive DBSCAN: %s", enable_adaptive_dbscan_ ? "enabled" : "disabled");
    if (enable_adaptive_dbscan_) {
        RCLCPP_INFO(this->get_logger(), "DBSCAN params - base_eps: %.2f, min_points: %d, adaptive_factor: %.3f", 
                   dbscan_base_eps_, dbscan_min_points_, distance_adaptive_factor_);
    }
    RCLCPP_INFO(this->get_logger(), "Sliding window: %s (size: %d frames, min_obs: %d)", 
               enable_sliding_window_ ? "enabled" : "disabled", 
               sliding_window_size_, min_voxel_observations_);
}

void ObstacleTrackerNode::validateParameters()
{
    bool params_valid = true;
    std::vector<std::string> error_messages;
    std::vector<std::string> warning_messages;
    
    // 距離関連パラメータの検証（定数使用）
    if (robot_processing_range_ <= MIN_PROCESSING_RANGE || robot_processing_range_ > MAX_PROCESSING_RANGE) {
        error_messages.push_back("Invalid robot_processing_range: " + std::to_string(robot_processing_range_) + 
                               " (must be " + std::to_string(MIN_PROCESSING_RANGE) + " < value <= " + std::to_string(MAX_PROCESSING_RANGE) + ")");
        params_valid = false;
    }
    
    if (voxel_size_ <= MIN_VOXEL_SIZE || voxel_size_ > MAX_VOXEL_SIZE) {
        error_messages.push_back("Invalid voxel_size: " + std::to_string(voxel_size_) + 
                               " (must be " + std::to_string(MIN_VOXEL_SIZE) + " < value <= " + std::to_string(MAX_VOXEL_SIZE) + ")");
        params_valid = false;
    }
    
    if (clustering_max_distance_ <= MIN_CLUSTERING_DISTANCE || clustering_max_distance_ > MAX_CLUSTERING_DISTANCE) {
        error_messages.push_back("Invalid clustering_max_distance: " + std::to_string(clustering_max_distance_) + 
                               " (must be " + std::to_string(MIN_CLUSTERING_DISTANCE) + " < value <= " + std::to_string(MAX_CLUSTERING_DISTANCE) + ")");
        params_valid = false;
    }
    
    // 速度関連パラメータの検証
    if (dynamic_cluster_speed_threshold_ < 0.0 || dynamic_cluster_speed_threshold_ > MAX_SPEED_THRESHOLD) {
        error_messages.push_back("Invalid dynamic_cluster_speed_threshold: " + std::to_string(dynamic_cluster_speed_threshold_) + 
                               " (must be 0 <= value <= " + std::to_string(MAX_SPEED_THRESHOLD) + ")");
        params_valid = false;
    }
    
    // 整数パラメータの検証
    if (min_cluster_points_ < MIN_CLUSTER_POINTS || min_cluster_points_ > MAX_CLUSTER_POINTS) {
        error_messages.push_back("Invalid min_cluster_points: " + std::to_string(min_cluster_points_) + 
                               " (must be " + std::to_string(MIN_CLUSTER_POINTS) + " <= value <= " + std::to_string(MAX_CLUSTER_POINTS) + ")");
        params_valid = false;
    }
    
    if (dbscan_min_points_ < MIN_DBSCAN_POINTS || dbscan_min_points_ > MAX_DBSCAN_POINTS) {
        error_messages.push_back("Invalid dbscan_min_points: " + std::to_string(dbscan_min_points_) + 
                               " (must be " + std::to_string(MIN_DBSCAN_POINTS) + " <= value <= " + std::to_string(MAX_DBSCAN_POINTS) + ")");
        params_valid = false;
    }
    
    // 頻度・時間関連パラメータの検証
    if (processing_frequency_ <= MIN_PROCESSING_FREQUENCY || processing_frequency_ > MAX_PROCESSING_FREQUENCY) {
        error_messages.push_back("Invalid processing_frequency: " + std::to_string(processing_frequency_) + 
                               " (must be " + std::to_string(MIN_PROCESSING_FREQUENCY) + " < value <= " + std::to_string(MAX_PROCESSING_FREQUENCY) + ")");
        params_valid = false;
    }
    
    if (max_computation_time_ <= MIN_COMPUTATION_TIME || max_computation_time_ > MAX_COMPUTATION_TIME) {
        error_messages.push_back("Invalid max_computation_time: " + std::to_string(max_computation_time_) + 
                               " (must be " + std::to_string(MIN_COMPUTATION_TIME) + " < value <= " + std::to_string(MAX_COMPUTATION_TIME) + ")");
        params_valid = false;
    }
    
    // DBSCAN関連パラメータの検証
    if (dbscan_base_eps_ <= MIN_DBSCAN_EPS || dbscan_base_eps_ > MAX_DBSCAN_EPS) {
        error_messages.push_back("Invalid dbscan_base_eps: " + std::to_string(dbscan_base_eps_) + 
                               " (must be " + std::to_string(MIN_DBSCAN_EPS) + " < value <= " + std::to_string(MAX_DBSCAN_EPS) + ")");
        params_valid = false;
    }
    
    if (distance_adaptive_factor_ < 0.0 || distance_adaptive_factor_ > MAX_ADAPTIVE_FACTOR) {
        error_messages.push_back("Invalid distance_adaptive_factor: " + std::to_string(distance_adaptive_factor_) + 
                               " (must be 0 <= value <= " + std::to_string(MAX_ADAPTIVE_FACTOR) + ")");
        params_valid = false;
    }
    
    if (max_angular_difference_ < 0.0 || max_angular_difference_ > M_PI) {
        error_messages.push_back("Invalid max_angular_difference: " + std::to_string(max_angular_difference_) + 
                               " (must be 0 <= value <= π)");
        params_valid = false;
    }
    
    // スライディングウィンドウパラメータの検証
    if (sliding_window_size_ < 1 || sliding_window_size_ > 20) {
        error_messages.push_back("Invalid sliding_window_size: " + std::to_string(sliding_window_size_) +
                               " (must be 1 <= value <= 20)");
        params_valid = false;
    }
    
    if (min_voxel_observations_ < 1 || min_voxel_observations_ > sliding_window_size_) {
        error_messages.push_back("Invalid min_voxel_observations: " + std::to_string(min_voxel_observations_) +
                               " (must be 1 <= value <= sliding_window_size)");
        params_valid = false;
    }
    
    // パラメータ整合性チェック（警告）
    if (clustering_max_distance_ < voxel_size_) {
        warning_messages.push_back("clustering_max_distance (" + std::to_string(clustering_max_distance_) + 
                                 ") < voxel_size (" + std::to_string(voxel_size_) + ") may cause clustering issues");
    }
    
    if (enable_adaptive_dbscan_ && dbscan_base_eps_ < voxel_size_) {
        warning_messages.push_back("dbscan_base_eps (" + std::to_string(dbscan_base_eps_) + 
                                 ") < voxel_size (" + std::to_string(voxel_size_) + ") may cause clustering issues");
    }
    
    if (enable_adaptive_dbscan_ && dbscan_min_points_ < min_cluster_points_) {
        warning_messages.push_back("dbscan_min_points (" + std::to_string(dbscan_min_points_) + 
                                 ") < min_cluster_points (" + std::to_string(min_cluster_points_) + ") may cause inconsistent results");
    }
    
    // パフォーマンス関連警告
    if (processing_frequency_ > 50.0) {
        warning_messages.push_back("High processing_frequency (" + std::to_string(processing_frequency_) + 
                                 " Hz) may cause performance issues");
    }
    
    if (max_computation_time_ < 0.01) {
        warning_messages.push_back("Very low max_computation_time (" + std::to_string(max_computation_time_) + 
                                 " s) may cause frequent warnings");
    }
    
    // エラーメッセージの出力
    for (const auto& error : error_messages) {
        RCLCPP_ERROR(this->get_logger(), "%s", error.c_str());
    }
    
    // 警告メッセージの出力
    for (const auto& warning : warning_messages) {
        RCLCPP_WARN(this->get_logger(), "%s", warning.c_str());
    }
    
    if (!params_valid) {
        RCLCPP_FATAL(this->get_logger(), "Parameter validation failed with %zu errors. Node will not function correctly.", 
                    error_messages.size());
        throw std::invalid_argument("Invalid parameters detected: " + std::to_string(error_messages.size()) + " errors found");
    }
    
    RCLCPP_INFO(this->get_logger(), "All parameters validated successfully (%zu warnings)", warning_messages.size());
}

void ObstacleTrackerNode::laserScanCallback(const sensor_msgs::msg::LaserScan::SharedPtr msg)
{
    try {
        // 処理頻度制限チェック（秒単位で統一管理）
        rclcpp::Time current_time = this->now();
        double current_time_sec = current_time.seconds();
        static double last_processing_time_sec = 0.0;
        
        if (last_processing_time_sec > 0.0) {
            double dt = current_time_sec - last_processing_time_sec;
            double min_interval = 1.0 / processing_frequency_;
            
            if (dt < min_interval) {
                // 処理頻度制限により処理をスキップ
                RCLCPP_DEBUG(this->get_logger(), 
                    "Processing skipped due to frequency limit: dt=%.3f, min_interval=%.3f", 
                    dt, min_interval);
                return;
            }
        }
        
        // 処理時間計測開始
        auto start_time = std::chrono::high_resolution_clock::now();
        
        // スキャン時間を保存（現在時刻を使用して時間ソース統一）
        last_scan_time_ = current_time;
        
        // 1. レーザースキャンを点群に変換
        std::vector<Point3D> points = convertScanToPoints(msg);
        
        // 2. ボクセル化
        std::vector<Point3D> voxelized_points = voxelizePoints(points);
        
        // 3. スライディングウィンドウ処理と点群集約
        std::vector<Point3D> aggregated_points;
        
        if (enable_sliding_window_) {
            // 3-1. スライディングウィンドウに点群を追加
            updateSlidingWindow(voxelized_points, current_time);
            
            // 3-2. 集約された点群を取得
            aggregated_points = getAggregatedPoints();
        } else {
            aggregated_points = voxelized_points;
        }
        
        // 4. クラスタリング（集約された点群で実行）
        std::vector<Cluster> clusters = enable_adaptive_dbscan_ ? 
            adaptiveDBSCANCluster(aggregated_points, current_time) : clusterPoints(aggregated_points, current_time);
        
        // 5. クラスタ追跡
        if (enable_enhanced_tracking_) {
            enhancedTrackClusters(clusters);
        } else {
            trackClusters(clusters);
        }
        
        // 6. 動的・静的分類
        classifyClusters(clusters);
        
        // 6. 結果を配信（時刻を統一して渡す）
        publishObstacles(clusters, current_time);
        
        // 処理時間計測終了
        auto end_time = std::chrono::high_resolution_clock::now();
        auto computation_time = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        double computation_time_sec = computation_time.count() / 1000000.0;
        
        // 処理時間監視
        if (computation_time_sec > max_computation_time_) {
            RCLCPP_WARN(this->get_logger(), 
                "Processing time exceeded limit: %.3f s (limit: %.3f s)", 
                computation_time_sec, max_computation_time_);
        } else {
            RCLCPP_DEBUG(this->get_logger(), 
                "Processing completed in %.3f s", computation_time_sec);
        }
        
        // 最後の処理時間を更新（秒単位）
        last_processing_time_sec = current_time_sec;
        
    } catch (const std::exception& e) {
        RCLCPP_ERROR(this->get_logger(), "Error in laser scan processing: %s", e.what());
    }
}

std::vector<Point3D> ObstacleTrackerNode::convertScanToPoints(const sensor_msgs::msg::LaserScan::SharedPtr scan)
{
    std::vector<Point3D> points;
    
    // スキャンデータの妥当性チェック
    if (!scan || scan->ranges.empty()) {
        RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 5000,
            "Invalid or empty laser scan data received");
        return points;
    }
    
    // パラメータの妥当性チェック
    if (scan->angle_increment <= 0.0 || scan->range_max <= scan->range_min) {
        RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 5000,
            "Invalid laser scan parameters: angle_inc=%.6f, range_min=%.2f, range_max=%.2f",
            scan->angle_increment, scan->range_min, scan->range_max);
        return points;
    }
    
    points.reserve(scan->ranges.size()); // メモリ効率改善
    
    for (size_t i = 0; i < scan->ranges.size(); ++i) {
        double range = scan->ranges[i];
        
        // 範囲外や無効な値をスキップ（厳密チェック）
        if (range < scan->range_min || range > scan->range_max || 
            !std::isfinite(range) || range <= 0.0) {
            continue;
        }
        
        // 角度計算
        double angle = scan->angle_min + static_cast<double>(i) * scan->angle_increment;
        
        // 角度の妥当性チェック
        if (!std::isfinite(angle)) {
            continue;
        }
        
        // 直交座標に変換
        double cos_angle = std::cos(angle);
        double sin_angle = std::sin(angle);
        
        Point3D point;
        point.x = range * cos_angle;
        point.y = range * sin_angle;
        point.z = 0.0;
        point.angle = angle;  // 元の角度情報を保持
        
        // 変換結果の妥当性チェック
        if (!std::isfinite(point.x) || !std::isfinite(point.y)) {
            continue;
        }
        
        // 処理範囲内かチェック
        if (isWithinProcessingRange(point)) {
            points.push_back(point);
        }
    }
    
    return points;
}

std::vector<Point3D> ObstacleTrackerNode::voxelizePoints(const std::vector<Point3D>& points)
{
    // メモリ効率改善: unordered_mapとreserve使用
    std::unordered_map<std::string, std::pair<Point3D, int>> voxel_map;
    voxel_map.reserve(points.size() / 2); // 予想サイズでreserve
    
    const double inv_voxel_size = 1.0 / voxel_size_; // 除算回避
    
    for (const auto& point : points) {
        // ボクセルグリッドのインデックス計算（除算回避）
        int vx = static_cast<int>(std::floor(point.x * inv_voxel_size));
        int vy = static_cast<int>(std::floor(point.y * inv_voxel_size));
        int vz = static_cast<int>(std::floor(point.z * inv_voxel_size));
        
        // 文字列キーで高速アクセス（tupleより高速）
        std::string key = std::to_string(vx) + "," + std::to_string(vy) + "," + std::to_string(vz);
        
        auto it = voxel_map.find(key);
        if (it != voxel_map.end()) {
            // 既存ボクセル: 累積平均で効率的に更新
            auto& [sum_point, count] = it->second;
            sum_point.x = (sum_point.x * count + point.x) / (count + 1);
            sum_point.y = (sum_point.y * count + point.y) / (count + 1);
            sum_point.z = (sum_point.z * count + point.z) / (count + 1);
            // 角度の平均計算（円周性を考慮した単純平均）
            sum_point.angle = (sum_point.angle * count + point.angle) / (count + 1);
            count++;
        } else {
            // 新規ボクセル
            voxel_map.emplace(std::move(key), std::make_pair(point, 1));
        }
    }
    
    // 結果を効率的に構築
    std::vector<Point3D> voxelized_points;
    voxelized_points.reserve(voxel_map.size());
    
    for (const auto& [key, value] : voxel_map) {
        voxelized_points.push_back(value.first);
    }
    
    RCLCPP_DEBUG(this->get_logger(), 
        "Voxelization: %zu input points -> %zu voxels (voxel_size=%.2fm)", 
        points.size(), voxelized_points.size(), voxel_size_);
    
    return voxelized_points;
}

std::vector<Cluster> ObstacleTrackerNode::clusterPoints(const std::vector<Point3D>& voxelized_points, const rclcpp::Time& timestamp)
{
    std::vector<Cluster> clusters;
    std::vector<bool> visited(voxelized_points.size(), false);
    
    for (size_t i = 0; i < voxelized_points.size(); ++i) {
        if (visited[i]) continue;
        
        Cluster cluster;
        cluster.id = next_cluster_id_++;
        std::vector<size_t> cluster_indices;
        
        // BFS (幅優先探索) を使用してクラスタを成長（高速化）
        std::vector<size_t> queue;
        queue.reserve(100); // 予想サイズでreserve
        queue.push_back(i);
        visited[i] = true;
        
        const double clustering_distance_squared = clustering_max_distance_ * clustering_max_distance_;
        
        while (!queue.empty()) {
            size_t current_idx = queue.back();
            queue.pop_back();
            cluster_indices.push_back(current_idx);
            cluster.points.push_back(voxelized_points[current_idx]);
            
            const Point3D& current_point = voxelized_points[current_idx];
            
            // 近隣点を探索（平方距離で高速化）
            for (size_t j = 0; j < voxelized_points.size(); ++j) {
                if (visited[j]) continue;
                
                const Point3D& candidate = voxelized_points[j];
                double dx = current_point.x - candidate.x;
                double dy = current_point.y - candidate.y;
                double dz = current_point.z - candidate.z;
                double distance_squared = dx * dx + dy * dy + dz * dz;
                
                if (distance_squared <= clustering_distance_squared) {
                    visited[j] = true;
                    queue.push_back(j);
                }
            }
        }
        
        // 最小点数チェック （デバッグ情報追加）
        if (static_cast<int>(cluster.points.size()) >= min_cluster_points_) {
            Point3D laser_centroid = calculateCentroid(cluster.points);
            // lidar_linkフレームからmapフレームに変換（統一された時刻使用）
            Point3D map_centroid = transformPointToMap(laser_centroid, "lidar_link", timestamp);
            
            // TF変換が有効かチェック
            if (std::isfinite(map_centroid.x) && std::isfinite(map_centroid.y) && std::isfinite(map_centroid.z)) {
                cluster.centroid = map_centroid;
                clusters.push_back(cluster);
                
                RCLCPP_DEBUG(this->get_logger(), 
                    "Cluster %d: %zu points, centroid=(%.2f,%.2f)", 
                    cluster.id, cluster.points.size(), cluster.centroid.x, cluster.centroid.y);
            } else {
                RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 5000,
                    "Invalid cluster centroid after TF transform, skipping cluster");
            }
        } else {
            RCLCPP_DEBUG(this->get_logger(), 
                "Cluster rejected: %zu points < min_cluster_points(%d)", 
                cluster.points.size(), min_cluster_points_);
        }
    }
    
    RCLCPP_DEBUG(this->get_logger(), 
        "Simple clustering: %zu voxels -> %zu clusters (threshold=%.2fm, min_points=%d)", 
        voxelized_points.size(), clusters.size(), clustering_max_distance_, min_cluster_points_);
    
    return clusters;
}

std::vector<Cluster> ObstacleTrackerNode::adaptiveDBSCANCluster(const std::vector<Point3D>& points, const rclcpp::Time& timestamp)
{
    std::vector<Cluster> clusters;
    if (points.empty()) return clusters;
    
    std::vector<int> labels(points.size(), -2);  // -2: 未処理, -1: ノイズ, 0以上: クラスタID
    int cluster_id = 0;
    
    for (size_t i = 0; i < points.size(); ++i) {
        if (labels[i] != -2) continue;  // 既に処理済み
        
        std::vector<int> neighbors = getNeighbors(i, points);
        
        if (neighbors.size() < static_cast<size_t>(dbscan_min_points_)) {
            labels[i] = -1;  // ノイズ点としてマーク
            continue;
        }
        
        // 新しいクラスタを開始
        labels[i] = cluster_id;
        
        // 近傍点を探索（幅優先探索）
        std::vector<int> seed_set = neighbors;
        for (size_t j = 0; j < seed_set.size(); ++j) {
            int point_idx = seed_set[j];
            
            if (labels[point_idx] == -1) {
                labels[point_idx] = cluster_id;  // ノイズ点をクラスタに変更
            }
            
            if (labels[point_idx] != -2) continue;  // 既に処理済み
            
            labels[point_idx] = cluster_id;
            
            std::vector<int> point_neighbors = getNeighbors(point_idx, points);
            if (point_neighbors.size() >= static_cast<size_t>(dbscan_min_points_)) {
                // コア点の場合、その近傍点を探索対象に追加
                for (int neighbor : point_neighbors) {
                    if (std::find(seed_set.begin(), seed_set.end(), neighbor) == seed_set.end()) {
                        seed_set.push_back(neighbor);
                    }
                }
            }
        }
        cluster_id++;
    }
    
    // クラスタを構築
    std::map<int, std::vector<int>> cluster_map;
    for (size_t i = 0; i < points.size(); ++i) {
        if (labels[i] >= 0) {  // ノイズでない点のみ
            cluster_map[labels[i]].push_back(i);
        }
    }
    
    for (const auto& pair : cluster_map) {
        const auto& point_indices = pair.second;
        if (point_indices.size() >= static_cast<size_t>(dbscan_min_points_)) {
            Cluster cluster;
            cluster.id = next_cluster_id_++;
            
            for (int idx : point_indices) {
                cluster.points.push_back(points[idx]);
            }
            
            Point3D laser_centroid = calculateCentroid(cluster.points);
            cluster.centroid = transformPointToMap(laser_centroid, "lidar_link", timestamp);
            clusters.push_back(cluster);
        }
    }
    
    RCLCPP_DEBUG(this->get_logger(), "Adaptive DBSCAN: %zu points -> %zu clusters (noise filtered)", 
                points.size(), clusters.size());
    
    return clusters;
}

void ObstacleTrackerNode::trackClusters(std::vector<Cluster>& clusters)
{
    // ロボットの現在速度を取得（キャッシュ）
    static Point3D cached_robot_velocity(0, 0, 0);
    static rclcpp::Time last_velocity_update;
    
    rclcpp::Time current_time = this->now();
    // 時間ソース統一のため、秒単位で時間管理
    static double last_velocity_update_sec = 0.0;
    double current_time_sec = current_time.seconds();
    
    if (last_velocity_update_sec == 0.0 || 
        (current_time_sec - last_velocity_update_sec) > 0.05) { // 20Hzで更新
        cached_robot_velocity = getRobotVelocity();
        last_velocity_update_sec = current_time_sec;
    }
    
    const double match_threshold_squared = std::pow(clustering_max_distance_ * 2.0, 2); // 平方距離で比較
    
    // シンプルな最近隣マッチングによる追跡（最適化）
    for (auto& cluster : clusters) {
        double min_distance_squared = std::numeric_limits<double>::max();
        int best_match_id = -1;
        
        // 前フレームのクラスタと比較（平方距離で高速化）
        for (const auto& [prev_id, prev_cluster] : previous_clusters_) {
            double dx = cluster.centroid.x - prev_cluster.centroid.x;
            double dy = cluster.centroid.y - prev_cluster.centroid.y;
            double dz = cluster.centroid.z - prev_cluster.centroid.z;
            double distance_squared = dx * dx + dy * dy + dz * dz;
            
            if (distance_squared < min_distance_squared && distance_squared <= match_threshold_squared) {
                min_distance_squared = distance_squared;
                best_match_id = prev_cluster.id;
            }
        }
        
        if (best_match_id != -1) {
            // マッチした場合、IDを継承し速度を計算
            cluster.id = best_match_id;
            const auto& prev_cluster = previous_clusters_[best_match_id];
            
            // 時間差分の最適化（秒単位で統一管理）
            double dt = 0.1; // デフォルト値
            double current_time_sec = current_time.seconds();
            double last_scan_time_sec = last_scan_time_.seconds();
            
            // 秒単位で時間対算を実行（クロックタイプ非依存）
            if (last_scan_time_sec > 0.0) {
                dt = current_time_sec - last_scan_time_sec;
            }
            if (dt > 0.001 && dt < 1.0) { // 怪しい時間差分のみ使用
                Point3D relative_velocity;
                relative_velocity.x = (cluster.centroid.x - prev_cluster.centroid.x) / dt;
                relative_velocity.y = (cluster.centroid.y - prev_cluster.centroid.y) / dt;
                relative_velocity.z = 0.0;
                
                // ロボットの移動速度を差し引いて絶対速度を計算
                cluster.velocity.x = relative_velocity.x - cached_robot_velocity.x;
                cluster.velocity.y = relative_velocity.y - cached_robot_velocity.y;
                cluster.velocity.z = 0.0;
            } else {
                // 異常な時間差分の場合は前回の速度を使用
                cluster.velocity = prev_cluster.velocity;
            }
            
            cluster.track_count = prev_cluster.track_count + 1;
            
            RCLCPP_DEBUG(this->get_logger(), 
                "Cluster %d: dt=%.3f, robot_vel=(%.2f,%.2f), absolute_vel=(%.2f,%.2f)",
                cluster.id, dt, cached_robot_velocity.x, cached_robot_velocity.y,
                cluster.velocity.x, cluster.velocity.y);
        } else {
            // 新しいクラスタの場合
            cluster.velocity = Point3D(0, 0, 0);
            cluster.track_count = 1;
        }
    }
    
    // 現在のクラスタを保存（moveセマンティクスで高速化）
    previous_clusters_.clear();
    for (const auto& cluster : clusters) {
        previous_clusters_.emplace(cluster.id, cluster);
    }
}

void ObstacleTrackerNode::classifyClusters(std::vector<Cluster>& clusters)
{
    for (auto& cluster : clusters) {
        // 改善された追跡が有効な場合は平滑化された速度を使用
        Point3D velocity_to_use = enable_enhanced_tracking_ ? cluster.smoothed_velocity : cluster.velocity;
        
        // 速度ベクトルの大きさを計算
        double speed = std::sqrt(velocity_to_use.x * velocity_to_use.x +
                                velocity_to_use.y * velocity_to_use.y);
        
        // しきい値と比較して動的・静的を判別
        cluster.is_dynamic = (speed > dynamic_cluster_speed_threshold_);
        
        RCLCPP_DEBUG(this->get_logger(), 
            "Cluster %d: speed=%.2f m/s, dynamic=%s, confidence=%.2f", 
            cluster.id, speed, cluster.is_dynamic ? "true" : "false", cluster.confidence);
    }
}

void ObstacleTrackerNode::publishObstacles(const std::vector<Cluster>& clusters, const rclcpp::Time& timestamp)
{
    std::vector<Cluster> dynamic_clusters, static_clusters;
    
    // 動的・静的クラスタを分離
    for (const auto& cluster : clusters) {
        if (cluster.is_dynamic) {
            dynamic_clusters.push_back(cluster);
        } else {
            static_clusters.push_back(cluster);
        }
    }
    
    // 連結されたマーカー配列を作成（隙間を埋める）
    auto dynamic_markers = createConnectedMarkerArray(dynamic_clusters, true, timestamp);
    auto static_markers = createConnectedMarkerArray(static_clusters, false, timestamp);
    
    // 古いマーカーをクリアするためのマーカーを追加
    addClearMarkers(dynamic_markers, "dynamic_obstacles", timestamp);
    addClearMarkers(static_markers, "static_obstacles", timestamp);
    
    dynamic_obstacles_publisher_->publish(dynamic_markers);
    static_obstacles_publisher_->publish(static_markers);
    
    
    RCLCPP_DEBUG(this->get_logger(), "Published %zu dynamic and %zu static clusters",
                dynamic_clusters.size(), static_clusters.size());
}

visualization_msgs::msg::MarkerArray ObstacleTrackerNode::createConnectedMarkerArray(
    const std::vector<Cluster>& clusters, bool is_dynamic, const rclcpp::Time& timestamp)
{
    visualization_msgs::msg::MarkerArray marker_array;
    int marker_id = 1; // 0はクリアマーカー用に予約
    
    // 全クラスタにわたって重複チェック用のset
    std::set<std::tuple<int, int, int>> global_unique_voxels;
    
    for (const auto& cluster : clusters) {
        // クラスタ内の点を連結して隙間を埋める
        auto connected_points = fillGapsInCluster(cluster);
        
        // 連結された各点に対してマーカーを作成
        for (const auto& point : connected_points) {
            // スライディングウィンドウ有効時は既にmapフレーム座標、無効時は変換が必要
            Point3D map_point;
            if (enable_sliding_window_) {
                map_point = point;  // 既にmapフレーム座標
            } else {
                map_point = transformPointToMap(point, "lidar_link", timestamp);
            }
            
            // ボクセル位置で重複チェック（mapフレーム座標で）
            int vx = static_cast<int>(std::round(map_point.x / voxel_size_));
            int vy = static_cast<int>(std::round(map_point.y / voxel_size_));
            int vz = static_cast<int>(std::round(map_point.z / voxel_size_));
            
            auto voxel_key = std::make_tuple(vx, vy, vz);
            if (global_unique_voxels.find(voxel_key) != global_unique_voxels.end()) {
                continue; // 既に存在するボクセル位置はスキップ
            }
            global_unique_voxels.insert(voxel_key);
            
            visualization_msgs::msg::Marker marker;
            marker.header.frame_id = "map";
            marker.header.stamp = timestamp;
            marker.ns = is_dynamic ? "dynamic_obstacles" : "static_obstacles";
            marker.id = marker_id++;
            marker.type = visualization_msgs::msg::Marker::CUBE;
            marker.action = visualization_msgs::msg::Marker::ADD;
            
            // 位置設定（ボクセルグリッドの中心に配置）
            marker.pose.position.x = vx * voxel_size_;
            marker.pose.position.y = vy * voxel_size_;
            marker.pose.position.z = vz * voxel_size_;
            marker.pose.orientation.w = 1.0;
            
            // サイズ設定
            marker.scale.x = voxel_size_;
            marker.scale.y = voxel_size_;
            marker.scale.z = voxel_size_;
            
            // 色設定
            if (is_dynamic) {
                marker.color.r = 1.0; // 動的障害物は赤
                marker.color.g = 0.0;
                marker.color.b = 0.0;
                marker.color.a = 0.7;
            } else {
                marker.color.r = 0.0; // 静的障害物は青
                marker.color.g = 0.0;
                marker.color.b = 1.0;
                marker.color.a = 0.7;
            }
            
            marker.lifetime = rclcpp::Duration::from_seconds(1.2);
            marker_array.markers.push_back(marker);
        }
    }
    
    RCLCPP_DEBUG(this->get_logger(), "Created %zu unique voxel markers for %zu clusters (%s)", 
                marker_array.markers.size(), clusters.size(), 
                is_dynamic ? "dynamic" : "static");
    
    return marker_array;
}

std::vector<Point3D> ObstacleTrackerNode::fillGapsInCluster(const Cluster& cluster)
{
    if (cluster.points.size() < 2) {
        return cluster.points; // 点が少なすぎる場合はそのまま返す
    }
    
    // 重複排除のためにsetを使用（ボクセルグリッド位置での管理）
    std::set<std::tuple<int, int, int>> unique_voxels;
    std::vector<Point3D> filled_points;
    
    // 元の点をvoxel位置でハッシュ化して追加
    for (const auto& point : cluster.points) {
        int vx = static_cast<int>(std::round(point.x / voxel_size_));
        int vy = static_cast<int>(std::round(point.y / voxel_size_));
        int vz = static_cast<int>(std::round(point.z / voxel_size_));
        
        auto voxel_key = std::make_tuple(vx, vy, vz);
        if (unique_voxels.find(voxel_key) == unique_voxels.end()) {
            unique_voxels.insert(voxel_key);
            filled_points.push_back(point);
        }
    }
    
    // クラスタ内の各点間の隙間を埋める
    for (size_t i = 0; i < cluster.points.size(); ++i) {
        for (size_t j = i + 1; j < cluster.points.size(); ++j) {
            const Point3D& p1 = cluster.points[i];
            const Point3D& p2 = cluster.points[j];
            
            // 2点間の距離を計算
            double distance = std::sqrt(
                (p2.x - p1.x) * (p2.x - p1.x) +
                (p2.y - p1.y) * (p2.y - p1.y) +
                (p2.z - p1.z) * (p2.z - p1.z)
            );
            
            // 隙間がvoxel_sizeの1.5倍より大きい場合、補間点を追加
            double gap_threshold = voxel_size_ * 1.5;
            if (distance > gap_threshold && distance < voxel_size_ * 3.0) { // 最大3倍まで
                // 必要な補間点数を計算
                int num_interpolations = static_cast<int>(std::ceil(distance / voxel_size_)) - 1;
                
                for (int k = 1; k <= num_interpolations; ++k) {
                    double t = static_cast<double>(k) / (num_interpolations + 1);
                    
                    Point3D interpolated_point;
                    interpolated_point.x = p1.x + t * (p2.x - p1.x);
                    interpolated_point.y = p1.y + t * (p2.y - p1.y);
                    interpolated_point.z = p1.z + t * (p2.z - p1.z);
                    
                    // ボクセル位置で重複チェック
                    int vx = static_cast<int>(std::round(interpolated_point.x / voxel_size_));
                    int vy = static_cast<int>(std::round(interpolated_point.y / voxel_size_));
                    int vz = static_cast<int>(std::round(interpolated_point.z / voxel_size_));
                    
                    auto voxel_key = std::make_tuple(vx, vy, vz);
                    if (unique_voxels.find(voxel_key) == unique_voxels.end()) {
                        unique_voxels.insert(voxel_key);
                        filled_points.push_back(interpolated_point);
                    }
                }
            }
        }
    }
    
    RCLCPP_DEBUG(this->get_logger(), "Cluster gap filling: original=%zu, filled=%zu (unique voxels: %zu)", 
                cluster.points.size(), filled_points.size(), unique_voxels.size());
    
    return filled_points;
}

void ObstacleTrackerNode::updateSlidingWindow(const std::vector<Point3D>& points, const rclcpp::Time& timestamp)
{
    // 新しいフレームを履歴に追加
    point_history_.push_back(points);
    frame_timestamps_.push_back(timestamp);
    
    // 古いフレームを削除（指定したウィンドウサイズを維持）
    while (static_cast<int>(point_history_.size()) > sliding_window_size_) {
        point_history_.pop_front();
        frame_timestamps_.pop_front();
    }
    
    // ボクセル履歴を更新（mapフレームに変換して保存）
    for (const auto& point : points) {
        // 各点をmapフレームに変換（観測時点のTF使用）
        Point3D map_point = transformPointToMap(point, "lidar_link", timestamp);
        
        // mapフレーム座標でボクセル位置を計算
        int vx = static_cast<int>(std::round(map_point.x / voxel_size_));
        int vy = static_cast<int>(std::round(map_point.y / voxel_size_));
        int vz = static_cast<int>(std::round(map_point.z / voxel_size_));
        
        auto voxel_key = std::make_tuple(vx, vy, vz);
        
        // 観測記録を追加（mapフレーム座標で保存）
        VoxelObservation obs;
        obs.position = map_point;  // mapフレーム座標を保存
        obs.timestamp = timestamp;
        obs.angle = point.angle;    // 元の角度情報は保持
        
        voxel_history_[voxel_key].push_back(obs);
    }
    
    // 古い観測記録をクリーンアップ
    cleanupOldObservations(timestamp);
    
    RCLCPP_DEBUG(this->get_logger(), "Sliding window updated: %zu frames, %zu unique voxels", 
                point_history_.size(), voxel_history_.size());
}

std::vector<Point3D> ObstacleTrackerNode::getAggregatedPoints() const
{
    std::vector<Point3D> aggregated_points;
    
    // 各ボクセルの観測回数をチェックして、閾値以上のもののみを含める
    for (const auto& [voxel_key, observations] : voxel_history_) {
        if (static_cast<int>(observations.size()) >= min_voxel_observations_) {
            // 最新の観測を代表点として使用（既にmapフレーム座標）
            if (!observations.empty()) {
                auto latest_obs = std::max_element(observations.begin(), observations.end(),
                    [](const VoxelObservation& a, const VoxelObservation& b) {
                        return a.timestamp < b.timestamp;
                    });
                // 既にmapフレーム座標なのでそのまま使用
                aggregated_points.push_back(latest_obs->position);
            }
        }
    }
    
    RCLCPP_DEBUG(this->get_logger(), "Aggregated points: %zu (from %zu voxels, min_obs=%d)", 
                aggregated_points.size(), voxel_history_.size(), min_voxel_observations_);
    
    return aggregated_points;
}

void ObstacleTrackerNode::cleanupOldObservations(const rclcpp::Time& current_time)
{
    // 最も古いフレームのタイムスタンプを基準とする
    if (frame_timestamps_.empty()) return;
    
    rclcpp::Time oldest_valid_time = frame_timestamps_.front();
    
    // 各ボクセルの観測履歴から古い観測を削除
    for (auto it = voxel_history_.begin(); it != voxel_history_.end();) {
        auto& observations = it->second;
        
        // 古い観測を削除
        observations.erase(
            std::remove_if(observations.begin(), observations.end(),
                [&oldest_valid_time](const VoxelObservation& obs) {
                    return obs.timestamp < oldest_valid_time;
                }),
            observations.end()
        );
        
        // 観測がなくなったボクセルを削除
        if (observations.empty()) {
            it = voxel_history_.erase(it);
        } else {
            ++it;
        }
    }
}

visualization_msgs::msg::MarkerArray ObstacleTrackerNode::createMarkerArray(
    const std::vector<Cluster>& clusters, bool is_dynamic, const rclcpp::Time& timestamp)
{
    visualization_msgs::msg::MarkerArray marker_array;
    int marker_id = 1; // 0はクリアマーカー用に予約
    
    for (const auto& cluster : clusters) {
        // クラスタ内の各点（ボクセル）に対して四角形マーカーを作成
        for (const auto& point : cluster.points) {
            // 各ボクセルの位置をmapフレームに変換（統一された時刻使用）
            Point3D map_point = transformPointToMap(point, "lidar_link", timestamp);
            
            visualization_msgs::msg::Marker marker;
            marker.header.frame_id = "map";
            marker.header.stamp = timestamp;  // 統一された時刻を使用
            marker.ns = is_dynamic ? "dynamic_obstacles" : "static_obstacles";
            marker.id = marker_id++; // 各ボクセルに一意のID
            marker.type = visualization_msgs::msg::Marker::CUBE; // 四角形（立方体）
            marker.action = visualization_msgs::msg::Marker::ADD;
            
            // 位置設定（各ボクセルの位置）
            marker.pose.position.x = map_point.x;
            marker.pose.position.y = map_point.y;
            marker.pose.position.z = map_point.z;
            marker.pose.orientation.w = 1.0;
            
            // サイズ設定（ボクセルサイズ）
            marker.scale.x = voxel_size_;
            marker.scale.y = voxel_size_;
            marker.scale.z = voxel_size_;
            
            // 色設定
            if (is_dynamic) {
                marker.color.r = 1.0; // 赤
                marker.color.g = 0.0;
                marker.color.b = 0.0;
            } else {
                marker.color.r = 0.0;
                marker.color.g = 0.0;
                marker.color.b = 1.0; // 青
            }
            marker.color.a = 0.8;
            
            // マーカーを常時表示するため、lifetimeを0（永続）に設定
            marker.lifetime = rclcpp::Duration::from_seconds(0.0);
            marker_array.markers.push_back(marker);
        }
    }
    
    return marker_array;
}

double ObstacleTrackerNode::calculateDistance(const Point3D& p1, const Point3D& p2)
{
    return std::sqrt((p1.x - p2.x) * (p1.x - p2.x) +
                     (p1.y - p2.y) * (p1.y - p2.y) +
                     (p1.z - p2.z) * (p1.z - p2.z));
}

Point3D ObstacleTrackerNode::calculateCentroid(const std::vector<Point3D>& points)
{
    if (points.empty()) {
        return Point3D(0, 0, 0, 0);
    }
    
    Point3D centroid(0, 0, 0, 0);
    for (const auto& point : points) {
        centroid.x += point.x;
        centroid.y += point.y;
        centroid.z += point.z;
        centroid.angle += point.angle;
    }
    
    centroid.x /= points.size();
    centroid.y /= points.size();
    centroid.z /= points.size();
    centroid.angle /= points.size();
    
    return centroid;
}

bool ObstacleTrackerNode::isWithinProcessingRange(const Point3D& point)
{
    double distance = std::sqrt(point.x * point.x + point.y * point.y + point.z * point.z);
    return distance <= robot_processing_range_;
}

Point3D ObstacleTrackerNode::transformPointToMap(const Point3D& point, const std::string& source_frame, 
                                                 const rclcpp::Time& stamp)
{
    try {
        geometry_msgs::msg::PointStamped point_in, point_out;
        point_in.header.frame_id = source_frame;
        point_in.header.stamp = stamp;
        point_in.point.x = point.x;
        point_in.point.y = point.y;
        point_in.point.z = point.z;
        
        // 最新の利用可能な変換を取得してmapフレームに変換
        geometry_msgs::msg::TransformStamped transform = 
            tf_buffer_->lookupTransform("map", source_frame, tf2::TimePointZero);
        tf2::doTransform(point_in, point_out, transform);
        
        return Point3D(point_out.point.x, point_out.point.y, point_out.point.z, 
                       std::atan2(point_out.point.y, point_out.point.x));
        
    } catch (const tf2::TransformException& ex) {
        // より詳細なエラー情報をログ出力
        RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
            "TF変換に失敗しました (%s -> map): %s", source_frame.c_str(), ex.what());
        
        // 変換に失敗した場合でも処理を継続するため、元の座標をそのまま返す
        return point;
    }
}

Point3D ObstacleTrackerNode::getRobotVelocity()
{
    try {
        // base_footprintの現在位置をmapフレームで取得
        geometry_msgs::msg::TransformStamped transform = 
            tf_buffer_->lookupTransform("map", "base_footprint", tf2::TimePointZero);
        
        Point3D current_position(
            transform.transform.translation.x,
            transform.transform.translation.y,
            transform.transform.translation.z
        );
        
        rclcpp::Time current_time = this->now();
        
        if (robot_velocity_initialized_) {
            // 前回位置との差分から速度を計算
            double dt = (current_time - previous_robot_time_).seconds();
            if (dt > 0.0 && dt < 1.0) {  // 妥当な時間差の場合のみ
                current_robot_velocity_.x = (current_position.x - previous_robot_position_.x) / dt;
                current_robot_velocity_.y = (current_position.y - previous_robot_position_.y) / dt;
                current_robot_velocity_.z = 0.0;
                
                RCLCPP_DEBUG(this->get_logger(), 
                    "Robot velocity: (%.2f, %.2f) m/s, dt=%.3f",
                    current_robot_velocity_.x, current_robot_velocity_.y, dt);
            }
        } else {
            // 初回の場合は速度を0に設定
            current_robot_velocity_ = Point3D(0, 0, 0);
            robot_velocity_initialized_ = true;
        }
        
        // 現在位置と時刻を保存
        previous_robot_position_ = current_position;
        previous_robot_time_ = current_time;
        
        return current_robot_velocity_;
        
    } catch (const tf2::TransformException& ex) {
        RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
            "ロボット位置取得に失敗しました (base_footprint -> map): %s", ex.what());
        
        // 変換に失敗した場合は前回の速度を返す
        return current_robot_velocity_;
    }
}

double ObstacleTrackerNode::getAdaptiveEps(const Point3D& point)
{
    // ロボットからの距離を計算
    double distance_from_robot = std::sqrt(point.x * point.x + point.y * point.y);
    
    // 距離に応じて適応的にepsを調整
    return dbscan_base_eps_ * (1.0 + distance_from_robot * distance_adaptive_factor_);
}

double ObstacleTrackerNode::calculateAngularDifference(const Point3D& p1, const Point3D& p2)
{
    // 元のLiDAR角度情報を直接使用（atan2()による逆算を回避）
    double diff = p2.angle - p1.angle;
    
    // 角度差を計算（-π to π の範囲に正規化）
    while (diff > M_PI) diff -= 2.0 * M_PI;
    while (diff < -M_PI) diff += 2.0 * M_PI;
    
    return std::abs(diff);
}

std::vector<int> ObstacleTrackerNode::getNeighbors(int point_idx, const std::vector<Point3D>& points)
{
    std::vector<int> neighbors;
    const Point3D& query_point = points[point_idx];
    double adaptive_eps = getAdaptiveEps(query_point);
    double adaptive_eps_squared = adaptive_eps * adaptive_eps;  // 事前計算で効率化
    
    for (size_t i = 0; i < points.size(); ++i) {
        if (i == static_cast<size_t>(point_idx)) continue;
        
        const Point3D& candidate = points[i];
        
        // 空間距離チェック（平方距離で高速化）
        double dx = query_point.x - candidate.x;
        double dy = query_point.y - candidate.y;
        double dz = query_point.z - candidate.z;
        double distance_squared = dx * dx + dy * dy + dz * dz;
        if (distance_squared > adaptive_eps_squared) continue;
        
        // 角度差チェック（近い点同士のみ）
        double angular_diff = calculateAngularDifference(query_point, candidate);
        if (angular_diff > max_angular_difference_) continue;
        
        neighbors.push_back(i);
    }
    
    return neighbors;
}

bool ObstacleTrackerNode::isCore(int point_idx, const std::vector<Point3D>& points)
{
    std::vector<int> neighbors = getNeighbors(point_idx, points);
    return neighbors.size() >= static_cast<size_t>(dbscan_min_points_);
}




void ObstacleTrackerNode::addClearMarkers(visualization_msgs::msg::MarkerArray& marker_array, 
                                         const std::string& namespace_name, const rclcpp::Time& timestamp)
{
    // DELETEALLマーカーを先頭に追加して、古いマーカーをクリア
    visualization_msgs::msg::Marker clear_marker;
    clear_marker.header.frame_id = "map";
    clear_marker.header.stamp = timestamp;  // 統一された時刻を使用
    clear_marker.ns = namespace_name;
    clear_marker.id = 0;
    clear_marker.action = visualization_msgs::msg::Marker::DELETEALL;
    
    // marker_arrayの先頭に挿入
    marker_array.markers.insert(marker_array.markers.begin(), clear_marker);
    
    // 既存マーカーのIDを1から開始するように調整
    for (size_t i = 1; i < marker_array.markers.size(); ++i) {
        marker_array.markers[i].id = static_cast<int>(i);
    }
}

// 改善された追跡アルゴリズム
void ObstacleTrackerNode::enhancedTrackClusters(std::vector<Cluster>& clusters)
{
    // ロボットの現在速度を取得（キャッシュ）
    static Point3D cached_robot_velocity(0, 0, 0);
    static rclcpp::Time last_velocity_update;
    
    rclcpp::Time current_time = this->now();
    static double last_velocity_update_sec = 0.0;
    double current_time_sec = current_time.seconds();
    
    if (last_velocity_update_sec == 0.0 || 
        (current_time_sec - last_velocity_update_sec) > 0.05) { // 20Hzで更新
        cached_robot_velocity = getRobotVelocity();
        last_velocity_update_sec = current_time_sec;
    }
    
    // 時間差分計算
    double dt = 0.1; // デフォルト値
    double last_scan_time_sec = last_scan_time_.seconds();
    
    if (last_scan_time_sec > 0.0) {
        dt = current_time_sec - last_scan_time_sec;
    }
    if (dt <= 0.001 || dt > 1.0) {
        dt = 0.1; // 異常な時間差分の場合はデフォルト値を使用
    }
    
    // マッチングマトリックスの計算
    std::vector<std::vector<double>> matching_scores;
    matching_scores.resize(previous_clusters_.size());
    
    std::vector<int> prev_cluster_ids;
    for (const auto& [id, cluster] : previous_clusters_) {
        prev_cluster_ids.push_back(id);
    }
    
    for (size_t i = 0; i < prev_cluster_ids.size(); ++i) {
        matching_scores[i].resize(clusters.size());
        const auto& prev_cluster = previous_clusters_[prev_cluster_ids[i]];
        
        for (size_t j = 0; j < clusters.size(); ++j) {
            matching_scores[i][j] = calculateMatchingScore(prev_cluster, clusters[j], dt);
        }
    }
    
    // シンプルなGreedy matching（Hungarian algorithmの代替）
    std::vector<bool> prev_matched(prev_cluster_ids.size(), false);
    std::vector<bool> curr_matched(clusters.size(), false);
    
    // 高いスコアから順にマッチング
    for (int iteration = 0; iteration < static_cast<int>(std::min(prev_cluster_ids.size(), clusters.size())); ++iteration) {
        double best_score = -1.0;
        int best_prev_idx = -1;
        int best_curr_idx = -1;
        
        for (size_t i = 0; i < prev_cluster_ids.size(); ++i) {
            if (prev_matched[i]) continue;
            
            for (size_t j = 0; j < clusters.size(); ++j) {
                if (curr_matched[j]) continue;
                
                if (matching_scores[i][j] > best_score) {
                    best_score = matching_scores[i][j];
                    best_prev_idx = i;
                    best_curr_idx = j;
                }
            }
        }
        
        // 最低スコア閾値チェック
        if (best_score < 0.3) {
            break; // スコアが低すぎる場合は残りはマッチングしない
        }
        
        // マッチングを適用
        if (best_prev_idx >= 0 && best_curr_idx >= 0) {
            prev_matched[best_prev_idx] = true;
            curr_matched[best_curr_idx] = true;
            
            const auto& prev_cluster = previous_clusters_[prev_cluster_ids[best_prev_idx]];
            auto& curr_cluster = clusters[best_curr_idx];
            
            // IDを継承
            curr_cluster.id = prev_cluster.id;
            curr_cluster.track_count = prev_cluster.track_count + 1;
            curr_cluster.missing_count = 0;
            
            // 速度計算
            Point3D relative_velocity;
            relative_velocity.x = (curr_cluster.centroid.x - prev_cluster.centroid.x) / dt;
            relative_velocity.y = (curr_cluster.centroid.y - prev_cluster.centroid.y) / dt;
            relative_velocity.z = 0.0;
            
            // ロボットの移動速度を差し引いて絶対速度を計算
            curr_cluster.velocity.x = relative_velocity.x - cached_robot_velocity.x;
            curr_cluster.velocity.y = relative_velocity.y - cached_robot_velocity.y;
            curr_cluster.velocity.z = 0.0;
            
            // 速度平滑化
            curr_cluster.smoothed_velocity = smoothVelocity(
                prev_cluster.smoothed_velocity, curr_cluster.velocity, velocity_smoothing_alpha_);
            
            // 信頼度更新
            updateClusterConfidence(curr_cluster, true);
            
            RCLCPP_DEBUG(this->get_logger(), 
                "Enhanced tracking - Cluster %d matched with score %.3f, confidence %.2f",
                curr_cluster.id, best_score, curr_cluster.confidence);
        }
    }
    
    // マッチングされなかった現在のクラスタは新規として扱う
    for (size_t j = 0; j < clusters.size(); ++j) {
        if (!curr_matched[j]) {
            clusters[j].id = next_cluster_id_++;
            clusters[j].velocity = Point3D(0, 0, 0);
            clusters[j].smoothed_velocity = Point3D(0, 0, 0);
            clusters[j].track_count = 1;
            clusters[j].confidence = 0.5; // 初期信頼度
            clusters[j].missing_count = 0;
            
            RCLCPP_DEBUG(this->get_logger(), 
                "Enhanced tracking - New cluster %d created", clusters[j].id);
        }
    }
    
    // 現在のクラスタを保存
    previous_clusters_.clear();
    for (const auto& cluster : clusters) {
        previous_clusters_.emplace(cluster.id, cluster);
    }
}

double ObstacleTrackerNode::calculateMatchingScore(const Cluster& prev, const Cluster& curr, double dt)
{
    // 1. 位置距離スコア
    double dx = curr.centroid.x - prev.centroid.x;
    double dy = curr.centroid.y - prev.centroid.y;
    double position_distance = std::sqrt(dx * dx + dy * dy);
    double position_score = std::exp(-position_distance / 1.0); // 1m で大幅減衰
    
    // 2. サイズ類似度スコア
    double prev_size = static_cast<double>(prev.points.size());
    double curr_size = static_cast<double>(curr.points.size());
    if (prev_size < 1.0) prev_size = 1.0; // ゼロ除算防止
    
    double size_ratio = curr_size / prev_size;
    if (size_ratio > 1.0) size_ratio = 1.0 / size_ratio; // 0-1に正規化
    double size_score = size_ratio;
    
    // 3. 予測位置との一致度スコア
    Point3D predicted_pos;
    predicted_pos.x = prev.centroid.x + prev.smoothed_velocity.x * dt;
    predicted_pos.y = prev.centroid.y + prev.smoothed_velocity.y * dt;
    
    double pred_dx = curr.centroid.x - predicted_pos.x;
    double pred_dy = curr.centroid.y - predicted_pos.y;
    double prediction_error = std::sqrt(pred_dx * pred_dx + pred_dy * pred_dy);
    double prediction_score = std::exp(-prediction_error / 0.5); // 0.5m で大幅減衰
    
    // 4. 信頼度による重み付け
    double confidence_weight = std::max(0.3, prev.confidence);
    
    // 重み付き統合スコア
    double total_score = confidence_weight * (
        0.5 * position_score + 
        0.2 * size_score + 
        0.3 * prediction_score
    );
    
    return total_score;
}

Point3D ObstacleTrackerNode::smoothVelocity(const Point3D& prev_vel, const Point3D& measured_vel, double alpha)
{
    Point3D smooth_vel;
    smooth_vel.x = alpha * measured_vel.x + (1.0 - alpha) * prev_vel.x;
    smooth_vel.y = alpha * measured_vel.y + (1.0 - alpha) * prev_vel.y;
    smooth_vel.z = 0.0;
    return smooth_vel;
}

void ObstacleTrackerNode::updateClusterConfidence(Cluster& cluster, bool matched)
{
    if (matched) {
        // マッチングされた場合、信頼度を向上
        cluster.confidence = std::min(1.0, cluster.confidence + 0.1);
        cluster.missing_count = 0;
    } else {
        // マッチングされなかった場合、信頼度を減衰
        cluster.confidence = std::max(0.0, cluster.confidence - confidence_decay_rate_);
        cluster.missing_count++;
    }
}

} // namespace obstacle_tracker
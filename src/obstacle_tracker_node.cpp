#include "obstacle_tracker/obstacle_tracker_node.hpp"
#include <cmath>
#include <algorithm>
#include <chrono>
#include <unordered_map>
#include <string>

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
    
    // 楕円表示パラメータ
    this->declare_parameter("enable_ellipse_markers", true);
    this->declare_parameter("ellipse_scale_factor", 1.2);
    
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
    
    // 楕円表示パラメータ取得
    enable_ellipse_markers_ = this->get_parameter("enable_ellipse_markers").as_bool();
    ellipse_scale_factor_ = this->get_parameter("ellipse_scale_factor").as_double();
    
    // パラメータ検証
    validateParameters();
    
    // サブスクライバーとパブリッシャーの作成
    scan_subscriber_ = this->create_subscription<sensor_msgs::msg::LaserScan>(
        "/scan", 10, std::bind(&ObstacleTrackerNode::laserScanCallback, this, std::placeholders::_1));
    
    dynamic_obstacles_publisher_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(
        "/dynamic_obstacles", 10);
    
    static_obstacles_publisher_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(
        "/static_obstacles", 10);
    
    // 楕円マーカー用パブリッシャー
    dynamic_ellipse_publisher_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(
        "/dynamic_ellipses", 10);
    static_ellipse_publisher_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(
        "/static_ellipses", 10);
    
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
    RCLCPP_INFO(this->get_logger(), "Ellipse markers: %s", enable_ellipse_markers_ ? "enabled" : "disabled");
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
    
    // 楕円表示関連パラメータの検証
    if (ellipse_scale_factor_ <= MIN_ELLIPSE_SCALE || ellipse_scale_factor_ > MAX_ELLIPSE_SCALE) {
        error_messages.push_back("Invalid ellipse_scale_factor: " + std::to_string(ellipse_scale_factor_) + 
                               " (must be " + std::to_string(MIN_ELLIPSE_SCALE) + " < value <= " + std::to_string(MAX_ELLIPSE_SCALE) + ")");
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
        
        // 3. クラスタリング（時刻を統一して渡す）
        std::vector<Cluster> clusters = enable_adaptive_dbscan_ ? 
            adaptiveDBSCANCluster(voxelized_points, current_time) : clusterPoints(voxelized_points, current_time);
        
        // 4. クラスタ追跡
        trackClusters(clusters);
        
        // 5. 動的・静的分類
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
        // 速度ベクトルの大きさを計算
        double speed = std::sqrt(cluster.velocity.x * cluster.velocity.x +
                                cluster.velocity.y * cluster.velocity.y);
        
        // しきい値と比較して動的・静的を判別
        cluster.is_dynamic = (speed > dynamic_cluster_speed_threshold_);
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
    
    // MarkerArrayを作成して配信（統一された時刻使用）
    auto dynamic_markers = createMarkerArray(dynamic_clusters, true, timestamp);
    auto static_markers = createMarkerArray(static_clusters, false, timestamp);
    
    // 古いマーカーをクリアするためのマーカーを追加
    addClearMarkers(dynamic_markers, "dynamic_obstacles", timestamp);
    addClearMarkers(static_markers, "static_obstacles", timestamp);
    
    dynamic_obstacles_publisher_->publish(dynamic_markers);
    static_obstacles_publisher_->publish(static_markers);
    
    // 楕円マーカーを作成して配信（統一された時刻使用）
    if (enable_ellipse_markers_) {
        auto dynamic_ellipses = createEllipseMarkers(dynamic_clusters, true, timestamp);
        auto static_ellipses = createEllipseMarkers(static_clusters, false, timestamp);
        
        // 楕円マーカーの古いマーカーもクリア
        addClearMarkers(dynamic_ellipses, "dynamic_ellipses", timestamp);
        addClearMarkers(static_ellipses, "static_ellipses", timestamp);
        
        dynamic_ellipse_publisher_->publish(dynamic_ellipses);
        static_ellipse_publisher_->publish(static_ellipses);
    }
    
    RCLCPP_DEBUG(this->get_logger(), "Published %zu dynamic and %zu static clusters",
                dynamic_clusters.size(), static_clusters.size());
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

visualization_msgs::msg::MarkerArray ObstacleTrackerNode::createEllipseMarkers(
    const std::vector<Cluster>& clusters, bool is_dynamic, const rclcpp::Time& timestamp)
{
    visualization_msgs::msg::MarkerArray marker_array;
    int marker_id = 1; // 0はクリアマーカー用に予約
    
    for (const auto& cluster : clusters) {
        if (cluster.points.empty()) continue;
        
        // クラスタの楕円パラメータを計算
        EllipseParams ellipse = calculateClusterEllipse(cluster.points);
        
        visualization_msgs::msg::Marker marker;
        marker.header.frame_id = "map";
        marker.header.stamp = timestamp;  // 統一された時刻を使用
        marker.ns = is_dynamic ? "dynamic_ellipses" : "static_ellipses";
        marker.id = marker_id++; // 連番でID割り当て
        marker.type = visualization_msgs::msg::Marker::CYLINDER; // 楕円は円柱で近似
        marker.action = visualization_msgs::msg::Marker::ADD;
        
        // 位置設定（楕円中心をmapフレームに変換、統一された時刻使用）
        Point3D map_center = transformPointToMap(ellipse.center, "lidar_link", timestamp);
        marker.pose.position.x = map_center.x;
        marker.pose.position.y = map_center.y;
        marker.pose.position.z = map_center.z;
        
        // 楕円の向きをロボットの回転に合わせて変換（統一された時刻使用）
        double map_orientation = transformOrientationToMap(ellipse.orientation, timestamp);
        
        // 回転設定（Z軸回りの回転）
        marker.pose.orientation.x = 0.0;
        marker.pose.orientation.y = 0.0;
        marker.pose.orientation.z = std::sin(map_orientation / 2.0);
        marker.pose.orientation.w = std::cos(map_orientation / 2.0);
        
        // サイズ設定（楕円のサイズをスケール）
        marker.scale.x = ellipse.semi_major_axis * 2.0 * ellipse_scale_factor_;
        marker.scale.y = ellipse.semi_minor_axis * 2.0 * ellipse_scale_factor_;
        marker.scale.z = 0.1; // 薄い円柱
        
        // 色設定（透明度を上げて境界線として表示）
        if (is_dynamic) {
            marker.color.r = 1.0; // 赤
            marker.color.g = 0.0;
            marker.color.b = 0.0;
        } else {
            marker.color.r = 0.0;
            marker.color.g = 0.0;
            marker.color.b = 1.0; // 青
        }
        marker.color.a = 0.3; // 透明度
        
        // 楕円マーカーを常時表示するため、lifetimeを0（永続）に設定
        marker.lifetime = rclcpp::Duration::from_seconds(0.0);
        marker_array.markers.push_back(marker);
    }
    
    return marker_array;
}

ObstacleTrackerNode::EllipseParams ObstacleTrackerNode::calculateClusterEllipse(
    const std::vector<Point3D>& points)
{
    EllipseParams ellipse;
    const double EPSILON = 1e-10;
    const double MIN_AXIS_LENGTH = 0.05;
    const double MAX_AXIS_LENGTH = 10.0;
    
    if (points.size() < 2) {
        // 点数が少ない場合は小さな円として扱う
        ellipse.center = points.empty() ? Point3D(0, 0, 0, 0) : calculateCentroid(points);
        ellipse.semi_major_axis = MIN_AXIS_LENGTH;
        ellipse.semi_minor_axis = MIN_AXIS_LENGTH;
        ellipse.orientation = 0.0;
        return ellipse;
    }
    
    // 重心を計算
    ellipse.center = calculateCentroid(points);
    
    // 共分散行列を計算（数値安定性改善）
    double cxx = 0.0, cyy = 0.0, cxy = 0.0;
    double sum_weight = 0.0;
    
    for (const auto& point : points) {
        double dx = point.x - ellipse.center.x;
        double dy = point.y - ellipse.center.y;
        
        // 重み（距離の逆数で遠い点の影響を減らす）
        double distance = std::sqrt(dx * dx + dy * dy);
        double weight = 1.0 / (1.0 + distance);
        
        cxx += weight * dx * dx;
        cyy += weight * dy * dy;
        cxy += weight * dx * dy;
        sum_weight += weight;
    }
    
    // 正規化（ゼロ除算防止）
    if (sum_weight < EPSILON) {
        ellipse.semi_major_axis = MIN_AXIS_LENGTH;
        ellipse.semi_minor_axis = MIN_AXIS_LENGTH;
        ellipse.orientation = 0.0;
        return ellipse;
    }
    
    cxx /= sum_weight;
    cyy /= sum_weight;
    cxy /= sum_weight;
    
    // 固有値計算の数値安定性改善
    double trace = cxx + cyy;
    double det = cxx * cyy - cxy * cxy;
    
    // 判別式の計算（負値チェック）
    double discriminant_squared = trace * trace - 4.0 * det;
    if (discriminant_squared < 0.0) {
        discriminant_squared = 0.0; // 数値誤差による負値を修正
    }
    double discriminant = std::sqrt(discriminant_squared);
    
    double eigenvalue1 = (trace + discriminant) / 2.0;
    double eigenvalue2 = (trace - discriminant) / 2.0;
    
    // 固有値の安全性チェック
    eigenvalue1 = std::max(eigenvalue1, EPSILON);
    eigenvalue2 = std::max(eigenvalue2, EPSILON);
    
    // 楕円の軸長を計算（2σ = 95%信頼区間）
    double axis1 = 2.0 * std::sqrt(eigenvalue1);
    double axis2 = 2.0 * std::sqrt(eigenvalue2);
    
    ellipse.semi_major_axis = std::max(axis1, axis2);
    ellipse.semi_minor_axis = std::min(axis1, axis2);
    
    // サイズ制限
    ellipse.semi_major_axis = std::clamp(ellipse.semi_major_axis, MIN_AXIS_LENGTH, MAX_AXIS_LENGTH);
    ellipse.semi_minor_axis = std::clamp(ellipse.semi_minor_axis, MIN_AXIS_LENGTH, MAX_AXIS_LENGTH);
    
    // 楕円の回転角を計算（数値安定性改善）
    if (std::abs(cxy) < EPSILON) {
        ellipse.orientation = (cxx > cyy) ? 0.0 : M_PI / 2.0;
    } else {
        ellipse.orientation = 0.5 * std::atan2(2.0 * cxy, cxx - cyy);
        
        // 角度を [-π/2, π/2] に正規化
        while (ellipse.orientation > M_PI / 2.0) ellipse.orientation -= M_PI;
        while (ellipse.orientation < -M_PI / 2.0) ellipse.orientation += M_PI;
    }
    
    // 軸の長さ比チェック（極端な楕円を防ぐ）
    double aspect_ratio = ellipse.semi_major_axis / ellipse.semi_minor_axis;
    if (aspect_ratio > 5.0) {
        ellipse.semi_minor_axis = ellipse.semi_major_axis / 5.0;
    }
    
    return ellipse;
}

double ObstacleTrackerNode::transformOrientationToMap(double lidar_orientation, const rclcpp::Time& stamp)
{
    try {
        // lidar_linkからmapフレームへの変換を取得
        geometry_msgs::msg::TransformStamped transform = 
            tf_buffer_->lookupTransform("map", "lidar_link", tf2::TimePointZero);
        
        // クォータニオンからZ軸回りの回転角（ヨー角）を抽出
        // 簡単な変換式を使用: yaw = atan2(2(qw*qz + qx*qy), 1 - 2(qy² + qz²))
        double qx = transform.transform.rotation.x;
        double qy = transform.transform.rotation.y;
        double qz = transform.transform.rotation.z;
        double qw = transform.transform.rotation.w;
        
        double yaw = std::atan2(2.0 * (qw * qz + qx * qy), 
                               1.0 - 2.0 * (qy * qy + qz * qz));
        
        // lidar_linkフレームでの楕円の向きにロボットの回転を加算
        double map_orientation = lidar_orientation + yaw;
        
        // 角度を [-π, π] の範囲に正規化
        while (map_orientation > M_PI) map_orientation -= 2.0 * M_PI;
        while (map_orientation < -M_PI) map_orientation += 2.0 * M_PI;
        
        return map_orientation;
        
    } catch (const tf2::TransformException& ex) {
        RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
            "楕円の向き変換に失敗しました (lidar_link -> map): %s", ex.what());
        
        // 変換に失敗した場合は元の向きをそのまま返す
        return lidar_orientation;
    }
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

} // namespace obstacle_tracker
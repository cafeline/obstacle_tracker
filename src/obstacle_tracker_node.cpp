#include "obstacle_tracker/obstacle_tracker_node.hpp"
#include <cmath>
#include <algorithm>

namespace obstacle_tracker
{

ObstacleTrackerNode::ObstacleTrackerNode() : Node("obstacle_tracker_node"), next_cluster_id_(1)
{
    // TFバッファとリスナーの初期化
    tf_buffer_ = std::make_shared<tf2_ros::Buffer>(this->get_clock());
    tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);
    
    // パラメータの宣言とデフォルト値設定
    this->declare_parameter("robot_processing_range", 10.0);
    this->declare_parameter("dynamic_cluster_speed_threshold", 1.5);
    this->declare_parameter("voxel_size", 0.2);
    this->declare_parameter("clustering_max_distance", 0.3);
    this->declare_parameter("min_cluster_points", 2);
    
    // パラメータの取得
    robot_processing_range_ = this->get_parameter("robot_processing_range").as_double();
    dynamic_cluster_speed_threshold_ = this->get_parameter("dynamic_cluster_speed_threshold").as_double();
    voxel_size_ = this->get_parameter("voxel_size").as_double();
    clustering_max_distance_ = this->get_parameter("clustering_max_distance").as_double();
    min_cluster_points_ = this->get_parameter("min_cluster_points").as_int();
    
    // サブスクライバーとパブリッシャーの作成
    scan_subscriber_ = this->create_subscription<sensor_msgs::msg::LaserScan>(
        "/scan", 10, std::bind(&ObstacleTrackerNode::laserScanCallback, this, std::placeholders::_1));
    
    dynamic_obstacles_publisher_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(
        "/dynamic_obstacles", 10);
    
    static_obstacles_publisher_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(
        "/static_obstacles", 10);
    
    RCLCPP_INFO(this->get_logger(), "ObstacleTracker node initialized");
    RCLCPP_INFO(this->get_logger(), "Processing range: %.2f m", robot_processing_range_);
    RCLCPP_INFO(this->get_logger(), "Dynamic threshold: %.2f m/s", dynamic_cluster_speed_threshold_);
}

void ObstacleTrackerNode::laserScanCallback(const sensor_msgs::msg::LaserScan::SharedPtr msg)
{
    try {
        // スキャン時間を保存
        last_scan_time_ = msg->header.stamp;
        
        // 1. レーザースキャンを点群に変換
        std::vector<Point3D> points = convertScanToPoints(msg);
        
        // 2. ボクセル化
        std::vector<Point3D> voxelized_points = voxelizePoints(points);
        
        // 3. クラスタリング
        std::vector<Cluster> clusters = clusterPoints(voxelized_points);
        
        // 4. クラスタ追跡
        trackClusters(clusters);
        
        // 5. 動的・静的分類
        classifyClusters(clusters);
        
        // 6. 結果を配信
        publishObstacles(clusters);
        
    } catch (const std::exception& e) {
        RCLCPP_ERROR(this->get_logger(), "Error in laser scan processing: %s", e.what());
    }
}

std::vector<Point3D> ObstacleTrackerNode::convertScanToPoints(const sensor_msgs::msg::LaserScan::SharedPtr scan)
{
    std::vector<Point3D> points;
    
    for (size_t i = 0; i < scan->ranges.size(); ++i) {
        double range = scan->ranges[i];
        
        // 範囲外や無効な値をスキップ
        if (range < scan->range_min || range > scan->range_max || std::isnan(range) || std::isinf(range)) {
            continue;
        }
        
        // 角度計算
        double angle = scan->angle_min + i * scan->angle_increment;
        
        // 直交座標に変換
        Point3D point;
        point.x = range * std::cos(angle);
        point.y = range * std::sin(angle);
        point.z = 0.0;
        
        // 処理範囲内かチェック
        if (isWithinProcessingRange(point)) {
            points.push_back(point);
        }
    }
    
    return points;
}

std::vector<Point3D> ObstacleTrackerNode::voxelizePoints(const std::vector<Point3D>& points)
{
    std::map<std::tuple<int, int, int>, Point3D> voxel_map;
    
    for (const auto& point : points) {
        // ボクセルグリッドのインデックス計算
        int vx = static_cast<int>(std::floor(point.x / voxel_size_));
        int vy = static_cast<int>(std::floor(point.y / voxel_size_));
        int vz = static_cast<int>(std::floor(point.z / voxel_size_));
        
        auto key = std::make_tuple(vx, vy, vz);
        
        // 既存のボクセルがある場合は平均を取る
        if (voxel_map.find(key) != voxel_map.end()) {
            voxel_map[key].x = (voxel_map[key].x + point.x) / 2.0;
            voxel_map[key].y = (voxel_map[key].y + point.y) / 2.0;
            voxel_map[key].z = (voxel_map[key].z + point.z) / 2.0;
        } else {
            voxel_map[key] = point;
        }
    }
    
    std::vector<Point3D> voxelized_points;
    for (const auto& pair : voxel_map) {
        voxelized_points.push_back(pair.second);
    }
    
    return voxelized_points;
}

std::vector<Cluster> ObstacleTrackerNode::clusterPoints(const std::vector<Point3D>& voxelized_points)
{
    std::vector<Cluster> clusters;
    std::vector<bool> visited(voxelized_points.size(), false);
    
    for (size_t i = 0; i < voxelized_points.size(); ++i) {
        if (visited[i]) continue;
        
        Cluster cluster;
        cluster.id = next_cluster_id_++;
        std::vector<size_t> cluster_indices;
        
        // BFS (幅優先探索) を使用してクラスタを成長
        std::vector<size_t> queue = {i};
        visited[i] = true;
        
        while (!queue.empty()) {
            size_t current_idx = queue.back();
            queue.pop_back();
            cluster_indices.push_back(current_idx);
            cluster.points.push_back(voxelized_points[current_idx]);
            
            // 近隣点を探索
            for (size_t j = 0; j < voxelized_points.size(); ++j) {
                if (visited[j]) continue;
                
                double distance = calculateDistance(voxelized_points[current_idx], voxelized_points[j]);
                if (distance <= clustering_max_distance_) {
                    visited[j] = true;
                    queue.push_back(j);
                }
            }
        }
        
        // 最小点数チェック
        if (static_cast<int>(cluster.points.size()) >= min_cluster_points_) {
            Point3D laser_centroid = calculateCentroid(cluster.points);
            // lidar_linkフレームからmapフレームに変換（現在時刻を使用）
            cluster.centroid = transformPointToMap(laser_centroid, "lidar_link", this->now());
            clusters.push_back(cluster);
        }
    }
    
    return clusters;
}

void ObstacleTrackerNode::trackClusters(std::vector<Cluster>& clusters)
{
    // シンプルな最近隣マッチングによる追跡
    for (auto& cluster : clusters) {
        double min_distance = std::numeric_limits<double>::max();
        int best_match_id = -1;
        
        // 前フレームのクラスタと比較
        for (const auto& prev_pair : previous_clusters_) {
            const auto& prev_cluster = prev_pair.second;
            double distance = calculateDistance(cluster.centroid, prev_cluster.centroid);
            
            if (distance < min_distance && distance <= clustering_max_distance_ * 2.0) {
                min_distance = distance;
                best_match_id = prev_cluster.id;
            }
        }
        
        if (best_match_id != -1) {
            // マッチした場合、IDを継承し速度を計算
            cluster.id = best_match_id;
            const auto& prev_cluster = previous_clusters_[best_match_id];
            
            // シンプルな移動平均による速度計算
            double dt = 0.1; // 仮定: 10Hz
            cluster.velocity.x = (cluster.centroid.x - prev_cluster.centroid.x) / dt;
            cluster.velocity.y = (cluster.centroid.y - prev_cluster.centroid.y) / dt;
            cluster.velocity.z = 0.0;
            
            cluster.track_count = prev_cluster.track_count + 1;
        } else {
            // 新しいクラスタの場合
            cluster.velocity = Point3D(0, 0, 0);
            cluster.track_count = 1;
        }
    }
    
    // 現在のクラスタを保存
    previous_clusters_.clear();
    for (const auto& cluster : clusters) {
        previous_clusters_[cluster.id] = cluster;
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

void ObstacleTrackerNode::publishObstacles(const std::vector<Cluster>& clusters)
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
    
    // MarkerArrayを作成して配信
    auto dynamic_markers = createMarkerArray(dynamic_clusters, true);
    auto static_markers = createMarkerArray(static_clusters, false);
    
    dynamic_obstacles_publisher_->publish(dynamic_markers);
    static_obstacles_publisher_->publish(static_markers);
    
    RCLCPP_DEBUG(this->get_logger(), "Published %zu dynamic and %zu static clusters",
                dynamic_clusters.size(), static_clusters.size());
}

visualization_msgs::msg::MarkerArray ObstacleTrackerNode::createMarkerArray(
    const std::vector<Cluster>& clusters, bool is_dynamic)
{
    visualization_msgs::msg::MarkerArray marker_array;
    int marker_id = 0;
    
    for (const auto& cluster : clusters) {
        // クラスタ内の各点（ボクセル）に対して四角形マーカーを作成
        for (const auto& point : cluster.points) {
            // 各ボクセルの位置をmapフレームに変換（現在時刻を使用）
            Point3D map_point = transformPointToMap(point, "lidar_link", this->now());
            
            visualization_msgs::msg::Marker marker;
            marker.header.frame_id = "map";
            marker.header.stamp = this->now();
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
            
            marker.lifetime = rclcpp::Duration::from_seconds(0.2);
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
        return Point3D(0, 0, 0);
    }
    
    Point3D centroid(0, 0, 0);
    for (const auto& point : points) {
        centroid.x += point.x;
        centroid.y += point.y;
        centroid.z += point.z;
    }
    
    centroid.x /= points.size();
    centroid.y /= points.size();
    centroid.z /= points.size();
    
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
        
        return Point3D(point_out.point.x, point_out.point.y, point_out.point.z);
        
    } catch (const tf2::TransformException& ex) {
        // より詳細なエラー情報をログ出力
        RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
            "TF変換に失敗しました (%s -> map): %s", source_frame.c_str(), ex.what());
        
        // 変換に失敗した場合でも処理を継続するため、元の座標をそのまま返す
        return point;
    }
}

} // namespace obstacle_tracker
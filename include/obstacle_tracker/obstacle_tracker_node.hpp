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

namespace obstacle_tracker
{

struct Point3D
{
    double x, y, z;
    Point3D(double x = 0.0, double y = 0.0, double z = 0.0) : x(x), y(y), z(z) {}
};

struct Cluster
{
    std::vector<Point3D> points;
    Point3D centroid;
    Point3D velocity;
    int id;
    bool is_dynamic;
    int track_count;
    
    Cluster() : centroid(0, 0, 0), velocity(0, 0, 0), id(0), is_dynamic(false), track_count(0) {}
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
    std::vector<Cluster> clusterPoints(const std::vector<Point3D>& voxelized_points);
    void trackClusters(std::vector<Cluster>& clusters);
    void classifyClusters(std::vector<Cluster>& clusters);
    
    // 出力関数
    void publishObstacles(const std::vector<Cluster>& clusters);
    visualization_msgs::msg::MarkerArray createMarkerArray(
        const std::vector<Cluster>& clusters, bool is_dynamic);
    
    // ユーティリティ関数
    double calculateDistance(const Point3D& p1, const Point3D& p2);
    Point3D calculateCentroid(const std::vector<Point3D>& points);
    bool isWithinProcessingRange(const Point3D& point);
    Point3D transformPointToMap(const Point3D& point, const std::string& source_frame, 
                               const rclcpp::Time& stamp);
    Point3D getRobotVelocity();
    
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
    
    // TF関連
    std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
    std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
    
    // 内部状態
    std::map<int, Cluster> previous_clusters_;
    int next_cluster_id_;
    rclcpp::Time last_scan_time_;
    
    // ロボット速度追跡
    Point3D previous_robot_position_;
    rclcpp::Time previous_robot_time_;
    Point3D current_robot_velocity_;
    bool robot_velocity_initialized_;
    
    // 処理頻度制限
    rclcpp::Time last_processing_time_;
    bool processing_time_initialized_;
};

} // namespace obstacle_tracker

#endif // OBSTACLE_TRACKER_NODE_HPP_
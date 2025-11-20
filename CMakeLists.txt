cmake_minimum_required(VERSION 3.8)
project(obstacle_tracker)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(Eigen3 REQUIRED)

add_executable(obstacle_tracker
  src/obstacle_tracker.cpp
  src/obstacle_tracker_node.cpp
)
target_include_directories(obstacle_tracker PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
target_compile_features(obstacle_tracker PUBLIC c_std_99 cxx_std_17)  # Require C99 and C++17

ament_target_dependencies(obstacle_tracker
  rclcpp
  sensor_msgs
  visualization_msgs
  geometry_msgs
  tf2
  tf2_ros
  tf2_geometry_msgs
)
target_link_libraries(obstacle_tracker Eigen3::Eigen)

install(TARGETS obstacle_tracker
  DESTINATION lib/${PROJECT_NAME})

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  find_package(ament_cmake_gtest REQUIRED)

  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()

  # Add gtest
  ament_add_gtest(${PROJECT_NAME}_test
    test/test_obstacle_tracker.cpp
    src/obstacle_tracker_node.cpp
  )
  if(TARGET ${PROJECT_NAME}_test)
    target_include_directories(${PROJECT_NAME}_test PUBLIC
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
      $<INSTALL_INTERFACE:include>)
    ament_target_dependencies(${PROJECT_NAME}_test
      rclcpp
      sensor_msgs
      visualization_msgs
      geometry_msgs
      tf2
      tf2_ros
      tf2_geometry_msgs
    )
    target_link_libraries(${PROJECT_NAME}_test Eigen3::Eigen)
  endif()
endif()

ament_package()

cmake_minimum_required(VERSION 3.8)
project(motor_reset)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(interface_protocol REQUIRED)
find_package(yaml-cpp REQUIRED)

# ----------------------------------------
# Add your reset_node executable HERE
# ----------------------------------------
add_executable(reset_node src/reset_node.cpp)

ament_target_dependencies(reset_node
  rclcpp
  std_msgs
  interface_protocol
)
target_link_libraries(reset_node yaml-cpp)

install(
  TARGETS reset_node
  DESTINATION lib/${PROJECT_NAME}
)

# ----------------------------------------
# Testing (optional)
# ----------------------------------------
if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)

  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)

  ament_lint_auto_find_test_dependencies()
endif()

ament_package()

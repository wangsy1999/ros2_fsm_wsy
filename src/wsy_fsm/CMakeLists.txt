cmake_minimum_required(VERSION 3.8)
project(wsy_fsm)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(interface_protocol REQUIRED)
find_package(ament_index_cpp REQUIRED)
find_package(yaml-cpp REQUIRED)

include_directories(
  include
)

add_executable(fsm_node
  src/main.cpp
  src/fsm_controller.cpp
  src/reset_module.cpp
  src/estop_module.cpp
  src/run_module.cpp
  src/keyboard.cpp
)

ament_target_dependencies(fsm_node
  rclcpp
  std_msgs
  interface_protocol
  ament_index_cpp
)

target_link_libraries(fsm_node yaml-cpp)

install(TARGETS fsm_node
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY include/
  DESTINATION include
)

install(DIRECTORY config/
  DESTINATION share/${PROJECT_NAME}/config
)

ament_package()

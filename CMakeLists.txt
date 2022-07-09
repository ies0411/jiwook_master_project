cmake_minimum_required(VERSION 3.0.2)
project(jiwook_v1)
set(CMAKE_CXX_STANDARD 17)

add_compile_options(-std=c++17)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  std_msgs
  sensor_msgs
  cv_bridge
)
find_package(OpenCV REQUIRED)
# find_package(Ceres REQUIRED)
# find_package(FMT REQUIRED)
catkin_package(
  CATKIN_DEPENDS roscpp std_msgs
)

# ##########
# # Build ##
# ##########
include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  # ${CERES_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
)

add_executable(hello src/hello.cpp)
# add_executable(hello_sub src/hello_sub.cpp)
# add_executable(feature src/feature.cpp)
# add_executable(ceres_tutorial src/ceres_tutorial.cpp)

add_dependencies(hello ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
# add_dependencies(hello_sub ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
# add_dependencies(feature ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
# add_dependencies(ceres_tutorial ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

# target_link_libraries(ceres_tutorial
#   ${catkin_LIBRARIES} ${OpenCV_LIBRARIES} ${CERES_LIBRARIES}
# )
# target_link_libraries(feature
#   ${catkin_LIBRARIES} ${OpenCV_LIBRARIES}
# )
target_link_libraries(hello ${OpenCV_LIBRARIES}
  ${catkin_LIBRARIES}
)
# target_link_libraries(hello_sub
#   ${catkin_LIBRARIES}
# )

cmake_minimum_required(VERSION 3.0.2)
project(jiwook_v1)

add_compile_options(-std=c++17)


find_package(catkin REQUIRED COMPONENTS
   roscpp
   std_msgs

)

catkin_package(

 CATKIN_DEPENDS roscpp std_msgs

)

###########
## Build ##
###########


include_directories(
include
  ${catkin_INCLUDE_DIRS}

)

add_executable(hello src/hello.cpp)
add_executable(hello_sub src/hello_sub.cpp)


add_dependencies(hello ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
add_dependencies(hello_sub ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})


target_link_libraries(hello
  ${catkin_LIBRARIES}
)
target_link_libraries(hello_sub
  ${catkin_LIBRARIES}
)

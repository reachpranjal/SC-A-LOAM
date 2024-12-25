cmake_minimum_required(VERSION 2.8.3)
project(aloam_velodyne)

# if(DEFINED ENV{CONDA_PREFIX})
#     set(CMAKE_PREFIX_PATH $ENV{CONDA_PREFIX} ${CMAKE_PREFIX_PATH})
#     set(Boost_NO_SYSTEM_PATHS TRUE)
#     set(yaml-cpp_DIR $ENV{CONDA_PREFIX}/lib/cmake/yaml-cpp)
#     set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
#     set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
#     set(CMAKE_INSTALL_RPATH $ENV{CONDA_PREFIX}/lib)
#     set(CMAKE_FIND_ROOT_PATH $ENV{CONDA_PREFIX})
#     set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY BOTH)
#     set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE BOTH)
# endif()

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -g")

find_package(catkin REQUIRED COMPONENTS
    geometry_msgs
    nav_msgs
    sensor_msgs
    roscpp
    rospy
    rosbag
    std_msgs
    image_transport
    cv_bridge
    tf
    pcl_conversions
    pcl_ros
)

find_package(PCL REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Ceres REQUIRED)
find_package(OpenMP REQUIRED)
find_package(GTSAM REQUIRED QUIET)
find_package(yaml-cpp REQUIRED)
# find_package(yaml-cpp REQUIRED PATHS ${CMAKE_PREFIX_PATH} NO_DEFAULT_PATH)

include_directories(
    include
    ${catkin_INCLUDE_DIRS}
    ${PCL_INCLUDE_DIRS}
    ${CERES_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    ${GTSAM_INCLUDE_DIR}
    $ENV{CONDA_PREFIX}/include
)

catkin_package(
    CATKIN_DEPENDS geometry_msgs nav_msgs roscpp rospy std_msgs
    DEPENDS EIGEN3 PCL
    INCLUDE_DIRS include
)

function(add_executable_with_conda name)
    add_executable(${name} ${ARGN})
    if(DEFINED ENV{CONDA_PREFIX})
        set_target_properties(${name} PROPERTIES
            INSTALL_RPATH "$ENV{CONDA_PREFIX}/lib"
            BUILD_WITH_INSTALL_RPATH TRUE
        )
    endif()
endfunction()

add_executable_with_conda(ascanRegistration src/scanRegistration.cpp)
target_link_libraries(ascanRegistration
    ${catkin_LIBRARIES}
    ${PCL_LIBRARIES}
)

add_executable_with_conda(alaserOdometry src/laserOdometry.cpp)
target_link_libraries(alaserOdometry
    ${catkin_LIBRARIES}
    ${PCL_LIBRARIES}
    ${CERES_LIBRARIES}
)

add_executable_with_conda(alaserMapping src/laserMapping.cpp)
target_link_libraries(alaserMapping
    ${catkin_LIBRARIES}
    ${PCL_LIBRARIES}
    ${CERES_LIBRARIES}
)

add_executable_with_conda(alaserPGO
    src/laserPosegraphOptimization.cpp
    include/scancontext/Scancontext.cpp
)
target_compile_options(alaserPGO
    PRIVATE ${OpenMP_CXX_FLAGS}
)
target_link_libraries(alaserPGO
    ${catkin_LIBRARIES}
    ${PCL_LIBRARIES}
    ${CERES_LIBRARIES}
    ${OpenMP_CXX_FLAGS}
    gtsam
)

add_executable_with_conda(kittiHelper src/kittiHelper.cpp)
target_link_libraries(kittiHelper
    ${catkin_LIBRARIES}
    ${PCL_LIBRARIES}
    ${OpenCV_LIBS}
)

add_executable_with_conda(kittiTransformBroadcaster src/kittiTransformBroadcaster.cpp)
target_link_libraries(kittiTransformBroadcaster
    ${catkin_LIBRARIES}
    yaml-cpp
)

add_executable_with_conda(lidar_localization
    src/lidarLocalization.cpp
    include/scancontext/Scancontext.cpp
)
target_compile_options(lidar_localization
    PRIVATE ${OpenMP_CXX_FLAGS}
)
target_link_libraries(lidar_localization
    ${catkin_LIBRARIES}
    ${PCL_LIBRARIES}
    ${CERES_LIBRARIES}
    ${OpenMP_CXX_FLAGS}
    gtsam
)
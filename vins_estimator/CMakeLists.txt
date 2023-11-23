cmake_minimum_required(VERSION 2.8.3)
project(vins)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-std=c++14 -fopenmp")
#-DEIGEN_USE_MKL_ALL")
set(CMAKE_CXX_FLAGS_RELEASE "-O0 -Wall -g")
set(CMAKE_CXX_FLAGS_DEBUG "-fdiagnostics-color=auto ")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-variable -Wno-unused-but-set-variable") # 关闭未使用变量警告
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-deprecated-declarations") # ceres-solver版本过新，api变更警告
set(CMAKE_EXPORT_COMPILE_COMMANDS true) # 生成compile_commands.json


set(ENABLE_BACKWARD true)
set(ENABLE_VWORKS false)
set(DETECT_CUDA true)
set(ENABLE_VPI true)
#SET("OpenCV_DIR"  "/usr/local/share/OpenCV/")

find_package(OpenMP REQUIRED)

find_package(catkin REQUIRED COMPONENTS
    message_generation
    message_runtime
    roscpp
    std_msgs
    geometry_msgs
    nav_msgs
    tf
    cv_bridge_local
    camera_models
    image_transport
    nodelet
    message_filters
    sensor_msgs
)

if(ENABLE_BACKWARD)
    set(LIBDW "dw")
else()
    set(LIBDW "")
endif()

if(ENABLE_VPI)
    find_package(vpi REQUIRED)
    message(WARNING "VPI detected, automatically USE OpenCV with VPI support")
    set(LIBVPI "vpi")
else()
    add_definitions(-D WITHOUT_NVIDIA_VPI)
    set(LIBVPI "")
endif()

if(DETECT_CUDA)
    find_package(CUDA)
    if (CUDA_FOUND)
        message("CUDA detected, automatically USE OpenCV with CUDA support")
        set(LIBSGM "sgm")
        find_package(OpenCV REQUIRED cudaimgproc)
    else()
        add_definitions(-D WITHOUT_CUDA)
        message(WARNING "CUDA not detected, automatically USE OpenCV without CUDA support")
        set(LIBSGM "")
        find_package(OpenCV REQUIRED )
    endif()
else()
    message(WARNING "CUDA not detected, automatically USE OpenCV without CUDA support")
    set(LIBSGM "")
    find_package(OpenCV REQUIRED )
    add_definitions(-D WITHOUT_CUDA)
endif()


if(ENABLE_VWORKS)
    find_package(VisionWorks REQUIRED)
    #Set path to visionworks sample here
    # include_directories(
        # /home/xuhao/VisionWorks-1.6-Samples/nvxio/include
    # )
    # link_directories(/home/xuhao/VisionWorks-1.6-Samples/libs/x86_64/linux/release/)
    add_definitions(-DWITH_VWORKS)
else()
endif()

find_package(Ceres REQUIRED)


include_directories(${catkin_INCLUDE_DIRS} ${CERES_INCLUDE_DIRS})

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
find_package(Eigen3)

include_directories(
    include
  ${catkin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
  ${OpenCV_INCLUDE_DIRS}
  ${VisionWorks_INCLUDE_DIRS}
  src/
)

message(STATUS "OpenCV library status:")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")
message(STATUS "    catkin libraries: ${catkin_LIBRARIES}")
#catkin_package()

add_message_files(
    FILES
    VIOKeyframe.msg
    FlattenImages.msg
)

generate_messages(
    DEPENDENCIES
    geometry_msgs
    std_msgs
    sensor_msgs
)

catkin_package()

# add_executable(vins_node 
#     src/rosNodeFisheye.cpp
#     src/fisheyeNode.cpp
#     # depth_generation
#     src/depth_generation/depth_estimator.cpp
#     src/depth_generation/stereo_matching.cpp
#     src/depth_generation/depth_camera_manager.cpp
#     src/depth_generation/color_disparity_graph.cpp
#     src/depth_generation/stereo_online_calib.cpp
#     # estimator
#     src/estimator/estimator.cpp
#     src/estimator/feature_manager.cpp
#     src/estimator/parameters.cpp
#     # factor
#     src/factor/marginalization_factor.cpp
#     src/factor/pose_local_parameterization.cpp
#     src/factor/projection_factor.cpp
#     src/factor/projectionOneFrameTwoCamFactor.cpp
#     src/factor/projectionTwoFrameOneCamFactor.cpp
#     src/factor/projectionTwoFrameTwoCamFactor.cpp
#     # featureTracker
#     src/featureTracker/feature_tracker_fisheye_cuda.cpp
#     src/featureTracker/feature_tracker_fisheye_vworks.cpp
#     src/featureTracker/feature_tracker_fisheye.cpp
#     src/featureTracker/feature_tracker_pinhole.cpp
#     src/featureTracker/feature_tracker.cpp
#     src/featureTracker/vworks_feature_tracker.cpp
#     # initial
#     src/initial/initial_aligment.cpp
#     src/initial/initial_ex_rotation.cpp
#     src/initial/initial_sfm.cpp
#     src/initial/solve_5pts.cpp
#     # utility
#     src/utility/CameraPoseVisualization.cpp
#     src/utility/ros_utility.cpp
#     src/utility/utility.cpp
#     src/utility/visualization.cpp
# )
# target_link_libraries(vins_node ${catkin_LIBRARIES} ${OpenCV_LIBS}  ${LIBSGM} ${LIBDW} ${CERES_LIBRARIES} OpenMP::OpenMP_CXX)


add_library(vins_frontend SHARED
    src/featureTracker/feature_tracker_base.cpp
    src/featureTracker/feature_tracker_pinhole.cpp
    src/featureTracker/feature_tracker_vpi.cpp
)


add_library(vins_factors_lib SHARED
    src/factor/pose_local_parameterization.cpp
    src/factor/projectionTwoFrameOneCamFactor.cpp
    src/factor/projectionTwoFrameTwoCamFactor.cpp
    src/factor/projectionOneFrameTwoCamFactor.cpp
)

add_library(vins_lib SHARED
    src/estimator/feature_manager.cpp
    src/factor/marginalization_factor.cpp
    src/utility/utility.cpp
    src/utility/visualization.cpp
    src/utility/CameraPoseVisualization.cpp
    src/utility/ros_utility.cpp
    src/initial/solve_5pts.cpp
    src/initial/initial_aligment.cpp
    src/initial/initial_sfm.cpp
    src/initial/initial_ex_rotation.cpp
)


add_library(vins_params_lib src/estimator/parameters.cpp)

add_library(estimator_lib SHARED
    src/estimator/estimator.cpp
)

add_library(fisheyeNode_lib SHARED
     src/vins_base.cpp)

target_link_libraries(vins_lib ${catkin_LIBRARIES} ${OpenCV_LIBS} ${CERES_LIBRARIES} ${LIBDW} ${LIBVPI})
target_link_libraries(vins_params_lib ${catkin_LIBRARIES} ${OpenCV_LIBS} ${CERES_LIBRARIES} ${LIBDW} ${LIBVPI})
add_dependencies(vins_lib vins_generate_messages_cpp)
target_link_libraries(vins_frontend ${catkin_LIBRARIES} ${OpenCV_LIBS}  ${LIBDW} OpenMP::OpenMP_CXX ${LIBVPI})
target_link_libraries(estimator_lib vins_params_lib vins_lib ${catkin_LIBRARIES} ${OpenCV_LIBS}  ${LIBDW} ${LIBVPI})
target_link_libraries(fisheyeNode_lib ${catkin_LIBRARIES} ${OpenCV_LIBS}  ${LIBDW} OpenMP::OpenMP_CXX ${LIBVPI})


# add_executable(vins_node src/rosNodeTest.cpp)
# target_link_libraries(vins_node vins_lib estimator_lib vins_frontend vins_factors_lib vins_params_lib)

add_executable(vins_node src/vins_node.cpp)
target_link_libraries(vins_node vins_lib fisheyeNode_lib estimator_lib vins_frontend vins_factors_lib vins_params_lib)

add_executable(vins_test src/vins_test.cpp)
target_link_libraries(vins_test vins_lib fisheyeNode_lib estimator_lib vins_frontend vins_factors_lib vins_params_lib)

# add_library(vins_nodelet_lib src/rosNodelet.cpp)
# target_link_libraries(vins_nodelet_lib vins_lib fisheyeNode_lib estimator_lib vins_frontend vins_factors_lib vins_params_lib OpenMP::OpenMP_CXX)

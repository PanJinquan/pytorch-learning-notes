cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(opencv_libtorchDemo)

SET(OpenCV_DIR D:/opencv/build)
find_package(OpenCV REQUIRED)       # 查找OpenCV

SET(Torch_DIR "./libtorch")
set(CMAKE_PREFIX_PATH ${Torch_DIR})
# 若build出错：提示：特性不适用于任何实体->C10_DEPRECATED_USING
#S ET( CMAKE_CXX_FLAGS "-std=c++11 -O3")
# set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -pthread")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")
find_package(Torch REQUIRED)

add_executable(opencv_libtorch-app opencv_libtorch.cpp)
target_link_libraries(opencv_libtorch-app "${TORCH_LIBRARIES}" "${OpenCV_LIBS}")
set_property(TARGET opencv_libtorch-app PROPERTY CXX_STANDARD 11)

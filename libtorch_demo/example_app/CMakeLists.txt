cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(torchDemo)

SET(Torch_DIR "./libtorch")
set(CMAKE_PREFIX_PATH ${Torch_DIR})

find_package(Torch REQUIRED)

add_executable(torchDemo-app main.cpp)
target_link_libraries(torchDemo-app "${TORCH_LIBRARIES}")
set_property(TARGET torchDemo-app PROPERTY CXX_STANDARD 11)

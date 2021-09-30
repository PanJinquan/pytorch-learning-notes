# For more information about using CMake with Android Studio, read the
# documentation: https://d.android.com/studio/projects/add-native-code.html

# Sets the minimum version of CMake required to build the native library.

cmake_minimum_required(VERSION 3.4.1)


set(OpenCV_DIR ${CMAKE_SOURCE_DIR}/../OpenCV-android-sdk/sdk/native/jni)
find_package(OpenCV REQUIRED)
include_directories( ${CMAKE_SOURCE_DIR}/../OpenCV-android-sdk/sdk/native/jni/include)

add_library( # Sets the name of the library.
              imagePro-lib

              # Sets the library as a shared library.
              SHARED

              # Provides a relative path to your source file(s).
              src/main/cpp/imagePro.cpp
              src/main/cpp/AndroidDebug.cpp
              )


find_library( # Sets the name of the path variable.
              log-lib

              # Specifies the name of the NDK library that
              # you want CMake to locate.
              log )

##################################################################################################
#add_library(
#             native-lib
#              SHARED
#              src/main/cpp/native-lib.cpp
#              )

find_library(
          android-lib
          android
          )

include(AndroidNdkModules)
android_ndk_import_module_cpufeatures()


add_library(
    caffe2
    STATIC
    IMPORTED
    )
set_target_properties(
    caffe2
    PROPERTIES IMPORTED_LOCATION
    ${CMAKE_CURRENT_LIST_DIR}/src/main/jniLibs/${ANDROID_ABI}/cur_version/libcaffe2.a
    )
add_library(
    thread_pool
    STATIC
    IMPORTED
    )
set_target_properties(
    thread_pool
    PROPERTIES IMPORTED_LOCATION
    ${CMAKE_CURRENT_LIST_DIR}/src/main/jniLibs/${ANDROID_ABI}/cur_version/libpthreadpool.a
    )

add_library(
    protobuf
    STATIC
    IMPORTED
    )
set_target_properties(
    protobuf
    PROPERTIES IMPORTED_LOCATION
    ${CMAKE_CURRENT_LIST_DIR}/src/main/jniLibs/${ANDROID_ABI}/cur_version/libprotobuf.a
    )

add_library(
    NNPACK
    STATIC
    IMPORTED
    )
set_target_properties(
    NNPACK
    PROPERTIES IMPORTED_LOCATION
    ${CMAKE_CURRENT_LIST_DIR}/src/main/jniLibs/${ANDROID_ABI}/cur_version/libnnpack.a
    )

add_library(
    cpuinfo
    STATIC
    IMPORTED
    )
set_target_properties(
    cpuinfo
    PROPERTIES IMPORTED_LOCATION
    ${CMAKE_CURRENT_LIST_DIR}/src/main/jniLibs/${ANDROID_ABI}/cur_version/libcpuinfo.a
    )

add_library(
    onnx
    STATIC
    IMPORTED
    )
set_target_properties(
    onnx
    PROPERTIES IMPORTED_LOCATION
    ${CMAKE_CURRENT_LIST_DIR}/src/main/jniLibs/${ANDROID_ABI}/cur_version/libonnx.a
    )

add_library(
    onnx_proto
    STATIC
    IMPORTED
    )
set_target_properties(
    onnx_proto
    PROPERTIES IMPORTED_LOCATION
    ${CMAKE_CURRENT_LIST_DIR}/src/main/jniLibs/${ANDROID_ABI}/cur_version/libonnx_proto.a
    )

add_library(
    caffe2_proto
    STATIC
    IMPORTED
    )
set_target_properties(
    caffe2_proto
    PROPERTIES IMPORTED_LOCATION
    ${CMAKE_CURRENT_LIST_DIR}/src/main/jniLibs/${ANDROID_ABI}/cur_version/libcaffe2_protos.a
    )

add_library(
    nnpack_ref
    STATIC
    IMPORTED
    )
set_target_properties(
    nnpack_ref
    PROPERTIES IMPORTED_LOCATION
    ${CMAKE_CURRENT_LIST_DIR}/src/main/jniLibs/${ANDROID_ABI}/cur_version/libnnpack_reference_layers.a
    )

add_library(
    nomnigraph
    STATIC
    IMPORTED
    )
set_target_properties(
    nomnigraph
    PROPERTIES IMPORTED_LOCATION
    ${CMAKE_CURRENT_LIST_DIR}/src/main/jniLibs/${ANDROID_ABI}/cur_version/libnomnigraph.a
    )

add_library(
    onnxifi_dummy
    SHARED
    IMPORTED
    )
set_target_properties(
    onnxifi_dummy
    PROPERTIES IMPORTED_LOCATION
    ${CMAKE_CURRENT_LIST_DIR}/src/main/jniLibs/${ANDROID_ABI}/cur_version/libonnxifi_dummy.so
    )

add_library(
    onnxifi_loader
    STATIC
    IMPORTED
    )
set_target_properties(
    onnxifi_loader
    PROPERTIES IMPORTED_LOCATION
    ${CMAKE_CURRENT_LIST_DIR}/src/main/jniLibs/${ANDROID_ABI}/cur_version/libonnxifi_loader.a
    )

add_library(
    protobuf-lite
    STATIC
    IMPORTED
    )
set_target_properties(
    protobuf-lite
    PROPERTIES IMPORTED_LOCATION
    ${CMAKE_CURRENT_LIST_DIR}/src/main/jniLibs/${ANDROID_ABI}/cur_version/libprotobuf-lite.a
    )


include_directories( src/main/cpp )

find_library(
     log-lib
     log
     )

target_link_libraries( # Specifies the target library.
                       imagePro-lib

                       -Wl,--whole-archive
                       caffe2
                       -Wl,--no-whole-archive
                       caffe2_proto
                       NNPACK
                       protobuf
                       thread_pool
                       cpuinfo
                       onnx
                       onnx_proto
                       onnxifi_loader
                       nomnigraph
                       ${android-lib}

                       # Links the target library to the log library
                       # included in the NDK.
                       ${log-lib}
                       ${OpenCV_LIBS})



# OpencvDemo说明
## 一：OpenCV-android-sdk说明
- 由于Github不支持超过100M的大文件上传，因此原Demo的OpenCV-android-sdk的apk文件夹被删除了，当然这个不影响项目使用
- app/CMakeLists.txt中已经设置好OpenCV-android-sdk的路径， Demo的OpenCV版本已经升级到OpenCV-3.4.2-android-sdk，
若需要修改OpenCV的版本，请下载对应版本：https://opencv.org/releases.html 
- CMakeLists设置：
> set(OpenCV_DIR ${CMAKE_SOURCE_DIR}/../OpenCV-android-sdk/sdk/native/jni) 
> find_package(OpenCV REQUIRED)   
> include_directories( ${CMAKE_SOURCE_DIR}/../OpenCV-android-sdk/sdk/native/jni/include) 
## 二： OpenCV-Contrib-Android-Demo
- Opencv3.x以后，已经把很多功能模块放在contrib中，要想移植opencv contrib到Android需要自己编译，
- 这个过程还是相当麻烦的。如果你想支持opencv contrib开发，可以下载本人已经编译且移植好的Android Demo:
- Github地址：https://github.com/PanJinquan/OpenCV-Contrib-Android-Demo

## 三：参考资料：
- 1.https://blog.csdn.net/guyuealian/article/details/78374708

#ifndef DETECT_DEBUG_H
#define DETECT_DEBUG_H
#include "opencv2/opencv.hpp"
#include <chrono>

#define  millisecond 1000000.0

using namespace std;
//debug info ON-OFF
#define __DEBUG__ON
#ifdef  __DEBUG__ON
#define __DEBUG__WIN__OFF         //Window debug:print debug info
#define __DEBUG__IMSHOW__ON       //show debug images
#define __DEBUG__IMWRITE__OFF       //write debug images
#define __DEBUG__TIME__ON          //run times test on/off
#define __DEBUG__ANDROID__ON     //android debug on/off

//#include <assert.h>
//#define DEBUG_ASSERT(...) assert( __VA_ARGS__)
//#define DEBUG_CV_ASSERT(...) CV_Assert( __VA_ARGS__)

#else
#define __DEBUG__ON(format,...)
#endif

//print debug info
#ifdef  __DEBUG__WIN__ON
//#define DEBUG_PRINT(...) printf("File: %s, Line: %05d: "format"", __FILE__,__LINE__, ##__VA_ARGS__)
#define DEBUG_PRINT(...) printf( __VA_ARGS__);printf("\n")
#else
#define DEBUG_PRINT(format,...)
#endif



//show debug images
#ifdef  __DEBUG__IMSHOW__ON
#define DEBUG_IMSHOW(...) showImages(__VA_ARGS__)
#else
#define DEBUG_IMSHOW(format,...)
#endif

//write debug images
#ifdef  __DEBUG__IMWRITE__ON
#define DEBUG_IMWRITE(...) saveImage(__VA_ARGS__)
#else
#define DEBUG_IMWRITE(format,...)
#endif

//write debug images
#ifdef  __DEBUG__ANDROID__ON
#include <android/log.h>
// Define the LOGI and others for print debug infomation like the log.i in java
#define LOG_TAG    "SmartAlbum -- JNILOG"
//#undef LOG
#define LOGI(...)  __android_log_print(ANDROID_LOG_INFO,LOG_TAG, __VA_ARGS__)
#define LOGD(...)  __android_log_print(ANDROID_LOG_DEBUG,LOG_TAG, __VA_ARGS__)
#define LOGW(...)  __android_log_print(ANDROID_LOG_WARN,LOG_TAG, __VA_ARGS__)
#define LOGE(...)  __android_log_print(ANDROID_LOG_ERROR,LOG_TAG, __VA_ARGS__)
#define LOGF(...)  __android_log_print(ANDROID_LOG_FATAL,LOG_TAG, __VA_ARGS__)
#else
#ifdef __DEBUG__WIN__ON
#define LOGI(...)  printf( __VA_ARGS__); printf("\n")
#define LOGD(...)  printf( __VA_ARGS__); printf("\n")
#define LOGW(...)  printf( __VA_ARGS__); printf("\n")
#define LOGE(...)  printf( __VA_ARGS__); printf("\n")
#define LOGF(...)  printf( __VA_ARGS__); printf("\n")
#else
#define LOGI(...)
#define LOGD(...)
#define LOGW(...)
#define LOGE(...)
#define LOGF(...)
#endif
#endif

//run times test...
#ifdef  __DEBUG__TIME__ON
#define LOG_TIME  LOGE
#define RUN_TIME(time_)  (double)(time_).count()/millisecond
//#define RUN_TIME(...)  getTime_MS( __VA_ARGS__)

//设置计算运行时间的宏定义
#define DEBUG_TIME(time_) auto time_ =std::chrono::high_resolution_clock::now()
#define DEBUG_TIME_PRINT(time_) printf("run time: %s=%3.3f ms\n", #time_,(double)(time_).count()/millisecond)
#else
#define DEBUG_TIME(time_)
#endif

template<typename TYPE>
void PRINT_1D(string name,TYPE *p1, int len) {
    printf("%s", name.c_str());
    for (int i = 0; i < len; i++) {
        printf("%f,", p1[i]);
    }
    cout << endl;
}


void showImages(const char *imageName, cv::Mat image);
void showImages(const char *imageName, cv::Mat img, cv::Rect face);
void showImages(const char *imageName, cv::Mat img, cv::Rect face, std::vector<cv::Point> pts);
void showImages(const char *imageName, cv::Mat img, std::vector<cv::Point> pts);




void saveImage(const char *imageName, cv::Mat image);
void saveImage(const char *imageName, cv::Mat image, std::vector<int> para);
void saveImage(const char *imageName, cv::Mat image, cv::Rect face, std::vector<cv::Point> pts);
void saveImage(const char *imageName, cv::Mat img, cv::Rect face);

vector<string> getFilesList(string dir);
void writeDatatxt(string path, string data, bool bCover=false);

#ifdef linux

#define _LINUX
#define separator "/"

#endif

#ifdef _WIN32//__WINDOWS_

#define _WINDOWS
#define separator  "\\"
#endif


#endif

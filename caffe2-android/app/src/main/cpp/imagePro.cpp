//
// Created by panjq1 on 2017/10/22.
//
#include <string>
#include <android/log.h>
#include "opencv2/opencv.hpp"
#include "AndroidDebug.h"
#include "com_panjq_opencv_alg_ImagePro.h"


#include <jni.h>
#include <algorithm>
#define PROTOBUF_USE_DLLS 1
#define CAFFE2_USE_LITE_PROTO 1
#include <caffe2/predictor/predictor.h>
#include <caffe2/core/operator.h>
#include <caffe2/core/timer.h>

#include "caffe2/core/init.h"
#include  "caffe2/core/tensor.h"

#include <android/asset_manager.h>
#include <android/asset_manager_jni.h>
#include "classes.h"

#define IMG_H 224
#define IMG_W 224
#define IMG_C 3
#define MAX_DATA_SIZE IMG_H * IMG_W * IMG_C
#define LOG_TAG    "---JNILOG---" // 这个是自定义的LOG的标识
using namespace cv;


static caffe2::NetDef _initNet, _predictNet;
static caffe2::Predictor *_predictor;
static char raw_data[MAX_DATA_SIZE];
static float input_data[MAX_DATA_SIZE];
static cv::Mat input_mat=cv::Mat::zeros(cv::Size(IMG_W,IMG_H),CV_32FC3);
static caffe2::Workspace ws;

/***
 *
 * @param mgr
 * @param net
 * @param filename
 */
void loadToNetDef(AAssetManager* mgr, caffe2::NetDef* net, const char *filename) {
    AAsset* asset = AAssetManager_open(mgr, filename, AASSET_MODE_BUFFER);
    assert(asset != nullptr);
    const void *data = AAsset_getBuffer(asset);
    assert(data != nullptr);
    off_t len = AAsset_getLength(asset);
    assert(len != 0);
    if (!net->ParseFromArray(data, len)) {
        LOGE("Couldn't parse net from data.\n");
    }
    AAsset_close(asset);
}


extern "C"
JNIEXPORT jintArray JNICALL Java_com_panjq_opencv_alg_ImagePro_caffe2init(
        JNIEnv* env,
        jobject /* this */,
        jobject assetManager) {
    AAssetManager *mgr = AAssetManager_fromJava(env, assetManager);
    LOGE("Attempting to load protobuf netdefs...");
//    loadToNetDef(mgr, &_initNet,   "squeeze_init_net_6.pb");
//    loadToNetDef(mgr, &_predictNet,"squeeze_pred_net_6.pb");
    loadToNetDef(mgr, &_initNet,   "init_net.pb");
    loadToNetDef(mgr, &_predictNet,"predict_net.pb");
    LOGE("done.");
    LOGE("Instantiating predictor...");
    _predictor = new caffe2::Predictor(_initNet, _predictNet);
    LOGE("done.");
}


template<typename _Tp>
int softmax(const _Tp* src, _Tp* dst, int length)
{
    //为了避免溢出，需要减去最大值
    const _Tp max_value = *std::max_element(src, src + length);
    _Tp denominator{ 0 };

    for (int i = 0; i < length; ++i) {
        dst[i] = std::exp(src[i] - max_value);
        denominator += dst[i];
    }

    for (int i = 0; i < length; ++i) {
        dst[i] /= denominator;
    }
    return 0;
}
template<typename _Tp> void printVector(_Tp data){
    LOGE("_________________________________");
    for(size_t i =0;i<data.size();i++){
        LOGE("data:%3.3f,",data[i]);
    }
}
extern "C"
JNIEXPORT void JNICALL Java_com_panjq_opencv_alg_ImagePro_caffe2inference
        (JNIEnv *, jobject, jlong matAddrSrcImage, jlong matAddrDestImage) {
    DEBUG_TIME(T0);
    Mat &src_image = *(Mat *) matAddrSrcImage;//matAddrSrcImage是RGBA格式
    Mat &destImage = *(Mat *) matAddrDestImage;
    cv::Mat input_image;
    cv::cvtColor(src_image, input_image, CV_RGBA2RGB);
    cv::resize(input_image, input_image, cv::Size(IMG_W, IMG_H));
    //    cv::imwrite("/storage/emulated/0/standard-001.jpg",srcImage);

    int r,g,b;
    int chans=input_image.channels();
    LOGE("resize,srcImage.shape:[%d,%d,%d]\n",input_image.rows,input_image.cols,chans);
    //数据预处理方法[1]
//    for (size_t i = 0; i < input_image.rows; i++) {
//        uchar* ptr = input_image.ptr<uchar>(i);
//        for (size_t j = 0; j < input_image.cols; j ++) {
//            r = ptr[j * 3];
//            g = ptr[j * 3+1];
//            b = ptr[j * 3+2];
////            float b_mean = 0.406, b_std = 0.225;
////            float g_mean = 0.456, g_std = 0.224;
////            float r_mean = 0.485, r_std = 0.229;
//            input_data[i * IMG_W + j + 0*IMG_H * IMG_W] = r/255.0;
//            input_data[i * IMG_W + j + 1*IMG_H * IMG_W] = g/255.0;
//            input_data[i * IMG_W + j + 2*IMG_H * IMG_W] = b/255.0;
//        }
//    }
    //数据预处理方法[2]
    input_mat=input_image.clone();
    input_mat.convertTo(input_mat, CV_32FC3, 1/255.0);
    std::vector<cv::Mat> Channels;
    cv::split(input_mat, Channels);
    memcpy(input_data,Channels.at(0).data, IMG_H *IMG_W* sizeof(float));//r分量
    memcpy(input_data+IMG_H *IMG_W,Channels.at(1).data, IMG_H *IMG_W* sizeof(float));//b分量
    memcpy(input_data+2*IMG_H *IMG_W,Channels.at(2).data, IMG_H *IMG_W* sizeof(float));//g分量


    LOGE("input_mat,.shape:[%d,%d,%d]\n",input_mat.rows,input_mat.cols,input_mat.channels());
    bool infer_HWC=false;//input_vec是{ batch_size, IMG_C, IMG_H, IMG_W}格式
    caffe2::TensorCPU input;
    if (infer_HWC) {
        input.Resize(std::vector<int>({IMG_H, IMG_W, IMG_C}));
    } else {
        input.Resize(std::vector<int>({1, IMG_C, IMG_H, IMG_W}));
    }
    memcpy(input.mutable_data<float>(), input_data, IMG_H * IMG_W * IMG_C * sizeof(float));
//    memcpy(input.mutable_data<float>(), input_mat.data, IMG_H * IMG_W * IMG_C * sizeof(float));

    caffe2::Predictor::TensorVector input_tensor{&input};
    caffe2::Predictor::TensorVector output_tensor;
    _predictor->run(input_tensor, &output_tensor);

    // Find the top-k results manually.
    int k = 5;
    auto output = output_tensor[0];
    std::vector<float> output_vector(output->size());
    std::vector<float> preds(output->size());
    for(auto i=0; i < output->size(); ++i) {
        float tmp= output->template data<float>()[i];
        LOGE("tmp:%3.3f",tmp);
        output_vector[i]=tmp;
    }
    //softmax
    softmax(output_vector.data(), preds.data(),output_vector.size());
    //排序top-k
    std::vector<int> idxs(preds.size());
    std::iota(idxs.begin(), idxs.end(), 0);
    std::sort(idxs.begin(), idxs.end(),
              [&preds](const int &idx1, const int &idx2) {
                  return preds[idx1] > preds[idx2];
              }
    );
    DEBUG_TIME(T1);
    //打印输出结果
    std::ostringstream stringStream;
    stringStream  << " preds\n";
    for (auto j = 0; j < k; ++j) {
        int idx = idxs[j];
        stringStream << j << ": " << imagenet_classes[idx] << " - " << preds[idx] * 100 << "%\n";
    }
    cv::putText(src_image,imagenet_classes[idxs[0]],cv::Point(10, 50), cv::FONT_HERSHEY_COMPLEX, 1, cv::Scalar(0, 0,255), 2,8);
    destImage=src_image;
    LOGE("preds:%s\n",stringStream.str().c_str());
    LOGE("Run time:jniImagePro3=%3.3fms\n",RUN_TIME(T1-T0));
}

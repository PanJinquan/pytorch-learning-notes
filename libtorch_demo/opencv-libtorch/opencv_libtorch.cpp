#include "torch/script.h"
#include "torch/torch.h"
#include <opencv2/opencv.hpp>

#include <iostream>
#include <memory>
using namespace std;


//#define DEVICE torch::kCUDA
#define DEVICE torch::kCPU

cv::Mat resize_with_ratio(cv::Mat& img, int dst_w = 224,int dst_h = 224)
{
	// resize并保持图像比例不变
	cv::Mat temImage;
	int w = img.cols;
	int h = img.rows;
	float t = 1.;
	float len = t * std::max(w, h);
	cv::Mat image = cv::Mat(cv::Size(dst_w, dst_h), CV_8UC3, cv::Scalar(128, 128, 128));
	cv::Mat imageROI;
	if (len == w)
	{
		float ratio = (float)h / (float)w;
		cv::resize(img, temImage, cv::Size(224, 224 * ratio), 0, 0, cv::INTER_LINEAR);
		imageROI = image(cv::Rect(0, ((dst_h - 224 * ratio) / 2), temImage.cols, temImage.rows));
		temImage.copyTo(imageROI);
	}
	else
	{
		float ratio = (float)w / (float)h;
		cv::resize(img, temImage, cv::Size(224 * ratio, 224), 0, 0, cv::INTER_LINEAR);
		imageROI = image(cv::Rect(((dst_w - 224 * ratio) / 2), 0, temImage.cols, temImage.rows));
		temImage.copyTo(imageROI);
	}

	return image;
}

int main()
{
	string model_path = "../../models/trace_model.pth";
	string image_path = "../../test_image/1.jpg";

	//读取图片
	cv::Mat image = cv::imread(image_path);
	cv::Mat resize_image = resize_with_ratio(image);
	cv::Mat input;
	cv::cvtColor(resize_image, input, cv::COLOR_BGR2RGB);

	// 读取我们的权重信息
	std::shared_ptr<torch::jit::script::Module> model = torch::jit::load(model_path);
	model->to(DEVICE);
	assert(model != nullptr);

	// 下方的代码即将图像转化为Tensor，随后导入模型进行预测
	torch::Tensor tensor_image = torch::from_blob(input.data, { 1,input.rows, input.cols,3 }, torch::kByte);
	tensor_image = tensor_image.permute({ 0,3,1,2 });
	tensor_image = tensor_image.toType(torch::kFloat);
	tensor_image = tensor_image.div(255);
	tensor_image = tensor_image.to(DEVICE);

	//获得输出结果
	torch::Tensor result = model->forward({ tensor_image }).toTensor();
	std::cout << result.slice(/*dim=*/1, /*start=*/0, /*end=*/5) << '\n';
	auto max_result = result.max(1, true);
	auto max_index = std::get<1>(max_result).item<float>();
	cv::putText(image, to_string(max_index), { 40, 50 }, cv::FONT_HERSHEY_PLAIN, 2.0, cv::Scalar(0, 255, 0), 2);
	imshow("image", image);    //显示摄像头的数据
	cv::waitKey(0);
}
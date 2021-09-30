

#include <fstream>
#include <sstream>
#include <iostream>

#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
using namespace cv;
using namespace dnn;
using namespace std;

string pro_dir = "E:/git/opencv-learning-tutorials/"; //项目根目录
vector<string> classes;


// Get the names of the output layers
vector<String> getOutputsNames(const Net& net)
{
	static vector<String> names;
	if (names.empty())
	{
		//Get the indices of the output layers, i.e. the layers with unconnected outputs
		vector<int> outLayers = net.getUnconnectedOutLayers();

		//get the names of all the layers in the network
		vector<String> layersNames = net.getLayerNames();

		// Get the names of the output layers in names
		names.resize(outLayers.size());
		for (size_t i = 0; i < outLayers.size(); ++i)
			names[i] = layersNames[outLayers[i] - 1];
	}
	return names;
}

void softmax(int k, float *x) {
	double max = 0.0;
	double sum = 0.0;
	for (int i = 0; i < k; i++) if (max < x[i]) max = x[i];
#pragma omp parallel for  
	for (int i = 0; i < k; i++) {
		x[i] = exp(x[i] - max);//防止数据溢出
		sum += x[i];
	}
#pragma omp parallel for  
	for (int i = 0; i < k; i++) x[i] /= sum;
}


void printArray1D(float*data,int length) {
	printf("[");
	for (size_t i = 0; i < length; i++)
	{
		printf("%f, ", data[i]);
	}
	printf("]");
}

void postprocess(Mat& frame, const vector<Mat>& outs)
{
	for (size_t i = 0; i < outs.size(); ++i)
	{
		cv::Mat score = outs[i];
		std::cout << score << endl;
		float* data = (float*)score.data;
		softmax(5, data);
		printArray1D(data, 5);
	}
}




void opencv_dnn_onnx_predict(string image_path, string onnx_file, string classesFile) {
	// Initialize the parameters
	int inpWidth = 224;  // Width of network's input image
	int inpHeight = 224; // Height of network's input image


	// Load names of classes
	ifstream ifs(classesFile.c_str());
	string line;
	while (getline(ifs, line)) classes.push_back(line);

	// Load the network
	// Net net = readNetFromDarknet(modelConfiguration, onnx_file);
	Net net = readNetFromONNX(onnx_file);
	net.setPreferableBackend(DNN_BACKEND_OPENCV);
	net.setPreferableTarget(DNN_TARGET_OPENCL);

	// Open a video file or an image file or a camera stream.
	string str, outputFile;
	cv::Mat image = cv::imread(image_path);

	// Stop the program if reached end of video
	// Create a 4D blob from a frame.
	Mat blob;
	blobFromImage(image, blob, 1 / 255.0, cv::Size(inpWidth, inpHeight), Scalar(0, 0, 0), true, false);
	
	//Sets the input to the network
	net.setInput(blob);

	// Runs the forward pass to get output of the output layers
	vector<Mat> outs;
	net.forward(outs, getOutputsNames(net));
	postprocess(image, outs);//模型输出后处理
}

int main(int argc, char** argv)
{
	// Give the onnx files for the model
	String onnx_file = pro_dir + "data/models/onnx/resRegularBn/model.onnx";
	string image_path = pro_dir + "data/models/onnx/test_image/animal.jpg";
	string classesFile = pro_dir + "data/models/onnx/label.txt";
	opencv_dnn_onnx_predict(image_path, onnx_file, classesFile);
	cv::waitKey(0);
	return 0;
}

#include "torch/script.h"
#include "torch/torch.h"

#include <iostream>
#include <memory>

using namespace std;

int main(int argc, const char* argv[])
{
    if (argc != 2) {
        std::cerr << "usage: example-app <path-to-exported-script-module>\n";
        return -1;
    }

    // 读取我们的权重信息
    std::shared_ptr<torch::jit::script::Module> model = torch::jit::load(argv[1]);
	// model->to(at::kCUDA);
	// model->to(at::cpu);


    assert(model != nullptr);
    std::cout << "ok\n";

    // 建立一个输入，维度为(1,3,224,224)，并移动至cuda/cpu
    std::vector<torch::jit::IValue> inputs;
//    inputs.push_back(torch::ones({1, 3, 224, 224}).to(at::kCUDA));
    inputs.push_back(torch::ones({1, 3, 224, 224}));

// Execute the model and turn its output into a tensor.
    at::Tensor output = model->forward(inputs).toTensor();

    std::cout << output.slice(/*dim=*/1, /*start=*/0, /*end=*/5) << '\n';
}
## libtorch使用教程
### 1.下载 libtorch:https://pytorch.org/get-started/locally/
- 这里仅给出Ubuntu系统的编译过程，Windows也可以使用CMake build
```bash
wget https://download.pytorch.org/libtorch/nightly/cpu/libtorch-shared-with-deps-latest.zip
unzip libtorch-shared-with-deps-latest.zip
cp libtorch path/to/example_app

```
### 2. python libtorch_for_resnet18.py
> print(output[0, :5]):
```
tensor([-0.0391,  0.1145, -1.7968, -1.2343, -0.8190], grad_fn=<SliceBackward>)

```
### 3. build APP
> cd example_app <br/>
> mkdir build && cd build <br/>
> cmake .. <br/>
> make <br/>

Built target touchDemo-app <br/>

### 4. run APP

> ./torchDemo-app ../../models/model.pt <br/>

print info:

```
-0.0391  0.1145 -1.7968 -1.2343 -0.8190
[ Variable[CPUFloatType]{1,5} ]

```
### 5.opencv-libtorch
- 实现将OpenCV读取图像传递给libtorch进行预测

## 参考资料
> [利用Pytorch的C++前端(libtorch)读取预训练权重并进行预测] https://oldpan.me/archives/pytorch-c-libtorch-inference <br/>
> [Installing C++ Distributions of PyTorch] https://pytorch.org/cppdocs/installing.html <br/>
> [Loading a PyTorch Model in C++] https://pytorch.org/tutorials/advanced/cpp_export.html <br/>

import torch
import torch.nn.functional as F


# 定义一个构建神经网络的类
class Net(torch.nn.Module):  # 继承torch.nn.Module类
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()  # 获得Net类的超类（父类）的构造方法
        # 定义神经网络的每层结构形式
        # 各个层的信息都是Net类对象的属性
        self.hidden = torch.nn.Linear(n_feature, n_hidden)  # 隐藏层线性输出
        self.predict = torch.nn.Linear(n_hidden, n_output)  # 输出层线性输出
        self.norm = torch.nn.BatchNorm1d(1)

    # 将各层的神经元搭建成完整的神经网络的前向通路
    def forward(self, x):
        x = F.relu(self.hidden(x))  # 对隐藏层的输出进行relu激活
        x = self.predict(x)
        x = self.norm(x.view(int(x.size(0)), 1, -1))
        return x


if __name__ == '__main__':
    model = Net(16, 48, 5)
    model.eval()
    print(model.norm.training)
    dummy_inp = torch.randn(1, 16, device='cpu')
    input_names = ['input']
    output_names = ['output']
    torch.onnx.export(model, dummy_inp, './trt/norm_demo.onnx', verbose=True, input_names=input_names,
                      output_names=output_names)

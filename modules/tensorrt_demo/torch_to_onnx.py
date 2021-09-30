import sys
import torch

sys.path.append('..')

from net.model_irse import IR_18


def get_models(model_file):
    input_size = [64, 64]
    checkpoint = torch.load(model_file)
    output_shape = checkpoint['output_shape']
    embedding_size = output_shape[1]
    model = IR_18(input_size, embedding_size)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    return model


if __name__ == '__main__':
    model = get_models('./XMC2-Rec_face_recognition.pth.tar')
    onnx_name = 'XMC2-Rec_face_recognition.onnx'
    dummy_inp = torch.randn(1, 3, 64, 64, device='cpu')
    input_names = ['input']
    output_names = ['output']
    torch.onnx.export(model, dummy_inp, onnx_name, verbose=True, input_names=input_names,
                      output_names=output_names)

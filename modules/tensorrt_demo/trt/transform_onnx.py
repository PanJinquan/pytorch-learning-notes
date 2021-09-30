import os
import tensorrt as trt

# You can set the logger severity higher to suppress messages (or lower to display more messages).
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


def GiB(val):
    return val * 1 << 30


def MiB(val):
    return val * 1 << 20


def build_engine_onnx(model_file, engine_file, batch_size=1, mem=100):
    with trt.Builder(TRT_LOGGER) as builder, builder.create_network() as network, trt.OnnxParser(network,
                                                                                                 TRT_LOGGER) as parser:
        builder.fp16_mode = True
        builder.max_batch_size = batch_size
        builder.max_workspace_size = MiB(mem)
        # Load the Onnx model and parse it in order to populate the TensorRT network.
        print('pre parse')
        with open(model_file, 'rb') as model:
            model = model.read()
            # print('model: {}'.format(model))
            try:
                parser.parse(model)
            except Exception as e:
                print('Exception: {}'.format(e))
            except ZeroDivisionError as e:
                print('ZeroDivisionError: {}'.format(e))
        print('post parse')
        engine = builder.build_cuda_engine(network)
        with open(engine_file, 'wb') as file:
            print('Save engine file: {}'.format(engine_file))
            file.write(engine.serialize())
            print('Save engine file: {} finish'.format(engine_file))


if __name__ == '__main__':
    onnx_dir = './'
    model_conf = {
        # 'XMC2-Det_teacher_detector': {'batch_size': 1, 'mem': 1024},
        # 'XMC2-Cls_global_action_classifier': {'batch_size': 8, 'mem': 1024},
        # 'XMC2-Det_student_detector': {'batch_size': 1, 'mem': 1024},
        # 'XMC2-Cls_body_action_classifier': {'batch_size': 8, 'mem': 1024},
        # 'XMC2-face_emotion_classifier_and_direction_regressor': {'batch_size': 8, 'mem': 1024},
        # 'XMC2-Rec_face_recognition': {'batch_size': 8, 'mem': 256},
        # 'XMC2-landmark-detection': {'batch_size': 8, 'mem': 256}
        'norm_demo': {'batch_size': 8, 'mem': 256}
    }
    for model_name in model_conf.keys():
        onnx_file_path = os.path.join(onnx_dir, model_name + '.onnx')
        engine_file_path = os.path.join(onnx_dir, model_name + '.engine')
        batch_size = model_conf[model_name]['batch_size']
        mem = model_conf[model_name]['mem']
        build_engine_onnx(onnx_file_path, engine_file_path, batch_size, mem)

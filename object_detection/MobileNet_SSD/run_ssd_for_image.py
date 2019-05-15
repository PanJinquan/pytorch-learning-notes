from vision.ssd.vgg_ssd import create_vgg_ssd, create_vgg_ssd_predictor
from vision.ssd.mobilenetv1_ssd import create_mobilenetv1_ssd, create_mobilenetv1_ssd_predictor
from vision.ssd.mobilenetv1_ssd_lite import create_mobilenetv1_ssd_lite, create_mobilenetv1_ssd_lite_predictor
from vision.ssd.squeezenet_ssd_lite import create_squeezenet_ssd_lite, create_squeezenet_ssd_lite_predictor
from vision.ssd.mobilenet_v2_ssd_lite import create_mobilenetv2_ssd_lite, create_mobilenetv2_ssd_lite_predictor
from vision.utils.misc import Timer
import cv2
import sys
from utils import file_processing,image_processing
import torch

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = 'cpu'

def select_net(model_path,net_type,num_classes):
    '''
    选择模型
    :param model_path: 模型路径
    :param net_type: 模型类型
    :param num_classes: label个数，label=0,是背景
    :return:
    '''
    if net_type == 'vgg16-ssd':
        net = create_vgg_ssd(num_classes, is_test=True)
    elif net_type == 'mb1-ssd':
        net = create_mobilenetv1_ssd(num_classes, is_test=True)
    elif net_type == 'mb1-ssd-lite':
        net = create_mobilenetv1_ssd_lite(num_classes, is_test=True)
    elif net_type == 'mb2-ssd-lite':
        net = create_mobilenetv2_ssd_lite(num_classes, is_test=True, device=device)
    elif net_type == 'sq-ssd-lite':
        net = create_squeezenet_ssd_lite(num_classes, is_test=True)
    else:
        print("The net type is wrong. It should be one of vgg16-ssd, mb1-ssd and mb1-ssd-lite.")
        sys.exit(1)
    net.load(model_path)
    if net_type == 'vgg16-ssd':
        predictor = create_vgg_ssd_predictor(net, candidate_size=200)
    elif net_type == 'mb1-ssd':
        predictor = create_mobilenetv1_ssd_predictor(net, candidate_size=200)
    elif net_type == 'mb1-ssd-lite':
        predictor = create_mobilenetv1_ssd_lite_predictor(net, candidate_size=200)
    elif net_type == 'mb2-ssd-lite':
        predictor = create_mobilenetv2_ssd_lite_predictor(net, candidate_size=200,device=device)
    elif net_type == 'sq-ssd-lite':
        predictor = create_squeezenet_ssd_lite_predictor(net, candidate_size=200)
    else:
        predictor = create_vgg_ssd_predictor(net, candidate_size=200)
    return predictor

def predict_image(predictor,rgb_image,prob_threshold):
    '''
    预测图片
    :param predictor: 预测模型的对象
    :param rgb_image: rgb的图像
    :param prob_threshold: 置信度
    :return:
    '''
    boxes, labels, probs = predictor.predict(rgb_image, top_k=10, prob_threshold=prob_threshold)
    boxes=boxes.detach().cpu().numpy()
    labels=labels.detach().cpu().numpy()
    probs=probs.detach().cpu().numpy()
    return boxes, labels, probs

def batch_detect_image(image_list, class_names, prob_threshold, show=False):
    predictor=select_net(model_path, net_type, len(class_names))
    for image_path in image_list:
        orig_image = cv2.imread(image_path)
        rgb_image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
        boxes, labels, probs=predict_image(predictor,rgb_image,prob_threshold)
        label_names=file_processing.decode_label(labels,name_table=class_names)
        info=image_processing.combile_label_prob(label_names,probs)
        print("predict:{}".format(info))
        if show:
            image_processing.show_image_bboxes_text("image",rgb_image,boxes,info)
def batch_test(image_dir, class_names, prob_threshold, show=False):
    image_list=file_processing.get_images_list(image_dir,postfix=["*.bmp"])
    batch_detect_image(image_list, class_names, prob_threshold, show=show)

if __name__=="__main__":
    net_type = 'mb2-ssd-lite'
    model_path = 'models/PCwall135.pth'
    # model_path = 'models/mb2-ssd-lite-my.pth'
    label_path = 'dataset/PCwall_label.txt'
    # image_path = 'E:/git/VOC0712_dataset/my_voc/test/10color.bmp'
    class_names = [name.strip() for name in open(label_path).readlines()]
    # batch_detect_image([image_path], class_names, prob_threshold=0.9, show=True)
    image_dir='E:/git/PCwall/test2'
    batch_test(image_dir, class_names, prob_threshold=0.8                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 , show=True)
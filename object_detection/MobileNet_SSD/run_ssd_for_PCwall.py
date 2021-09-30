from vision.ssd.vgg_ssd import create_vgg_ssd, create_vgg_ssd_predictor
from vision.ssd.mobilenetv1_ssd import create_mobilenetv1_ssd, create_mobilenetv1_ssd_predictor
from vision.ssd.mobilenetv1_ssd_lite import create_mobilenetv1_ssd_lite, create_mobilenetv1_ssd_lite_predictor
from vision.ssd.squeezenet_ssd_lite import create_squeezenet_ssd_lite, create_squeezenet_ssd_lite_predictor
from vision.ssd.mobilenet_v2_ssd_lite import create_mobilenetv2_ssd_lite, create_mobilenetv2_ssd_lite_predictor
from vision.utils.misc import Timer
import cv2
import sys
import torch
'''
python run_ssd_example.py mb2-ssd-lite models/mb2-ssd-lite-my.pth models/voc-model-labels.txt ./dataset/images/1.jpg
mb2-ssd-lite models/mb2-ssd-lite-Epoch-199-Loss-2.8909272387407827.pth models/voc-model-labels.txt ./dataset/images/1.jpg
mb2-ssd-lite models/mb2-ssd-lite-Epoch-199-Loss-2.936172268504188.pth models/voc-model-labels.txt ./dataset/images/1.jpg

'''

# if len(sys.argv) < 5:
#     print('Usage: python run_ssd_example.py <net type>  <model path> <image path>')
#     sys.exit(0)
# net_type = sys.argv[1]
# model_path = sys.argv[2]
# label_path = sys.argv[3]
# image_path = sys.argv[4]
net_type = 'mb2-ssd-lite'
model_path = 'models/PCwall.pth'
# model_path = 'models/mb2-ssd-lite-my.pth'
label_path = 'dataset/my_voc.txt'
image_path = 'E:/git/VOC0712_dataset/my_voc/test/1color.bmp'

class_names = [name.strip() for name in open(label_path).readlines()]
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device='cpu'
if net_type == 'vgg16-ssd':
    net = create_vgg_ssd(len(class_names), is_test=True)
elif net_type == 'mb1-ssd':
    net = create_mobilenetv1_ssd(len(class_names), is_test=True)
elif net_type == 'mb1-ssd-lite':
    net = create_mobilenetv1_ssd_lite(len(class_names), is_test=True)
elif net_type == 'mb2-ssd-lite':
    net = create_mobilenetv2_ssd_lite(len(class_names), is_test=True,device=device)
elif net_type == 'sq-ssd-lite':
    net = create_squeezenet_ssd_lite(len(class_names), is_test=True)
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

orig_image = cv2.imread(image_path)
image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
boxes, labels, probs = predictor.predict(image, top_k=10, prob_threshold=0.4)

for i in range(boxes.size(0)):
    box = boxes[i, :]
    cv2.rectangle(orig_image, (box[0], box[1]), (box[2], box[3]), (255, 255, 0), 4)
    #label = f"""{voc_dataset.class_names[labels[i]]}: {probs[i]:.2f}"""
    label = f"{class_names[labels[i]]}: {probs[i]:.2f}"
    print("label:{}".format(label))
    cv2.putText(orig_image, label,
                (box[0] + 20, box[1] + 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,  # font scale
                (255, 0, 255),
                2)  # line type
path = "run_ssd_example_output.jpg"
# cv2.imwrite(path, orig_image)
cv2.imshow("detect",orig_image)
cv2.waitKey(0)
print(f"Found {len(probs)} objects. The output image is {path}")

def predictor():
    pass
if __name__=="__main__":
    pass
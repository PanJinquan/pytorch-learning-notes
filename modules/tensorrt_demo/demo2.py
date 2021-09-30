# -*-coding: utf-8 -*-
import os, sys

sys.path.append(os.getcwd())
import numpy as np
import torch
import PIL.Image as Image
import demo
import face_alignment.demo as face_alignment
import face_landmark.demo as face_landmark
from net.model_irse import IR_18
from face_alignment.alignment import align_trans

print(torch.__version__)


def read_image(image_path):
    image = np.asarray(Image.open(image_path))
    return image


def crop_face(image, face_bbox):
    face = image[face_bbox[1]:face_bbox[3], face_bbox[0]:face_bbox[2]]
    return face


def show_image(win_name, image):
    Image.fromarray(image).show(win_name)


def save_image(filename, image):
    Image.fromarray(image).save(filename)


if __name__ == "__main__":
    input_size = [64, 64]  # 模型输入大小
    device = "cuda:0"

    # 1.read image
    test_image1 = "./data/test_image/1.jpg"  # 测试图片1
    test_image2 = "./data/test_image/2.jpg"  # 测试图片2
    image1 = read_image(test_image1)
    image2 = read_image(test_image2)

    # 2.detect face
    # 3.get face bbox
    face_bbox1 = [95, 97, 208, 252]
    face_bbox2 = [69, 58, 173, 201]

    # 4.crop face roi
    face1 = crop_face(image1, face_bbox1)
    face2 = crop_face(image2, face_bbox2)
    show_image("face1", face1)
    show_image("face2", face2)

    # 5.face landmark Detection
    # init landmark Detection
    onet_path = "./face_landmark/XMC2-landmark-detection.pth.tar"
    lmdet = face_landmark.ONetLandmarkDet(onet_path, device="cuda:0")
    landmarks1 = lmdet.get_faces_landmarks([np.asarray(face1)])[0]
    landmarks2 = lmdet.get_faces_landmarks([np.asarray(face2)])[0]

    # 6.show face landmarks
    face_landmark.show_landmark("face-landmark1", face1, [landmarks1])
    face_landmark.show_landmark("face-landmark2", face2, [landmarks2])
    print("landmarks1:{}".format(landmarks1))
    print("landmarks2:{}".format(landmarks2))

    # 7.face aligment
    output_size = [112, 112]
    # get reference facial points
    refrence = align_trans.get_reference_facial_points(output_size, default_square=True)
    # face alignment and crop face roi
    alig_face1 = align_trans.warp_and_crop_face(face1, landmarks1, refrence, crop_size=output_size)
    alig_face2 = align_trans.warp_and_crop_face(face2, landmarks2, refrence, crop_size=output_size)
    show_image("alig_face1", alig_face1)
    show_image("alig_face2", alig_face2)
    save_image("test1.png", alig_face1)
    save_image("test2.png", alig_face2)
    # 8. face recognition
    # 加载模型
    score_threshold = 0.60  # 相似人脸分数人脸阈值
    model_file = "XMC2-Rec_face_recognition.pth.tar"
    model = torch.load(model_file)
    arch = model['arch']
    backbone_name = model['backbone_name']
    input_shape = model['input_shape']
    output_shape = model['output_shape']
    state_dict = model["state_dict"]

    data_transform = demo.pre_process(input_size)
    embedding_size = output_shape[1]
    backbone_name = backbone_name
    net = IR_18(input_size, embedding_size)
    net.load_state_dict(state_dict)
    net = net.to(device)
    net.eval()
    with torch.no_grad():
        # 输入图像预处理
        face_tensor1 = data_transform(Image.fromarray(alig_face1))
        face_tensor2 = data_transform(Image.fromarray(alig_face2))
        face_tensor1 = face_tensor1.unsqueeze(0)  # 增加一个维度
        face_tensor2 = face_tensor2.unsqueeze(0)  # 增加一个维度
        # forward
        embeddings1 = net(face_tensor1.to(device))
        embeddings2 = net(face_tensor2.to(device))
        # 特征后处理函数
        embeddings1 = demo.post_process(embeddings1)
        embeddings2 = demo.post_process(embeddings2)
        # 计算两个特征的欧式距离
        dist = demo.compare_embedding(embeddings1.cpu().numpy(), embeddings2.cpu().numpy())
        # 将距离映射为人脸相似性分数
        score = demo.get_scores(dist)
        # 判断是否是同一个人
        same_person = score > score_threshold
    print("embeddings1.shape:{}\nembeddings1:{}".format(embeddings1.shape, embeddings1[0, 0:20]))
    print("embeddings2.shape:{}\nembeddings2:{}".format(embeddings2.shape, embeddings2[0, 0:20]))
    print("distance:{},score:{},same person:{}".format(dist, score, same_person))

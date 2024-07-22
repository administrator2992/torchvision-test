import colorsys
import os
import time

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageDraw, ImageFont
from torchvision.transforms import functional as F
from utils.utils import (cvtColor, read_JSON, show_config)

#--------------------------------------------#
#   使用自己训练好的模型预测需要修改2个参数
#   model_path和classes_path都需要修改！
#   如果出现shape不匹配
#   一定要注意训练时的NUM_CLASSES、
#   model_path和classes_path参数的修改
#--------------------------------------------#
class MODEL(object):
    _defaults = {
        "classes_path"  : 'coco_classes.json',
        "confidence"    : 0.5,
        #-------------------------------#
        "cuda"          : True,
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    #---------------------------------------------------#
    #   初始化faster RCNN
    #---------------------------------------------------#
    def __init__(self, model_name="frcnn", **kwargs):
        self.__dict__.update(self._defaults)
        for name, value in kwargs.items():
            setattr(self, name, value)
            self._defaults[name] = value 
        #---------------------------------------------------#
        #   获得种类和先验框的数量
        #---------------------------------------------------#
        self.class_names = read_JSON(self.classes_path)[0]
        #---------------------------------------------------#
        hsv_tuples = [(x / len(self.class_names), 1., 1.) for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), self.colors))
        self.model_name = model_name
        self.generate()

        show_config(**self._defaults)

    #---------------------------------------------------#
    #   载入模型
    #---------------------------------------------------#
    def generate(self):
        #-------------------------------#
        #   载入模型与权值
        #-------------------------------#
        if self.model_name == 'frcnn':
            from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn as fastrcnn_mbnetv3
            self.net = fastrcnn_mbnetv3(pretrained=True)
        elif self.model_name == 'retinanet':
            from torchvision.models.detection import retinanet_resnet50_fpn_v2 as retinanet_resnet50
            self.net = retinanet_resnet50(pretrained=True)
        elif self.model_name == 'ssdlite':
            from torchvision.models.detection import ssdlite320_mobilenet_v3_large as ssdlite_mbnetv3
            self.net = ssdlite_mbnetv3(pretrained=True)
        else:
            print(f"model {self.model_name} is not suitable for this program")
            raise ValueError
            
        self.net    = self.net.eval()
    
    def save_model(self, PATH):
        model = self.net.state_dict()
        torch.save(model, PATH)
        print(f"{self.model_name} saved in {PATH}")

    def activate_cuda(self):
        self.net = nn.DataParallel(self.net)
        self.net = self.net.cuda()
    
    #---------------------------------------------------#
    #   检测图片
    #---------------------------------------------------#
    def detect_image(self, img_id, image, crop = False, count = False):
        image       = cvtColor(image)
        with torch.no_grad():
            images = F.to_tensor(image).unsqueeze(0)
            if self.cuda:
                images = images.cuda()
            
            results = self.net(images)
            #-------------------------------------------------------------#
            #   如果没有检测出物体，返回原图
            #---------------------------------------------------------#           
            if len(results[0]) <= 0:
                return image
                
            top_label   = np.array(results[0]['labels'].cpu(), dtype = 'int32')
            top_conf    = results[0]['scores'].cpu().numpy()
            top_boxes   = results[0]['boxes'].cpu().numpy()
        #---------------------------------------------------------#
        #   设置字体与边框厚度
        #---------------------------------------------------------#
        font        = ImageFont.truetype(font='simhei.ttf', size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
        if count:
            print("top_label:", top_label)
            num_classes = np.zeros([len(self.class_names)])
            for i in range(len(self.class_names)):
                num = np.sum(top_label == i)
                if num > 0:
                    print(self.class_names[str(i)], " : ", num)
                num_classes[i] = num
            print("classes_nums:", num_classes)
        if crop:
            for i, c in list(enumerate(top_label)):
                top, left, bottom, right = top_boxes[i]
                
                dir_save_path = "img_crop"
                if not os.path.exists(dir_save_path):
                    os.makedirs(dir_save_path)
                crop_image = image.crop([left, top, right, bottom])
                crop_image.save(os.path.join(dir_save_path, "crop_" + str(i) + ".png"), quality=95, subsampling=0)
                print("save crop_" + str(i) + ".png to " + dir_save_path)
        #---------------------------------------------------------#
        #   图像绘制
        #---------------------------------------------------------#
        objs = {f'frame_{img_id}':[]}
        for i, c in list(enumerate(top_label)):
            predicted_class = self.class_names[str(c)]
            box             = top_boxes[i]
            score           = top_conf[i]

            if score > self.confidence:
                left, top, right, bottom = box
    
                top     = max(0, np.floor(top).astype('int32'))
                left    = max(0, np.floor(left).astype('int32'))
                bottom  = min(image.size[1], np.floor(bottom).astype('int32'))
                right   = min(image.size[0], np.floor(right).astype('int32'))

                box = [left, top, right, bottom]

                obj = {'label': predicted_class, 'box': box, 'scores': round(float(score), 2)}
                objs[f'frame_{img_id}'].append(obj)
    
                label = '{} {:.2f}'.format(predicted_class, score)
                draw = ImageDraw.Draw(image)
                label_size = draw.textsize(label, font)
                label = label.encode('utf-8')
                # print(label, top, left, bottom, right)
                
                if top - label_size[1] >= 0:
                    text_origin = np.array([left, top - label_size[1]])
                else:
                    text_origin = np.array([left, top + 1])
    
                draw.rectangle([(left, top), (right, bottom)], outline=self.colors[c], width=2)
                draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)], fill=self.colors[c])
                draw.text(text_origin, str(label,'UTF-8'), fill=(0, 0, 0), font=font)
                del draw

        return image, objs

    def get_FPS(self, image, test_interval):
        #---------------------------------------------------#
        #   计算输入图片的高和宽
        #---------------------------------------------------#
        image_shape = np.array(np.shape(image)[0:2])
        input_shape = get_new_img_size(image_shape[0], image_shape[1])
        #---------------------------------------------------------#
        #   在这里将图像转换成RGB图像，防止灰度图在预测时报错。
        #   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
        #---------------------------------------------------------#
        image       = cvtColor(image)
        
        #---------------------------------------------------------#
        #   给原图像进行resize，resize到短边为600的大小上
        #---------------------------------------------------------#
        image_data  = resize_image(image, [input_shape[1], input_shape[0]])
        #---------------------------------------------------------#
        #   添加上batch_size维度
        #---------------------------------------------------------#
        image_data  = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()

            results = self.net(images)

        t1 = time.time()
        for _ in range(test_interval):
            with torch.no_grad():
                results = self.net(images)
                
        t2 = time.time()
        tact_time = (t2 - t1) / test_interval
        return tact_time

    #---------------------------------------------------#
    #   检测图片
    #---------------------------------------------------#
    def get_map_txt(self, image_id, image, class_names, map_out_path):
        f = open(os.path.join(map_out_path, "detection-results/"+image_id+".txt"),"w")
        #---------------------------------------------------#
        #   计算输入图片的高和宽
        #---------------------------------------------------#
        image_shape = np.array(np.shape(image)[0:2])
        input_shape = get_new_img_size(image_shape[0], image_shape[1])
        #---------------------------------------------------------#
        #   在这里将图像转换成RGB图像，防止灰度图在预测时报错。
        #   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
        #---------------------------------------------------------#
        image       = cvtColor(image)
        
        #---------------------------------------------------------#
        #   给原图像进行resize，resize到短边为600的大小上
        #---------------------------------------------------------#
        image_data  = resize_image(image, [input_shape[1], input_shape[0]])
        #---------------------------------------------------------#
        #   添加上batch_size维度
        #---------------------------------------------------------#
        image_data  = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()

            results = self.net(images)
            #--------------------------------------#
            #   如果没有检测到物体，则返回原图
            #--------------------------------------#
            if len(results[0]) <= 0:
                return 

            top_label   = np.array(results[0][:, 5], dtype = 'int32')
            top_conf    = results[0][:, 4]
            top_boxes   = results[0][:, :4]
        
        for i, c in list(enumerate(top_label)):
            predicted_class = self.class_names[str(c)]
            box             = top_boxes[i]
            score           = str(top_conf[i])

            top, left, bottom, right = box
            if predicted_class not in class_names:
                continue

            f.write("%s %s %s %s %s %s\n" % (predicted_class, score[:6], str(int(left)), str(int(top)), str(int(right)),str(int(bottom))))

        f.close()
        return 

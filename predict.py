#-----------------------------------------------------------------------#
#   predict.py将单张图片预测、摄像头检测、FPS测试和目录遍历检测等功能
#   整合到了一个py文件中，通过指定mode进行模式的修改。
#-----------------------------------------------------------------------#
import time
import os
import subprocess
import torch
import cv2
import threading
import queue
import numpy as np
from jtop import jtop
from PIL import Image
from json import loads as json_load
from model import MODEL
import warnings

warnings.filterwarnings("ignore")

def read_JSON(filename):
    """Reads json file of the given name, parses it and returns the dictionary object"""
    try:
        with open(filename, 'r') as f:
            data=f.read()
        return json_load(data)
    except Exception as e:
        #print(f"readjson error {e}")
        pass

def capture_video(stop_flag, video_path, preprocessed_frame_queue):
    capture = cv2.VideoCapture(video_path)
    while capture.isOpened() and not stop_flag.is_set():
        # Read a frame
        ref, frame = capture.read()
        if not ref:
            break
        gpu_frame = cv2.cuda_GpuMat()
        # Convert format (BGR to RGB) on GPU
        gpu_frame.upload(frame)
        gray_gpu = cv2.cuda.cvtColor(gpu_frame, cv2.COLOR_BGR2RGB)
        preprocessed_frame_queue.put(gray_gpu.download())
    stop_flag.set()
    capture.release()

def inference(stop_flag, preprocessed_frame_queue, detections_queue, fps_queue, network):
    while not stop_flag.is_set():
        frame = preprocessed_frame_queue.get()
        prev_time = time.time()
        frame = np.array(network.detect_image(0, Image.fromarray(np.uint8(frame)))[0])
        fps = 1 / (time.time() - prev_time)
        detections_queue.put(frame)
        fps_queue.put(float(fps))

def drawing(pid, stop_flag, queues, start_time, duration, video_save_path, out, fontsize):
    with open("out/od_perf.txt", mode='a') as file:
        detections_queue, fps_queue = queues
        # Convert format (RGB to BGR) on GPU
        fps = 1
        while not stop_flag.is_set():
            gpu_frame = cv2.cuda_GpuMat()
            gpu_frame.upload(detections_queue.get())
            gray_gpu = cv2.cuda.cvtColor(gpu_frame, cv2.COLOR_RGB2BGR)
            frame = gray_gpu.download()
            hw = read_JSON(f'perf_{pid}.json')
            fps = fps_queue.get()
            if hw:
                text = "FPS %.2f, %s, CPU %.2f%%, MEM/SWAP %.2f/%.2fMB, POWER %.2fmW, GPU %.2f%%" % (fps, hw["J_MODE"], hw["CPU"], hw["MEM"], hw["SWAP"], hw["POWER"], hw["GPU"])
                frame = cv2.putText(frame, text, (3, 40), cv2.FONT_HERSHEY_SIMPLEX, fontsize, (0, 255, 0), 2)
                file.write(text + "\n")
            if video_save_path != "":
                out.write(frame)
            c = cv2.waitKey(int(fps)) 
            if c == 27 or time.time() - start_time > duration:
                break
        # Cleanup and closing
        stop_flag.set()
        if video_save_path != "":
            print("Save processed video to the path: " + video_save_path)
            out.release()
        cv2.destroyAllWindows()
        timeout = 1 / (fps if fps > 0 else 0.5)
        for q in (detections_queue, fps_queue):
            try:
                q.get(block=True, timeout=timeout)
            except queue.Empty:
                pass

if __name__ == "__main__":
    with jtop() as jetson:
        if jetson.ok():
            jetson.memory.clear_cache()
    #----------------------------------------------------------------------------------------------------------#
    #   mode用于指定测试的模式：
    #   'predict'           表示单张图片预测，如果想对预测过程进行修改，如保存图片，截取对象等，可以先看下方详细的注释
    #   'video'             表示视频检测，可调用摄像头或者视频进行检测，详情查看下方注释。
    #   'fps'               表示测试fps，使用的图片是img里面的street.jpg，详情查看下方注释。
    #   'dir_predict'       表示遍历文件夹进行检测并保存。默认遍历img文件夹，保存img_out文件夹，详情查看下方注释。
    #   'heatmap'           表示进行预测结果的热力图可视化，详情查看下方注释。
    #   'export_onnx'       表示将模型导出为onnx，需要pytorch1.7.1以上。
    #----------------------------------------------------------------------------------------------------------#
    mode = "video"
    model_type = "torch" # torch
    model_name = "retinanet" # frcnn, retinanet, sddlite
    #-------------------------------------------------------------------------#
    #   crop                指定了是否在单张图片预测后对目标进行截取
    #   count               指定了是否进行目标的计数
    #   crop、count仅在mode='predict'时有效
    #-------------------------------------------------------------------------#
    crop            = False
    count           = False
    #----------------------------------------------------------------------------------------------------------#
    #  Used to specify the path of the video. When video_path=0, it means detecting the camera.
    # If you want to detect videos, set video_path = "xxx.mp4", which means reading the xxx.mp4 file in the root directory.
    # video_save_path represents the path to save the video. When video_save_path="", it means not to save it.
    # If you want to save the video, set it like video_save_path = "yyy.mp4", which means saving it as a yyy.mp4 file in the root directory.
    # video_fps fps used for saved video
    #
    #   video_path, video_save_path and video_fps are only valid when mode='video'
    # When saving the video, you need to ctrl+c to exit or run to the last frame to complete the complete saving step.
    #----------------------------------------------------------------------------------------------------------#
    video_path      = "img/fullhd.mp4"
    video_save_path = "out/output.mp4"
    #----------------------------------------------------------------------------------------------------------#
    #   test_interval       用于指定测量fps的时候，图片检测的次数。理论上test_interval越大，fps越准确。
    #   fps_image_path      用于指定测试的fps图片
    #   
    #   test_interval和fps_image_path仅在mode='fps'有效
    #----------------------------------------------------------------------------------------------------------#
    test_interval   = 100
    fps_image_path  = "img/street.jpg"
    #-------------------------------------------------------------------------#
    #   dir_origin_path     指定了用于检测的图片的文件夹路径
    #   dir_save_path       指定了检测完图片的保存路径
    #   
    #   dir_origin_path和dir_save_path仅在mode='dir_predict'时有效
    #-------------------------------------------------------------------------#
    dir_origin_path = "img/"
    dir_save_path   = "img_out/"
    #-------------------------------------------------------------------------#
    #   heatmap_save_path   热力图的保存路径，默认保存在model_data下
    #   
    #   heatmap_save_path仅在mode='heatmap'有效
    #-------------------------------------------------------------------------#
    heatmap_save_path = "model_data/heatmap_vision.png"
    #-------------------------------------------------------------------------#
    #   simplify            使用Simplify onnx
    #   onnx_save_path      指定了onnx的保存路径
    #-------------------------------------------------------------------------#
    simplify        = False
    onnx_save_path  = "../DL_model/fasterrcnn.onnx"

    pid = os.getpid()
    print("PID :", pid)

    if model_type == "torch":
        model = MODEL(model_name=model_name)
        model.activate_cuda()
    else:
        print(mode, "or", model_type, "not suitable")
        raise ValueError

    if mode == "predict":
        '''
        1、如果想要进行检测完的图片的保存，利用r_image.save("img.jpg")即可保存，直接在predict.py里进行修改即可。 
        2、如果想要获得预测框的坐标，可以进入yolo.detect_image函数，在绘图部分读取top，left，bottom，right这四个值。
        3、如果想要利用预测框截取下目标，可以进入yolo.detect_image函数，在绘图部分利用获取到的top，left，bottom，right这四个值
        在原图上利用矩阵的方式进行截取。
        4、如果想要在预测图上写额外的字，比如检测到的特定目标的数量，可以进入yolo.detect_image函数，在绘图部分对predicted_class进行判断，
        比如判断if predicted_class == 'car': 即可判断当前目标是否为车，然后记录数量即可。利用draw.text即可写字。
        '''
        while True:
            img = input('Input image filename:')
            try:
                image = Image.open(img)
            except:
                print('Open Error! Try again!')
                continue
            else:
                r_image = model.detect_image(image, crop = crop, count=count)
                r_image.show()

    elif mode == "video":
        ExecUnit = threading.Thread
        Queue = queue.Queue
        stop_flag = threading.Event()

        preprocessed_frame_queue = Queue(maxsize=1)
        detections_queue = Queue(maxsize=1)
        fps_queue = Queue(maxsize=1)

        # Open the performance log file
        with open("out/od_perf.txt", mode='w') as file:
            capture = cv2.VideoCapture(video_path)
            if video_save_path != "":
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                size = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
                video_fps = capture.get(cv2.CAP_PROP_FPS)
                out = cv2.VideoWriter(video_save_path, fourcc, video_fps, size)

            ref, _ = capture.read()
            if not ref:
                raise ValueError("The camera (video) cannot be read correctly. Please check if the video path is correct.")
            w_res = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            if w_res == 1920:
                fontsize = 1
            elif w_res == 1280:
                fontsize = 0.75
            elif w_res == 854:
                fontsize = 0.5
            else:
                fontsize = 0.66

            capture.release()
            del capture

            duration = 600  # Duration of 10 minutes
            
            start_time = time.time()  # Record the start time
            exec_units = (
                ExecUnit(target=capture_video, args=(stop_flag, video_path, preprocessed_frame_queue)),
                ExecUnit(target=inference, args=(stop_flag, preprocessed_frame_queue, detections_queue, fps_queue, model)),
                ExecUnit(target=drawing, args=(pid, stop_flag, (detections_queue, fps_queue), start_time, duration, video_save_path, out, fontsize)),
            )
            for exec_unit in exec_units:
                exec_unit.start()
            for exec_unit in exec_units:
                exec_unit.join()

            print("Video Detection Done!")
        
    elif mode == "fps":
        img = Image.open(fps_image_path)
        tact_time = frcnn.get_FPS(img, test_interval)
        print(str(tact_time) + ' seconds, ' + str(1/tact_time) + 'FPS, @batch_size 1')

    elif mode == "dir_predict":
        import os

        from tqdm import tqdm

        img_names = os.listdir(dir_origin_path)
        for img_name in tqdm(img_names):
            if img_name.lower().endswith(('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')):
                image_path  = os.path.join(dir_origin_path, img_name)
                image       = Image.open(image_path)
                r_image     = frcnn.detect_image(image)
                if not os.path.exists(dir_save_path):
                    os.makedirs(dir_save_path)
                r_image.save(os.path.join(dir_save_path, img_name.replace(".jpg", ".png")), quality=95, subsampling=0)

    elif mode == "heatmap":
        while True:
            img = input('Input image filename:')
            try:
                image = Image.open(img)
            except:
                print('Open Error! Try again!')
                continue
            else:
                model.detect_heatmap(image, heatmap_save_path)
                
    elif mode == "export_onnx":
        model.convert_to_onnx(simplify, onnx_save_path)

    else:
        raise AssertionError("Please specify the correct mode: 'predict', 'video', 'fps', 'heatmap', 'export_onnx', 'dir_predict'.")

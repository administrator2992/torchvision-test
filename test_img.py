import cv2
import os
import time
import numpy as np
import torch
import warnings
import subprocess
from PIL import Image
from signal import SIGINT as signal_SIGINT
from model import MODEL
from json import load as json_load
from json import dumps as json_dump
import multiprocessing as mp
from collections import defaultdict
warnings.filterwarnings('ignore')

def calculate_iou(box1, box2):
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2

    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)

    if inter_x_min < inter_x_max and inter_y_min < inter_y_max:
        inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
    else:
        inter_area = 0

    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)

    union_area = box1_area + box2_area - inter_area

    iou = inter_area / union_area
    return iou

def compute_precision_recall(predictions, ground_truths, iou_threshold=0.5):
    TP = 0
    FP = 0
    FN = 0

    for gt in ground_truths:
        matched = False
        for pred in predictions:
            if pred['label'] == gt['label']:
                iou = calculate_iou(pred['box'], gt['box'])
                if iou >= iou_threshold:
                    TP += 1
                    matched = True
                    predictions.remove(pred)
                    break
        if not matched:
            FN += 1

    FP = len(predictions)
    precision = TP / (TP + FP) if TP + FP > 0 else 0
    recall = TP / (TP + FN) if TP + FN > 0 else 0

    return precision, recall

def calculate_map(predictions, ground_truths, iou_threshold=0.5):
    ap_per_class = {}
    unique_classes = set([pred['label'] for frame in predictions for frame_key, frame_list in frame.items() for pred in frame_list] + 
                         [gt['label'] for gt in ground_truths['frame_1']])

    for cls in unique_classes:
        cls_predictions = []
        for frame in predictions:
            if cls in [pred['label'] for pred in frame.get('frame_1', [])]:
                cls_predictions.extend([{'box': pred['box'], 'score': pred['scores'], 'label': pred['label']} for pred in frame['frame_1'] if pred['label'] == cls])

        cls_ground_truths = [gt for gt in ground_truths['frame_1'] if gt['label'] == cls]

        cls_predictions = sorted(cls_predictions, key=lambda x: x['score'], reverse=True)
        precisions = []
        recalls = []

        for i in range(len(cls_predictions)):
            p = cls_predictions[:i+1]
            precision, recall = compute_precision_recall(p, cls_ground_truths, iou_threshold)
            precisions.append(precision)
            recalls.append(recall)

        precisions = [0] + precisions + [0]
        recalls = [0] + recalls + [1]

        for i in range(len(precisions) - 1, 0, -1):
            precisions[i-1] = max(precisions[i-1], precisions[i])

        ap = 0
        for i in range(1, len(precisions)):
            ap += (recalls[i] - recalls[i-1]) * precisions[i]

        ap_per_class[cls] = ap

    mAP = sum(ap_per_class.values()) / len(ap_per_class)
    return mAP

class BufferedFileWriter:
    def __init__(self, file_path):
        self.file_path = file_path
        self.buffer = []

    def write(self, text):
        self.buffer.append(text)

    def flush(self):
        with open(self.file_path, mode='a') as file:
            file.writelines(self.buffer)

def read_TXT(filename):
    """Reads txt file of the given name, parses it and returns the dictionary object"""
    try:
        with open(filename, 'r') as f:
            data=f.read()
        return data
    except Exception as e:
        print("Error occured in read_TXT", e)

def read_JSON(filename):
    """Reads json file of the given name, parses it and returns the dictionary object"""
    try:
        with open(filename, 'r') as f:
            data=json_load(f)
        return data
    except Exception as e:
        print("Error occured in read_JSON", e)

def write_JSON(dictionary, filename):
    with open(filename, 'w') as f:
        f.write(to_JSON(dictionary))

def to_JSON(dictionary):
    return json_dump(dictionary, indent=4)

def execute_sys_command(args, stdout=None):
    """Executes the given system-level command"""
    return subprocess.Popen(args, stdout=stdout, shell=False)

def load_model(model_name):
    model = MODEL(model_name=model_name)
    return model

def generatepid():
    with open('pid.txt', 'w') as pid:
        pid.write(str(os.getpid()))

def detection(cl, round_id, img_path, model, perf_txt, perf):
    img = cv2.imread(img_path)
    model.activate_cuda()
    frame_id = -1
    count_id = 0
    if perf:
        generatepid()
        list_objs = []
        hw_dict = {'CPU':[], 'sys':[], 'Mem':[], 'Power':[], 'GPU':[], 'GPU Mem':[], 'IOREAD':[], 'IOWRITE':[]}
    while frame_id < 30:
        gpu_frame = cv2.cuda_GpuMat()
        if count_id == 0:
            f_time = time.time()
        if frame_id == 0:
            prev_time = time.time()
        count_id = 1
        gpu_frame.upload(img)
        gpu_gray = cv2.cuda.cvtColor(gpu_frame, cv2.COLOR_BGR2RGB)
        frame = gpu_gray.download()
        raw_frame, objs = model.detect_image(frame_id, Image.fromarray(np.uint8(frame)))
        frame = np.array(raw_frame)
        if perf:
            if round(time.time() - f_time) >= 2 and frame_id > -1:
                try:
                    pmode, cpu, sys, mem, power, gpu, gpumem, read, write = read_TXT('out/perf.txt').split(',')
                except:
                    continue
                count_id = 0
            if frame_id == -1:
                print("The warm-up has been successful")
            else:
                try:
                    hw_dict['CPU'].append(float(cpu)); hw_dict['sys'].append(float(sys)); hw_dict['Mem'].append(float(mem)); hw_dict['Power'].append(float(power)); hw_dict['GPU'].append(float(gpu)); hw_dict['GPU Mem'].append(float(gpumem)); hw_dict['IOREAD'].append(float(read)); hw_dict['IOWRITE'].append(float(write))
                    list_objs.append(objs)
                except Exception as e:
                    print(f"Error occured at perf output: {e}")
        frame_id += 1
    dtime = time.time() - prev_time
    if perf:
        if dtime < 2:
            time.sleep(2)
            pmode, cpu, sys, mem, power, gpu, gpumem, read, write = read_TXT('out/perf.txt').split(',')
        gt = read_JSON('img/annotations.json')
        map_value = calculate_map(list_objs, gt)
        ioread_mb = [x * 512 / 1000000 for x in hw_dict['IOREAD']]
        iowrite_mb = [x * 512 / 1000000 for x in hw_dict['IOWRITE']]

        hw = "%s, CPU[user] %.2f-%.2f/%.2f/%.2f/%.2f%%, CPU[system] %.2f-%.2f/%.2f/%.2f/%.2f%%, MEM %.2f-%.2f/%.2f/%.2f/%.2fMB, POWER %.2f-%.2f/%.2f/%.2f/%.2fmW, GPU %.2f-%.2f/%.2f/%.2f/%.2f%%, GPUmem %.2f-%.2f/%.2f/%.2f/%.2fMB, IOREAD %.2f-%.2f/%.2f/%.2f/%.2fMB/s, IOWRITE %.2f-%.2f/%.2f/%.2f/%.2fMB/s, mAP %.2f%%\n" % (
            pmode,
            min(hw_dict['CPU']), max(hw_dict['CPU']), np.average(hw_dict['CPU']), np.std(hw_dict['CPU']), np.median(hw_dict['CPU']),
            min(hw_dict['sys']), max(hw_dict['sys']), np.average(hw_dict['sys']), np.std(hw_dict['sys']), np.median(hw_dict['sys']),
            min(hw_dict['Mem']), max(hw_dict['Mem']), np.average(hw_dict['Mem']), np.std(hw_dict['Mem']), np.median(hw_dict['Mem']),
            min(hw_dict['Power']), max(hw_dict['Power']), np.average(hw_dict['Power']), np.std(hw_dict['Power']), np.median(hw_dict['Power']),
            min(hw_dict['GPU']), max(hw_dict['GPU']), np.average(hw_dict['GPU']), np.std(hw_dict['GPU']), np.median(hw_dict['GPU']),
            min(hw_dict['GPU Mem']), max(hw_dict['GPU Mem']), np.average(hw_dict['GPU Mem']), np.std(hw_dict['GPU Mem']), np.median(hw_dict['GPU Mem']),
            min(ioread_mb), max(ioread_mb), np.average(ioread_mb), np.std(ioread_mb), np.median(ioread_mb),
            min(iowrite_mb), max(iowrite_mb), np.average(iowrite_mb), np.std(iowrite_mb), np.median(iowrite_mb),
            map_value*100
        )
        text = "Round-%s CL-%s: Detection time for %s frames: %.2f sec, Throughput %.2f, %s" % (round_id, cl, frame_id, dtime, (30/dtime)*cl, hw)
        perf_txt.write(text)
        
        perf_txt.flush()

def main(round_id, perf):
    try:
        if perf:
            generatepid()
            inspector_process = execute_sys_command(['python3', 'inspector.py'])
            while not os.path.exists('out/perf.txt'):
                continue
            time.sleep(2)
        
            saved_perf = "out/od_frcnn_coco_3_4.txt"
            if round_id == 1:
                if os.path.exists(saved_perf):
                    os.remove(saved_perf)
                with open(saved_perf, 'w') as p:
                    pass

            perf_txt = BufferedFileWriter(saved_perf)
            hw = read_TXT('out/perf.txt')
            text = "Before Load Model: %s\n" % (hw)
            perf_txt.write(text)
        else:
            perf_txt = None
        prev_time = time.time()
        model = load_model("frcnn") # frcnn, retinanet, sddlite
        elap_time = time.time() - prev_time
        if perf:
            # model load hw overhead
            time.sleep(2)
            hw = read_TXT('out/perf.txt')
            text = "Loaded model in %.2f sec, %s\n" % (elap_time, hw)
            perf_txt.write(text)
            os.remove('pid.txt')
        # initial pre-processing OD
        mp.set_start_method('spawn')
        num_processes = 3  # Number of processes
        img_path = 'img/frame_1.jpg'  # Path to the video file
    
        t1 = time.perf_counter()
        # Start video processing segments
        processes = []
        for _ in range(num_processes):
            p = mp.Process(target=detection, args=(num_processes, round_id, img_path, model, perf_txt, perf))
            p.start()
            processes.append(p)
    
        print("Waiting...")
        # Cleanup
        for p in processes:
            p.join()
        
        if perf:
            # Terminate inspector process
            inspector_process.send_signal(signal_SIGINT)
        
            if os.path.exists('out/perf.txt'):
                os.remove('out/perf.txt')
                os.remove('pid.txt')
        print(f"Round {round_id}: Time Elapsed {round(time.perf_counter()-t1, 2)} sec")
    except Exception as e:
        print("Error occured in main", e)
        if perf:
            # Terminate inspector process
            inspector_process.send_signal(signal_SIGINT)
        
            if os.path.exists('out/perf.txt'):
                os.remove('out/perf.txt')
                os.remove('pid.txt')

if __name__ == '__main__':
    import sys
    main(int(sys.argv[1]), bool(int(sys.argv[2])))

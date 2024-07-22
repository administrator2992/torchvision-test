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
from json import loads as json_load
from json import dumps as json_dump
import multiprocessing as mp

warnings.filterwarnings('ignore')

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
    """Reads json file of the given name, parses it and returns the dictionary object"""
    try:
        with open(filename, 'r') as f:
            data=f.read()
        return data
    except Exception as e:
        print("Error occured in read_TXT", e)

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

def video_process_segment(process_id, round_id, video_path, model, perf_txt):
    cap = cv2.VideoCapture(video_path)
    model.activate_cuda()
    
    frame_id = 0
    count_id = 0
    list_objs = []
    hw_dict = {'CPU':[], 'sys':[], 'Mem':[], 'Power':[], 'GPU':[], 'IOREAD':[], 'IOWRITE':[]}
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        gpu_frame = cv2.cuda_GpuMat()
        if frame_id == 0:
            prev_time = time.time()
        if count_id == 0:
            f_time = time.time()
        frame_id += 1
        count_id += 1
        gpu_frame.upload(frame)
        gpu_gray = cv2.cuda.cvtColor(gpu_frame, cv2.COLOR_BGR2RGB)
        frame = gpu_gray.download()
        raw_frame, objs = model.detect_image(frame_id, Image.fromarray(np.uint8(frame)))
        frame = np.array(raw_frame)
        if round(time.time() - f_time) >= 2:
            try:
                pmode, cpu, sys, mem, power, gpu, read, write = read_TXT('out/perf.txt').split(',')
            except:
                continue
            count_id = 0
        hw_dict['CPU'].append(float(cpu)); hw_dict['sys'].append(float(sys)); hw_dict['Mem'].append(float(mem)); hw_dict['Power'].append(float(power)); hw_dict['GPU'].append(float(gpu)); hw_dict['IOREAD'].append(float(read)); hw_dict['IOWRITE'].append(float(write))
        if frame_id == cap.get(cv2.CAP_PROP_FPS):
            dtime = time.time() - prev_time
            hw = "%s, CPU[user] %.2f-%.2f%%, CPU[system] %.2f-%.2f%%, MEM %.2f-%.2fMB, POWER %.2f-%.2fmW, GPU %.2f-%.2f%%, IOREAD %.2f-%.2fbread/s, IOWRITE %.2f-%.2fbwrtn/s\n" % (pmode, min(hw_dict['CPU']), max(hw_dict['CPU']), min(hw_dict['sys']), max(hw_dict['sys']), min(hw_dict['Mem']), max(hw_dict['Mem']), min(hw_dict['Power']), max(hw_dict['Power']), min(hw_dict['GPU']), max(hw_dict['GPU']), min(hw_dict['IOREAD']), max(hw_dict['IOREAD']), min(hw_dict['IOWRITE']), max(hw_dict['IOWRITE']))
            # Detection time elapsed per-second of video
            text = "Round-%s Process-%s: Detection time for %s frames: %.2f sec, %s" % (round_id, process_id, frame_id, dtime, hw)
            perf_txt.write(text)
            frame_id = 0
            hw_dict = {'CPU':[], 'sys':[], 'Mem':[], 'Power':[], 'GPU':[], 'IOREAD':[], 'IOWRITE':[]}
        # fps = 1 / (time.time() - prev_time)
        # if fps < 1 and frame_id == 2:
        #     print("Forbidden FPS, under 1 FPS")
        #     break
        # if hw and round(time.time() - txt_loop) >= 3:
        #     text = "FPS %.2f, %s" % (fps, hw)
        #     perf_txt.write(text)
        #     txt_loop = time.time()
        # list_objs.append(objs)
    
    perf_txt.flush()
    #write_JSON(list_objs, 'out/objs_fhd_frcnn.json')
    cap.release()

def main(round_id):
    try:
        inspector_process = execute_sys_command(['python3', 'inspector.py'])
        while not os.path.exists('out/perf.txt'):
            continue
        time.sleep(2)
        
        saved_perf = "out/od_frcnn_fhd_1_8.txt"
        if round_id == 1:
            if os.path.exists(saved_perf):
                os.remove(saved_perf)
            with open(saved_perf, 'w') as p:
                pass

        perf_txt = BufferedFileWriter(saved_perf)
        hw = read_TXT('out/perf.txt')
        text = "Before Load Model: %s\n" % (hw)
        perf_txt.write(text)
        prev_time = time.time()
        model = load_model("frcnn") # frcnn, retinanet, sddlite
        elap_time = time.time() - prev_time
        # model load hw overhead
        time.sleep(2)
        hw = read_TXT('out/perf.txt')
        text = "Loaded model in %.2f sec, %s\n" % (elap_time, hw)
        perf_txt.write(text)
    
        # initial pre-processing OD
        mp.set_start_method('spawn')
        num_processes = 1  # Number of processes
        video_path = 'img/fullhd.mp4'  # Path to the video file
    
        t1 = time.perf_counter()
        # Start video processing segments
        processes = []
        for i in range(num_processes):
            p = mp.Process(target=video_process_segment, args=(i, round_id, video_path, model, perf_txt))
            p.start()
            processes.append(p)
    
        print("Waiting...")
        # Cleanup
        for p in processes:
            p.join()
        
        # Terminate inspector process
        inspector_process.send_signal(signal_SIGINT)
    
        if os.path.exists('out/perf.txt'):
            os.remove('out/perf.txt')
        
        print(f"Round {round_id}: Time Elapsed {round(time.perf_counter()-t1, 2)} sec")
    except Exception as e:
        print("Error occured in main", e)
        # Terminate inspector process
        inspector_process.send_signal(signal_SIGINT)
    
        if os.path.exists('out/perf.txt'):
            os.remove('out/perf.txt')

if __name__ == '__main__':
    import sys
    main(sys.argv[1])

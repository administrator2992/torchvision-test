import subprocess
import os
import time
import psutil
import warnings
import traceback
from platform import uname
import re
from signal import SIGINT as signal_SIGINT
from sar_parser import SARParser

warnings.filterwarnings('ignore')

def execute_sys_command(args, stdout=None):
    """Executes the given system-level command"""
    return subprocess.Popen(args, stdout=stdout, shell=False)

class Inspector:    
    def parse_SAR_log(self, file):
        #  we do not care about the output file given; we need the binary file for parsing using sadf
        parser = SARParser(file)

        cpu_stats, sys = parser.parse_cpu_stats()
        memory_stats = parser.parse_memory_stats()
        IO_reads_stats, IO_writes_stats = parser.parse_IO_stats()

        if cpu_stats:
            return cpu_stats, sys, memory_stats/1024, IO_reads_stats, IO_writes_stats
        else:
            return 0, 0, 0, 0, 0

    def cpu(self):
        execute_sys_command(["sar", "-u", "-r", "-b", "1", "1", "-o", "sar_bin"], subprocess.DEVNULL)

    def jstat_start(self, passwd="CloudLab12#$%"):
        subprocess.run(["sudo", "-S", "tegrastats", "--start", "--logfile", "tegrastats.txt"], input=passwd, universal_newlines=True)
    
    def jstat_stop(self, dev_type="gpu", passwd="CloudLab12#$%"):
        subprocess.run(["sudo", "-S", "tegrastats", "--stop"], input=passwd, universal_newlines=True)
        try:
            file_path = "tegrastats.txt"
            if os.path.exists(file_path):
                out = open(file_path, 'r')
            else:
                raise FileNotFoundError
            lines = out.read().split('\n')

            entire_gpu = []
            entire_power = []
            entire_power_gpu = []
            entire_power_cpu = []
            entire_freq_gpu = []
            for line in lines:
                if '4.9.337-tegra' == uname().release:
                    pattern_pow = r"POM_5V_IN (\d+)/(\d+)"
                    match_pow = re.search(pattern_pow, line)
                    if match_pow:
                        power_ = match_pow.group(2)
                        entire_power.append(float(power_))
                    pattern_pow_cpu = r"POM_5V_CPU (\d+)/(\d+)"
                    match_pow_cpu = re.search(pattern_pow_cpu, line)
                    if match_pow_cpu:
                        power_cpu_ = match_pow_cpu.group(2)
                        entire_power_cpu.append(float(power_cpu_))
                elif '5.10.104-tegra' == uname().release:
                    pattern_pow = r"VDD_IN (\d+)mW/(\d+)mW"
                    match_pow = re.search(pattern_pow, line)
                    if match_pow:
                        power_ = match_pow.group(2)
                        entire_power.append(float(power_))
                    pattern_pow_cpu = r"VDD_SOC (\d+)mW/(\d+)mW"
                    match_pow_cpu = re.search(pattern_pow_cpu, line)
                    if match_pow_cpu:
                        power_cpu_ = match_pow_cpu.group(2)
                        entire_power_cpu.append(float(power_cpu_))
                if dev_type == 'gpu':
                    pattern_gpu = r"GR3D_FREQ (\d+)%@(\d+)"
                    match_gpu = re.search(pattern_gpu, line)
                    if match_gpu:
                        gpu_ = match_gpu.group(1)
                        freq = match_gpu.group(2)
                        entire_freq_gpu.append(float(freq))
                        entire_gpu.append(float(gpu_))
                    if '4.9.337-tegra' == uname().release:
                        pattern_pow_gpu = r"POM_5V_GPU (\d+)/(\d+)"
                        match_pow_gpu = re.search(pattern_pow_gpu, line)
                        if match_pow_gpu:
                            power_gpu_ = match_pow_gpu.group(2)
                            entire_power_gpu.append(float(power_gpu_))
                    elif '5.10.104-tegra' == uname().release:
                        pattern_pow_gpu = r"VDD_CPU_GPU_CV (\d+)mW/(\d+)mW"
                        match_pow_gpu = re.search(pattern_pow_gpu, line)
                        if match_pow_gpu:
                            power_gpu_ = match_pow_gpu.group(2)
                            entire_power_gpu.append(float(power_gpu_))
            entire_gpu_ = [num for num in entire_gpu if num > 20.0]
            entire_power_ = [num for num in entire_power if num > 0.0]
            entire_power_gpu_ = [num for num in entire_power_gpu if num > 0.0]
            entire_power_cpu_ = [num for num in entire_power_cpu if num > 0.0]
            result_freq = sum(entire_freq_gpu) / len(entire_freq_gpu) if entire_freq_gpu else 0
            result_gpu = sum(entire_gpu_) / len(entire_gpu_) if entire_gpu_ else 0
            result_power = sum(entire_power_) / len(entire_power_) if entire_power_ else 0
            result_power_gpu = sum(entire_power_gpu_) / len(entire_power_gpu_) if entire_power_gpu_ else 0
            result_power_cpu = sum(entire_power_cpu_) / len(entire_power_cpu_) if entire_power_cpu_ else 0

            return result_gpu, result_power#, result_freq, result_power_cpu, result_power_gpu

        except Exception as e:
            print("tegrastats error :", e)
            result_gpu = 0
            result_power = 0
            result_power_cpu = 0
            result_power_gpu = 0
            result_freq = 0

            return result_gpu, result_power#, result_freq, result_power_cpu, result_power_gpu

    def GPUmem(self, pid):
        pattern = re.compile(rf"{pid}\s+(\d+)K")
        with open("/sys/kernel/debug/nvmap/iovmm/maps", "r") as fp:
            content = fp.read()
            match = pattern.search(content)
            if match:
                res = float(match.group(1))/1024
                return res
            else:
                return 0
    
    def getpmode(self):
        out = subprocess.check_output("sudo nvpmodel -q", shell=True)
        pattern = r"NV Power Mode: (\w+)\n(\d+)"
        match = re.search(pattern, out.decode('utf-8'))
        if match:
            power_mode = match.group(1)

            return power_mode

    def getperf(self, pid):
        self.cpu()
        self.jstat_start()
        time.sleep(2)
        cpu_res, sys, mem, ioread, iowrite = self.parse_SAR_log("sar_bin")
        gpu, power = self.jstat_stop()
        gpumem = self.GPUmem(pid)
        pmode = self.getpmode()
        results = f"{pmode}, {round(cpu_res * float(os.cpu_count()), 2)}, {round(sys * float(os.cpu_count()), 2)}, {mem}, {power}, {gpu}, {gpumem}, {ioread}, {iowrite}"

        return results

def write(txt, filename):
    with open(filename, 'w') as f:
        f.write(txt)

def delete_files(files_name):
    for file in files_name:
        if os.path.exists(file):
            os.remove(file)

def getpid():
    if os.path.exists('pid.txt'):
        with open('pid.txt') as pid:
            return pid.read()

if __name__ == "__main__":
    inspector = Inspector()
    try:
        while True:
            try:
                if getpid():
                    write(inspector.getperf(getpid()), 'out/perf.txt')
                    delete_files(["sar_bin", "sar_bin.xml", "tegrastats.txt"])

            except Exception as e:
                print("Error occured in inspector,", e)
                subprocess.run(["sudo", "-S", "tegrastats", "--stop"], input="CloudLab12#$%", universal_newlines=True)
                delete_files(["sar_bin", "sar_bin.xml", "tegrastats.txt"])
                break
    
    except KeyboardInterrupt:
        subprocess.run(["sudo", "-S", "tegrastats", "--stop"], input="CloudLab12#$%", universal_newlines=True)
        delete_files(["sar_bin", "sar_bin.xml", "tegrastats.txt"])


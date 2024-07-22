import xml.etree.ElementTree as ET

def execute_sys_command(args, stdout=None):
    import subprocess
    """Executes the given system-level command"""
    return subprocess.Popen(args, stdout=stdout, shell=False)

'''
    Class parses a sar output using 'sadf' XML output file to return cpu, memory and IO metrics 
'''
class SARParser:

    CMD_SAR_BIN_TO_XML = ["sadf", "-x", "", "--", "-u", "-r", "-b"]

    def __init__(self, file):
        self.tree = None
        self.ns = { "ns" : "http://pagesperso-orange.fr/sebastien.godard/sysstat" } # TODO: make ns dynamic

        self.convert_to_XML(file)

        self.cpu_stats = None
        self.sys_stats = None
        self.memory_stats = None
        self.reads_stats = None
        self.writes_stats = None

    def convert_to_XML(self, file):
        SARParser.CMD_SAR_BIN_TO_XML[2] = file

        output_file = str(file) + ".xml"

        proc = None
        with open(output_file, "w") as f:
            proc = execute_sys_command(SARParser.CMD_SAR_BIN_TO_XML, f)

        while proc.poll() is None:
            proc.wait()

        self.tree = ET.parse(output_file)


    def parse_cpu_stats(self):

        root = self.tree.getroot()

        cpu_stat_elements = root.findall("ns:host/ns:statistics//ns:timestamp//ns:cpu-load//ns:cpu[@user]", self.ns)
        system_stat = root.findall("ns:host/ns:statistics//ns:timestamp//ns:cpu-load//ns:cpu[@system]", self.ns)

        stats = []
        syss = []
        for cpu, system in zip(cpu_stat_elements, system_stat):
            stats.append(float(cpu.attrib['user']))
            syss.append(float(system.attrib['system']))

        self.cpu_stats = stats
        self.sys_stats = syss

        return max(stats) if len(stats) > 0 else None, max(syss) if len(syss) > 0 else None

    def parse_memory_stats(self):

        root = self.tree.getroot()

        memory_stat_elements = root.findall("ns:host/ns:statistics//ns:timestamp//ns:memory//ns:memused", self.ns)

        stats = []
        for elem in memory_stat_elements:
            stats.append(int(elem.text))

        self.memory_stats = stats

        return max(stats) if len(stats) > 0 else None

    def parse_IO_stats(self):

        root = self.tree.getroot()

        io_reads_elements = root.findall("ns:host/ns:statistics//ns:timestamp//ns:io//ns:io-reads", self.ns)
        io_writes_elements = root.findall("ns:host/ns:statistics//ns:timestamp//ns:io//ns:io-writes", self.ns)

        reads_stats = []
        for elem in io_reads_elements:
            reads_stats.append(float(elem.attrib['bread']))

        self.reads_stats = reads_stats

        writes_stats = []
        for elem in io_writes_elements:
            writes_stats.append(float(elem.attrib['bwrtn']))

        self.writes_stats = writes_stats

        return max(reads_stats) if len(reads_stats) > 0 else None, max(writes_stats) if len(writes_stats) > 0 else None


    def get_all_data_points(self):
        self.parse_cpu_stats()
        self.parse_memory_stats()
        self.parse_IO_stats()

        return self.cpu_stats, self.sys_stats, self.memory_stats, self.reads_stats, self.writes_stats



if __name__ == "__main__":
    import sys

    file = sys.argv[1]

    parser = SARParser(file)
    print (parser.parse_cpu_stats())
    print (parser.parse_memory_stats())
    print (parser.parse_IO_stats())

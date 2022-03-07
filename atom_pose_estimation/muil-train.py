"""
    @roserland
    This file will train multimodel of each amino type at the same time:
    1. Training next after finish 1.
    2. Train 5 models at the same time
    3. Need estimating your GPU loads.
"""
import os, time
from multiprocessing import Process, Queue
import subprocess

AMINO_ACIDS = ['ALA', 'CYS', 'ASP', 'GLU', 'PHE', 
               'GLY', 'HIS', 'ILE', 'LYS', 'LEU',
               'MET', 'ASN', 'PRO', 'GLN', 'ARG', 
               'SER', 'THR', 'VAL', 'TRP', 'TYR']

def my_subprocess(comand_str):
    return subprocess.Popen(comand_str, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


# python unitype_main.py --gpu_id=0 --amino_type=ALA
if __name__ == '__main__':
    
    # subprocess.run("conda activate pysyft")
    i = 0
    proc_1 = Process(target=my_subprocess, args=("python unitype_main.py --gpu_id=0 --amino_type={}".format(AMINO_ACIDS[0 * i]),) )
    proc_2 = Process(target=my_subprocess, args=("python unitype_main.py --gpu_id=0 --amino_type={}".format(AMINO_ACIDS[1 * i]),) )
    proc_3 = Process(target=my_subprocess, args=("python unitype_main.py --gpu_id=1 --amino_type={}".format(AMINO_ACIDS[2 * i]),) )
    proc_4 = Process(target=my_subprocess, args=("python unitype_main.py --gpu_id=1 --amino_type={}".format(AMINO_ACIDS[3 * i]),) )
    proc_5 = Process(target=my_subprocess, args=("python unitype_main.py --gpu_id=1 --amino_type={}".format(AMINO_ACIDS[4 * i]),) )

    proc_1.start()
    proc_2.start()
    proc_3.start()
    proc_4.start()
    proc_5.start()

    proc_1.join()
    proc_2.join()
    proc_3.join()
    proc_4.join()
    proc_5.join()
    # proc_list = [proc_1, proc_2, proc_3, proc_4, proc_5, ]
    # for _proc in proc_list:
    #     print("PID: {} -- Is Finished: {}".format(_proc.pid, _proc.poll()))
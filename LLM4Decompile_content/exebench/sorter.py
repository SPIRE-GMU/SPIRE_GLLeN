import time
import gc
import os
import shutil
import subprocess

file_num = 0
name = "new_file" + str(file_num) + ".c"

compile_command = [
        "gcc", '-c', name, '-o', 'testing', "-lm"  # compile the code with GCC on Linux
]
while file_num < 10:
    

    test = subprocess.Popen(compile_command, stderr=subprocess.PIPE)
    print(test)
    file_num += 1
    name = "new_file" + str(file_num) + ".c"
    compile_command = [
        "gcc", '-c', name, '-o', 'testing', "-lm"  # compile the code with GCC on Linux
    ]





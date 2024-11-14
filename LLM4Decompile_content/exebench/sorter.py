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
    
    try:
        test = subprocess.check_output(compile_command, timeout=3,
        universal_newlines=True)
    except subprocess.CalledProcessError as exc:
        print(exc.stdout)
        if "declaration" in exc.stdout:
            print(exc.stderr)
        testing = exc.stderr
        print(test)
        file_num += 1
        name = "new_file" + str(file_num) + ".c"
        compile_command = [
            "gcc", '-c', name, '-o', 'testing', "-lm"  # compile the code with GCC on Linux
        ]





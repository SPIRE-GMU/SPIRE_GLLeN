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
        test = subprocess.run(compile_command, capture_output=True, text=True)
        print(test.stderr)
        file_num +=1
    except subprocess.CalledProcessError as exc:
        print(exc.stderr, exc.stdout)
        file_num += 1
        name = "new_file" + str(file_num) + ".c"
        compile_command = [
            "gcc", '-c', name, '-o', 'testing', "-lm"  # compile the code with GCC on Linux
        ]





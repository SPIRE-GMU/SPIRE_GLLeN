import time
import gc
import os
import shutil
import subprocess

file_num = 0
name = "new_file" + file_num + ".c"

compile_command = [
        "gcc", '-c', name, '-o', 'testing' "-lm"  # compile the code with GCC on Linux
]


subprocess.check_output(compile_command)



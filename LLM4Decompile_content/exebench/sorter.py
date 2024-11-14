import time
import gc
import os
import shutil
import subprocess

file_num = 0
name = "new_file" + str(file_num) + ".c"
jump_errors = 0
import_errors = 0
compile_command = [
        "gcc", '-c', name, '-o', 'testing', "-lm"  # compile the code with GCC on Linux
]
while file_num < 100:
    
    try:
        test = subprocess.run(compile_command, capture_output=True, text=True)
        if "expected declaration or statement" in test.stderr:
            jump_errors += 1
        elif "expected identifier or '(' at end of input" in test.stderr:
            print(test.stderr)

        file_num +=1
        name = "new_file" + str(file_num) + ".c"
        compile_command = [
            "gcc", '-c', name, '-o', 'testing', "-lm"  # compile the code with GCC on Linux
        ]
    except:
        break






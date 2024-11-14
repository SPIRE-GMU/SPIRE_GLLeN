import time
import gc
import os
import shutil
import subprocess

file_num = 0
name = "new_file" + str(file_num) + ".c"
jump_errors = 0
import_errors = 0
misc_errors = 0
compile_command = [
        "gcc", '-c', name, '-o', 'testing', "-lm"  # compile the code with GCC on Linux
]
while file_num < 1146:
    error_found = False
    try:
        test = subprocess.run(compile_command, capture_output=True, text=True)
        if "expected declaration or statement" in test.stderr:
            jump_errors += 1
            error_found = True

        elif "at end of input" in test.stderr:
            jump_errors += 1
            error_found = True

        elif 'missing terminating " character' in test.stderr:
            jump_errors += 1
            error_found = True
        elif "expected identifier or ‘(’" in test.stderr:
            jump_errors += 1
            error_found = True
        
        if "unknown type" in test.stderr:
            import_errors += 1
            error_found = True
        elif "undeclared" in test.stderr:
            import_errors += 1
            error_found = True
        elif "undefined type" in test.stderr:
            import_errors += 1
            error_found = True
        
        if not error_found:
            misc_errors += 1



        file_num +=1
        name = "new_file" + str(file_num) + ".c"
        compile_command = [
            "gcc", '-c', name, '-o', 'testing', "-lm"  # compile the code with GCC on Linux
        ]
    except:
        break


print("import errors: ", import_errors)

print('jump errors: ', jump_errors)

print('others: ', )

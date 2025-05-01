'''
This file is to test the ability to reassemble c files
it will create two folders: Successful Reassemble and Unsuccessful Reasemble
These contain copies of each C file with results.
'''
import os
import subprocess
import shutil
import tempfile
import logging
import re
from dotenv import load_dotenv
load_dotenv()


c_files_dir = os.environ.get('OUTDIR')
success_dir = "Evaluation_Testing_scripts/successful_compile/"
fail_dir = "Evaluation_Testing_scripts/unsuccessful_compile/"
garbage_path = "Evaluation_Testing_scripts/garbage_dir/"
rexe_fail = 'Evaluation_Testing_scripts/re-execute_fails/'
def main(method):
    '''
    Main takes a directory of uniquely named c files and tests them on the reassembler
    '''
    debug_num = 0
    count = {'success': 0, 'fail': 0, 'DNC': 0}
    # Loop gcc command
    if method == 0:
        
        for file in os.listdir(c_files_dir):
            if file.endswith(".c"):
                c_path = os.path.join(c_files_dir, file)
                count = recompile(c_path, success_dir, fail_dir, count, garbage_path)
                
        return count
    elif method == 1:
        for file in os.listdir(success_dir):
            if file.endswith(".c"):
                c_path = os.path.join(success_dir, file)
                count = re_exe_test(c_path, count)
                debug_num += 1
        return count




def recompile(file_path, success_path, fail_path, count, garbage_folder):
    filename = os.path.splitext(os.path.basename(file_path))[0]

    try:
        gcc_asm_command = [
            "gcc",
            "-S",
            file_path,
            "-o",
            os.path.join(garbage_folder, f"{filename}.s"),
        ]
        subprocess.run(gcc_asm_command, check=True, capture_output=True)
        shutil.copy(file_path, success_path)
        count["success"] = count["success"] + 1

        return count
    except:
        count["fail"] = count["fail"] + 1
        shutil.copy(file_path, fail_path)
        return count
    


def add_includes(files):
    includes = "#include <stdlib.h>\n#include <stdio.h>\n#include <string.h>\n#include <math.h>\n#include <time.h>\n#include <ctype.h>\n#include <assert.h>\n#include <errno.h>\n"
    for file in os.listdir(files):
        if file.endswith(".c"):
            c_path = os.path.join(files, file)
            with open(c_path, 'r+') as f:
                content = f.read()
                f.seek(0, 0)
                f.write(includes.rstrip('\r\n') + '\n' + content)
    pass

def re_exe_test(c_path, count):

    filename = os.path.splitext(os.path.basename(c_path))[0]

    name_list = re.split("_(\d+)",filename)
    print(name_list)
    filename = name_list[0]
    print(filename)
    try:
        gcc_command = [
                "gcc",
                f"-Wl,--defsym=main={filename}",
                c_path,
                "-o",
                "garbage.o",
            ]
        subprocess.run(gcc_command, check=True, capture_output=True)
        try:
            subprocess.run(['./garbage.o'])
            count['success'] = count["success"] + 1
        except:
            count['fail'] = count["fail"] + 1
            shutil.copy(c_path, rexe_fail)
    except subprocess.CalledProcessError as e:
        
        try:
            gcc_command = [
                "gcc",
                f"-Wl,--entry={filename}",
                c_path,
                "-o",
                "garbage.o",
            ]
            subprocess.run(gcc_command, check=True, capture_output=True)

            try:
                subprocess.run(['./garbage.o'])
                count['success'] = count["success"] + 1
            except:
                count['fail'] = count["fail"] + 1
                shutil.copy(c_path, rexe_fail)
        except subprocess.CalledProcessError as e:
            try:
                gcc_command = [
                    "gcc",
                    f"-Wl,-e_{filename}",
                    c_path,
                    "-o",
                    "garbage.o",
                ]
                subprocess.run(gcc_command, check=True, capture_output=True)
                try:
                    subprocess.run(['./garbage.o'])
                    count['success'] = count["success"] + 1
                except:
                    count['fail'] = count["fail"] + 1
                    shutil.copy(c_path, rexe_fail)

            except subprocess.CalledProcessError as e:
                count['DNC'] = count["DNC"] + 1

    return count

def copy_directory(src_path, dest_path):
    if not os.path.isabs(src_path) or not os.path.isabs(dest_path):
        raise ValueError("Both source and destination paths must be absolute.")

    try:
        shutil.copytree(src_path, dest_path)
        print(f"Directory copied from {src_path} to {dest_path}")
    except FileExistsError:
        print(f"Destination directory {dest_path} already exists.")
    except Exception as e:
        print(f"Error copying directory: {e}")

# print(main(0))
# add_includes(success_dir)
print(main(1))

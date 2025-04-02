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


c_files_dir = "/home/spire2/SPIRE_GLLeN/Neo4J/saved_controlResults/"
success_dir = "/home/spire2/SPIRE_GLLeN/Neo4J/successful_reassemble/"
fail_dir = "/home/spire2/SPIRE_GLLeN/Neo4J/unsuccessful_reassemble/"
garbage_path = "/home/spire2/SPIRE_GLLeN/Neo4J/garbage_dir"
def main():
    '''
    Main takes a directory of uniquely named c files and tests them on the reassembler
    '''
    # Loop gcc command
    count = {'success': 0, 'fail': 0}
    for file in os.listdir(c_files_dir):
        if file.endswith(".c"):
            c_path = os.path.join(c_files_dir, file)
            count = recompile(c_path, success_dir, fail_dir, count, garbage_path)
            
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
    
print(main())
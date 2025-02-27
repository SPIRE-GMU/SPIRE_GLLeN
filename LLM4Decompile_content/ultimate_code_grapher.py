import os
import subprocess
import shutil
import tempfile
import logging

# Setup logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Directories
c_files_dir = "/home/spire2/SPIRE_GLLeN/LLM4Decompile_content/exebench/red-team-success"
cfg_files_dir = "/home/spire2/SPIRE_GLLeN/LLM4Decompile_content/success_cfgs"
c_badfiles_dir = "/home/spire2/SPIRE_GLLeN/LLM4Decompile_content/oh_no"
asm_files_dir = "/home/spire2/SPIRE_GLLeN/LLM4Decompile_content/filled_with_asm"

# Ensure output directories exist
os.makedirs(cfg_files_dir, exist_ok=True)
os.makedirs(c_badfiles_dir, exist_ok=True)
os.makedirs(asm_files_dir, exist_ok=True)
optimization_list = ['', 'O1', 'O2', 'O3']

def generate_cfg_and_asm(c_file_path, cfg_output_dir, asm_output_dir, bad_files_dir, optimization):
    # Ensure the input file exists
    if optimization == '':
        optimization = 'O0'
    if not os.path.isfile(c_file_path):
        return

    # Extract the filename without extension
    filename = os.path.splitext(os.path.basename(c_file_path))[0]

    # Use a temporary directory for GCC output
    with tempfile.TemporaryDirectory() as temp_dir:
        # GCC command to generate the CFG dump without linking
        gcc_cfg_command = [
            "gcc",  # Adjusted to use gcc 11.4.0 for ubuntu
            "-fdump-tree-cfg-graph",
            "-c",  # Compile only, do not link (suitable for files without main)
            c_file_path,
            "-o",
            os.path.join(temp_dir, f"{filename}.out"),
            "-" + optimization,
            "-lm",
        ]
        # GCC command to generate the ASM file


        try:
            # Run the GCC command to create the CFG
            result_cfg = subprocess.run(
                gcc_cfg_command, check=True, capture_output=True, text=True
            )

            # Run the GCC command to create the ASM file

            # Locate any generated CFG file in the temporary directory
            generated_cfg_file = None
            for file in os.listdir(temp_dir):
                if file.endswith(".cfg"):
                    generated_cfg_file = os.path.join(temp_dir, file)
                    break

            if generated_cfg_file and os.path.exists(generated_cfg_file):
                # Move the generated CFG file to the specified output directory
                cfg_output_path = os.path.join(cfg_output_dir, f"{filename}.cfg")
                shutil.move(generated_cfg_file, cfg_output_path)
            else:
                
                shutil.move(
                    c_file_path,
                    os.path.join(bad_files_dir, os.path.basename(c_file_path)),
                )
                return

            # Locate the generated ASM file
            generated_asm_file = os.path.join(temp_dir, f"{filename}.s")
            if os.path.exists(generated_asm_file):
                # Move the generated ASM file to the specified output directory
                asm_output_path = os.path.join(asm_output_dir, f"{filename}.s")
                shutil.move(generated_asm_file, asm_output_path)
            else:
                shutil.move(
                    c_file_path,
                    os.path.join(bad_files_dir, os.path.basename(c_file_path)),
                )

        except subprocess.CalledProcessError as e:
            # Move the problematic C file to the bad files directory
            shutil.move(
                c_file_path, os.path.join(bad_files_dir, os.path.basename(c_file_path))
            )
            print(e)


        except Exception as e:
            # Move the problematic C file to the bad files directory
            shutil.move(
                c_file_path, os.path.join(bad_files_dir, os.path.basename(c_file_path))
            )
            print(e)


# Loop through all .c files in the c_files_dir and generate CFGs and ASM files
for x in optimization_list:
    for c_file in os.listdir(c_files_dir):
        if optimization_list[1] not in c_file and optimization_list[2] not in c_file and optimization_list[3] not in c_file and x in c_file:
            c_file_path = os.path.join(c_files_dir, c_file)
            generate_cfg_and_asm(c_file_path, cfg_files_dir, asm_files_dir, c_badfiles_dir, x)
        elif c_file.endswith(".c") and x in c_file:
            c_file_path = os.path.join(c_files_dir, c_file)
            generate_cfg_and_asm(c_file_path, cfg_files_dir, asm_files_dir, c_badfiles_dir, x)


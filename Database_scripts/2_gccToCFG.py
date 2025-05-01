import os
import subprocess
import shutil
import tempfile
import logging

# Setup logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Directories
c_files_dir = "/home/spire2/SPIRE_GLLeN/Neo4J/justin/c_files_train_real_compilable_newDB"
dot_files_dir = "/home/spire2/SPIRE_GLLeN/Neo4J/justin/dot_files_newDB"
cfg_files_dir = "/home/spire2/SPIRE_GLLeN/Neo4J/justin/cfg_files_newDB"
asm_files_dir = "/home/spire2/SPIRE_GLLeN/Neo4J/justin/asm_files_newDB"
c_badfiles_dir = "/home/spire2/SPIRE_GLLeN/Neo4J/justin/c_badfiles_newDB"

# Ensure output directories exist
os.makedirs(dot_files_dir, exist_ok=True)
os.makedirs(cfg_files_dir, exist_ok=True)
os.makedirs(asm_files_dir, exist_ok=True)
os.makedirs(c_badfiles_dir, exist_ok=True)


def generate_cfg_dot_asm(c_file_path, dot_output_dir, cfg_output_dir, asm_output_dir, bad_files_dir):
    if not os.path.isfile(c_file_path):
        logger.error(f"The file '{c_file_path}' does not exist.")
        return

    filename = os.path.splitext(os.path.basename(c_file_path))[0]

    with tempfile.TemporaryDirectory() as temp_dir:
        # Commands
        gcc_dot_cfg_command = [
            "gcc",
            "-fdump-tree-cfg-graph",
            "-c",
            c_file_path,
            "-o",
            os.path.join(temp_dir, f"{filename}.o"),
        ]
        gcc_asm_command = [
            "gcc",
            "-S",
            c_file_path,
            "-o",
            os.path.join(temp_dir, f"{filename}.s"),
        ]

        try:
            # Generate .dot and .cfg files
            subprocess.run(gcc_dot_cfg_command, check=True, capture_output=True, text=True)
            # Generate .s file
            subprocess.run(gcc_asm_command, check=True, capture_output=True, text=True)

            found_dot = False
            found_cfg = False

            for file in os.listdir(temp_dir):
                if file.endswith(".cfg.dot"):
                    shutil.move(
                        os.path.join(temp_dir, file),
                        os.path.join(dot_output_dir, f"{filename}.dot")
                    )
                    found_dot = True
                    logger.info(f"DOT file saved: {filename}.dot")

                elif file.endswith(".cfg"):
                    shutil.move(
                        os.path.join(temp_dir, file),
                        os.path.join(cfg_output_dir, f"{filename}.cfg")
                    )
                    found_cfg = True
                    logger.info(f"CFG file saved: {filename}.cfg")

            # Move .s file
            asm_path = os.path.join(temp_dir, f"{filename}.s")
            if os.path.exists(asm_path):
                shutil.move(asm_path, os.path.join(asm_output_dir, f"{filename}.s"))
                logger.info(f"ASM file saved: {filename}.s")
            else:
                logger.warning(f"No ASM file for: {filename}")
                shutil.move(c_file_path, os.path.join(bad_files_dir, os.path.basename(c_file_path)))
                return

            # Check if we got both CFG and DOT, else mark as bad
            if not (found_dot and found_cfg):
                logger.warning(f"Incomplete CFG/DOT for: {filename}")
                shutil.move(c_file_path, os.path.join(bad_files_dir, os.path.basename(c_file_path)))

        except subprocess.CalledProcessError as e:
            logger.error(f"GCC error ({filename}): {e.stderr}")
            shutil.move(c_file_path, os.path.join(bad_files_dir, os.path.basename(c_file_path)))
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            shutil.move(c_file_path, os.path.join(bad_files_dir, os.path.basename(c_file_path)))


# Process each C file
for c_file in os.listdir(c_files_dir):
    if c_file.endswith(".c"):
        c_path = os.path.join(c_files_dir, c_file)
        generate_cfg_dot_asm(c_path, dot_files_dir, cfg_files_dir, asm_files_dir, c_badfiles_dir)

print("DOT, CFG, and ASM generation complete.")

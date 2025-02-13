import os
import subprocess
import shutil
import tempfile
import logging

# Setup logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Directories
c_files_dir = "/home/spire2/SPIRE_GLLeN/ragPipeline/asm_C"
cfg_files_dir = "/home/spire2/SPIRE_GLLeN/Neo4J/temp"
c_badfiles_dir = "/home/spire2/SPIRE_GLLeN/Neo4J/c_badfiles"
asm_file_dir = "/home/spire2/SPIRE_GLLeN/Neo4J/asm_files"

# Ensure output directories exist
os.makedirs(cfg_files_dir, exist_ok=True)
os.makedirs(c_badfiles_dir, exist_ok=True)


def generate_cfg(c_file_path, output_dir, bad_files_dir):
    # Ensure the input file exists
    if not os.path.isfile(c_file_path):
        logger.error(f"The file '{c_file_path}' does not exist.")
        return

    # Extract the filename without extension
    filename = os.path.splitext(os.path.basename(c_file_path))[0]

    # Use a temporary directory for GCC  output
    with tempfile.TemporaryDirectory() as temp_dir:
        # GCC command to generate the CFG dump without linking
        gcc_command = [
            "gcc",  # Adjusted to use gcc 11.4.0 for ubuntu
            "-fdump-tree-all-graph",
            "-c",  # Compile only, do not link (suitable for files without main)
            c_file_path,
            "-o",
            os.path.join(temp_dir, f"{filename}.out"),
        ]

        try:
            # Run the GCC command to create the CFG
            result = subprocess.run(
                gcc_command, check=True, capture_output=True, text=True
            )
            logger.info(f"GCC Output: {result.stdout}")
            logger.info(f"GCC Error (if any): {result.stderr}")

            # Locate any generated CFG file in the temporary directory
            generated_cfg_file = None
            for file in os.listdir(temp_dir):
                if file.endswith(".cfg"):
                    generated_cfg_file = os.path.join(temp_dir, file)
                    break

            if generated_cfg_file and os.path.exists(generated_cfg_file):
                # Move the generated CFG file to the specified output directory
                cfg_output_path = os.path.join(output_dir, f"{filename}.cfg")
                shutil.move(generated_cfg_file, cfg_output_path)
                logger.info(f"CFG generated and saved to: {cfg_output_path}")

                # Send the CFG to Neo4j (placeholder, uncomment if applicable)
                # send_cfg_to_neo4j(cfg_output_path)
            else:
                logger.error(
                    f"CFG file was not generated as expected. Moving '{c_file_path}' to bad files directory."
                )
                shutil.move(
                    c_file_path,
                    os.path.join(bad_files_dir, os.path.basename(c_file_path)),
                )

        except subprocess.CalledProcessError as e:
            logger.error(f"GCC failed with return code {e.returncode}")
            logger.error(f"GCC Output: {e.stdout}")
            logger.error(f"GCC Error: {e.stderr}")
            # Move the problematic C file to the bad files directory
            shutil.move(
                c_file_path, os.path.join(bad_files_dir, os.path.basename(c_file_path))
            )

        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
            # Move the problematic C file to the bad files directory
            shutil.move(
                c_file_path, os.path.join(bad_files_dir, os.path.basename(c_file_path))
            )


# Loop through all .c files in the c_files_dir and generate CFGs
for c_file in os.listdir(c_files_dir):
    if c_file.endswith(".c"):
        c_file_path = os.path.join(c_files_dir, c_file)
        generate_cfg(c_file_path, cfg_files_dir, c_badfiles_dir)

print("CFG generation process completed.")

"""
Craig Kimball

Control script to iterate over a directory passing arguments to assemblySearchHandler
Accepts directory of .s file
Creates directory of output c files

"""

import os
import subprocess
import sys
import io
import assemblySearchHandler
from dotenv import load_dotenv

load_dotenv()

OUTDIR = os.environ.get("OUTDIR")

stdout = sys.stdout


def batch_process_assembly(input_directory):
    sys.stdout = stdout

    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    assemblySearchHandler.loadModel()

    os.makedirs(OUTDIR, exist_ok=True)

    handler_script = "assemblySearchHandler.py"

    for filename in os.listdir(input_directory):

        if filename.endswith(".o"):

            # control flow stop
            # input(f"File: {filename}")

            input_file_path = os.path.join(input_directory, filename)
            filenametrim = filename.split(".o")[0]
            output_file_path = os.path.join(OUTDIR, f"{filenametrim}.c")

            if os.path.isfile(output_file_path):
                print(f"{filenametrim} already calculated, continuing")
                continue

            print(input_file_path)
            print(output_file_path)

            try:
                # print("command")
                # result = subprocess.run(
                #     ['python3', handler_script, input_file_path],
                #     capture_output=True,
                #     text=True,
                #     check=True
                # )

                # stores args because assemblySearchHandler references them
                original_argv = sys.argv
                sys.argv = ["assemblySearchHandler.py", input_file_path]
                sys.stdout = io.StringIO()

                result = assemblySearchHandler.main()

                # Restore original sys.argv
                sys.argv = original_argv

                with open(output_file_path, "w") as output_file:
                    output_file.write(result)
                sys.stdout = stdout
                print(f"Processed {filename}: Result saved to {output_file_path}")

            except Exception as e:
                print(f"Unexpected error in {filename}: {e}")


def main():
    input_directory = sys.argv[1]

    if not os.path.isdir(input_directory):
        sys.exit(1)

    batch_process_assembly(input_directory)


if __name__ == "__main__":
    main()

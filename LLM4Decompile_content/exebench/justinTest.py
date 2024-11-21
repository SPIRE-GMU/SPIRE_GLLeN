import os
from datasets import load_dataset

def main():
    # Load the dataset
    dataset = load_dataset("jordiae/exebench", split="test_synth")

    # Ensure the subfolder 'DataBase' exists
    folder = "DataBase_test_synth"
    os.makedirs(folder, exist_ok=True)

    # Open files to store function summaries and missing information
    with open(os.path.join(folder, "functionDataBase.txt"), "w") as fname_file, \
         open(os.path.join(folder, "missingData.txt"), "w") as missing_file:

        for row in dataset:
            fname = row["fname"]
            func_def = row["func_def"]
            asm_code = row.get("asm", {}).get("code", [None])[0]  # Use .get() for safe access

            # Initialize the missing list at te start of each iteration
            missing = []

            # Check for missing function or assembly and log it
            if not func_def:
                missing.append("C function")
            if asm_code is None:
                missing.append("Assembly")

            # If any part is missing, log it and skip writing files
            if missing:
                missing_file.write(f"{fname} is missing: {', '.join(missing)}\n")
                continue

            # Check for long filename
            if len(fname) > 255:  # Adjust this limit if necessary
                missing_file.write(f"{fname} cannot be processed: filename too long\n")
                continue

            # Write the C function to a .c file named after the function
            c_file_path = os.path.join(folder, f"{fname}.c")
            with open(c_file_path, "w") as c_file:
                c_file.write(func_def + "\n")

            # Write the assembly code to a .asm file named after the function
            asm_file_path = os.path.join(folder, f"{fname}.asm")
            with open(asm_file_path, "w") as asm_file:
                asm_file.write(asm_code + "\n")

            # Collect metadata for the function summary
            c_size = os.path.getsize(c_file_path)  # Get file size in bytes
            asm_size = os.path.getsize(asm_file_path)  # Get file size in bytes

            c_line_count = len(func_def.splitlines())  # Count lines in C code
            asm_line_count = len(asm_code.splitlines())  # Count lines in assembly

            # Write metadata to functionDataBase.txt
            fname_file.write(f"Function Name: {fname}\n")
            fname_file.write(f"  C file size: {c_size} bytes, Lines of code: {c_line_count}\n")
            fname_file.write(f"  ASM file size: {asm_size} bytes, Lines of code: {asm_line_count}\n")
            fname_file.write("  Additional info: C function and assembly code are present.\n")
            fname_file.write("*****\n")

    print("Files have been written successfully into the DataBase folder.")

if __name__ == "__main__":
    main()


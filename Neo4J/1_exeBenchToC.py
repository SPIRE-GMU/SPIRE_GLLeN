from datasets import load_dataset
import os

# Load the dataset from the specified location
dataset = load_dataset("jordiae/exebench", split="test_real", trust_remote_code=True)

# Directory where you want to save the .c files
c_files_dir = "/home/spire2/SPIRE_GLLeN/Neo4J/c_files"

# Create the directory if it doesn't exist
os.makedirs(c_files_dir, exist_ok=True)

# Set a limit if desired (or remove limit variable to iterate over all entries)
limit = None
processed_count = 0

# Loop through the dataset to extract C files
for idx, row in enumerate(dataset):
    # Stop if processed count reaches limit
    if limit and processed_count >= limit:
        break

    # Extract C code from the row
    c_code = row.get("func_def")
    if c_code:
        # Generate the base filename
        base_fname = row.get("fname", f"sample_{idx}")
        c_filename = f"{base_fname}.c"
        full_path = os.path.join(c_files_dir, c_filename)

        # Ensure unique filename by appending a number if file already exists
        file_count = 1
        while os.path.exists(full_path):
            c_filename = f"{base_fname}_{file_count}.c"
            full_path = os.path.join(c_files_dir, c_filename)
            file_count += 1

        # Save the retrieved file to the local directory for processing
        with open(full_path, 'w') as f:
            f.write(c_code)

        processed_count += 1

print(f"Processed {processed_count} files.")

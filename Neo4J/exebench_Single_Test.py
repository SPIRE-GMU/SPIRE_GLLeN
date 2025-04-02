import os
from datasets import load_dataset
import json

def main():
    # Load just one sample from the dataset
    dataset = load_dataset("jordiae/exebench", split="test_synth[:1]")

    row = dataset[0]

    # Output folder and file setup
    folder = "SingleSampleDump"
    os.makedirs(folder, exist_ok=True)
    output_file = os.path.join(folder, f"{row['fname']}_full_dump.txt")

    with open(output_file, "w") as f:
        for key, value in row.items():
            f.write(f"=== {key} ===\n")
            if isinstance(value, dict) or isinstance(value, list):
                f.write(json.dumps(value, indent=2) + "\n")
            else:
                f.write(str(value) + "\n")
            f.write("\n")

    print(f"Full data written to {output_file}")

if __name__ == "__main__":
    main()

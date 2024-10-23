'''
This file iterates through the exebench
dataset and tests if LLM4Decompile is able to 
decompile a recombilable C file
'''
from datasets import load_dataset
import os
import subprocess
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from accelerate import Accelerator
import argparse

MODEL_PATH = '/home/spire2/LLM4Decompile/llm4decompile-22b-v2'

func0=""

def main():
    '''
    The main function controling all other functions
    '''
    # 1) Load dataset split. In this case, synthetic test split
    dataset = load_dataset('jordiae/exebench', split='test_synth') # , use_auth_token=True)
  # 2) Iterate over dataset
    print(len(dataset))
#    row = 0
#    while row < 1:
    for row in dataset:
        # Do stuff with each row
        # 3) Option A: Manually access row fields. For instance, access the function definition:
        #pprint.pp(dataset[row])
        # print('*****')
        # print(row['func_def'])  # print function definition
        # print('*****')
        # print(row['asm']['code'][0])  # print assembly with the first target, angha_gcc_x86_O0
        # print first I/O example (synthetic)
        #print('Input:', exebench_dict_to_dict(row['synth_io_pairs']['input'][0]))
        #print('Output:', exebench_dict_to_dict(row['synth_io_pairs']['output'][0]))
        #print(row['synth_exe_wrapper'][0])  # print C++ wrapper to run function with IO
        # You can manually compile, run with IO, etc
 #       row += 1


    parser = argparse.ArgumentParser(description='Decompile assembly code')
    parser.add_argument('-c', '--compile', action='store_true', help='Compile the code')
    parser.add_argument('-d', '--decompile', action='store_true', help='Decompile the assembly code')
    parser.add_argument('-f', '--file', type=str, help='Input file path')
    parser.add_argument('-t', '--function', type=str, help='Function name to decompile')
    parser.add_argument('-m', '--model', type=str, help='Model size (6.7b, 9b, 22b)')
    args = parser.parse_args()

    if args.compile:
        assemble(args.file, args.model, args.function)
    elif args.decompile:
        if args.file and args.function and args.model:
            if args.model not in MODEL_PATHS:
                raise ValueError("Invalid model option. Choose from '6.7b', '9b', or '22b'.")
            run(args.file, args.model, args.function)
        else:
            print("Error: Please provide the file name, function name, and model size.")
    else:
        # Clear the terminal screen
        os.system('clear')

        # Display the ASCII art
        print(TITLE_CARD)

        # Get a list of files with .c or .o extensions in the current working directory
        current_dir = os.path.dirname(__file__)
        files = [f for f in os.listdir(current_dir) if f.endswith('.c') or f.endswith('.asm')]

        # Prompt the user for the file name and display the list of files
        print("Select the file name of the file:")
        # print("Available files:")
        for i, file in enumerate(files, start=1):
            print(f"{i}. {file}")

        # Get user input as a string
        file_input = input()

        # Check if the input is a digit
        if file_input.isdigit():
            file_number = int(file_input)
            if file_number < 1 or file_number > len(files):
                raise ValueError("Invalid file number.")
            file_name = files[file_number - 1]
        else:
            # Check if the input is a file name
            if file_input in files:
                file_name = file_input
            else:
                raise ValueError("Invalid file name.")
            

        if file_name.endswith('.c'):
            
            # Get user input for the function name
            print("Enter the function name to decompile: [main]")
            func0 = input()

            if (assemble(file_name, func0)==0):
                print("Compilation successful.")    
             
            return 0
            
        else:
            # Get user input for the model size

            

            # Get user input for the function name
            print("Enter the function name to decompile: [main]")
            func0 = input()

            # Run the decompilation process
            return run(file_name, model_size, func0)

    #TODO


    return 0

def assemble():
    #TODO
    return 0

def load_model():
    #TODO
    return 0



def decompiler():
    #TODO
    return 0


def save_failure():
    #TODO
    return 0

main()
from datasets import load_dataset
import pprint

def main():
    dataset = load_dataset('jordiae/exebench', split='test_real') # , use_auth_token=True)
    for row in dataset:
        print('*****')
        print(row['fname'])  # print function tite
        print('*****')
        print(row['func_def'])  # print function definition
        print('*****')
        print(row['asm']['code'][0])  # print assembly with the first target, angha_gcc_x86_O0
main()

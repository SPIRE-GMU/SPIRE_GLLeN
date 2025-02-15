from datasets import load_dataset
import pprint

burner_file = "test.c"


def main():

    dataset = load_dataset(
        "jordiae/exebench", split="test_real"
    )  # , use_auth_token=True)

    row = 0
    while row < 1:
        pprint.pp(dataset[row])
        with open(burner_file, "w") as f:
            f.write("void main()\n{}\n" + dataset[row]["func_def"])

        row += 1
    return 0


main()
# 1) Load dataset split. In this case, synthetic test split
# 2) Iterate over dataset
# for row in dataset:
# Do stuff with each row
# 3) Option A: Manually access row fields. For instance, access the function definition:
# pprint.pp(dataset[row])
# print('*****')
# print(row['func_def'])  # print function definition
# print('*****')
# print(row['asm']['code'][0])  # print assembly with the first target, angha_gcc_x86_O0
# print first I/O example (synthetic)
# print('Input:', exebench_dict_to_dict(row['synth_io_pairs']['input'][0]))
# print('Output:', exebench_dict_to_dict(row['synth_io_pairs']['output'][0]))
# print(row['synth_exe_wrapper'][0])  # print C++ wrapper to run function with IO
# You can manually compile, run with IO, etc

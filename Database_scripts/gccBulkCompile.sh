# First create the output directory if it doesn't exist
#mkdir -p c_files_train_real_asm

for file in c_files/*.c; do
    filename=$(basename "$file")
    gcc -c "$file" -o "c_files_binaries/${filename%.c}.o"
    echo "Compiled $file into c_files_train_real_asm/${filename%.c}.o"
done
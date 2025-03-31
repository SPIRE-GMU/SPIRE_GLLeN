# First create the output directory if it doesn't exist
mkdir -p c_files_train_real_asm

for file in /c_files_train_real/*.c; do
    filename=$(basename "$file")
    gcc -S "$file" -o "c_files_train_real_asm/${filename%.c}.s"
    echo "Compiled $file into c_files_train_real_asm/${filename%.c}.s"
done
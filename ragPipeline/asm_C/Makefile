# Compiler and flags
CC = gcc
CFLAGS = -Wall
LDFLAGS = -lm  # Linker flags, including the math library

# Get all .c files in the directory
SRC = $(wildcard *.c)

# Create an executable for each .c file
EXEC = $(SRC:.c=)

# Default target
all: $(EXEC)

# Target to compile each .c file into its own executable
%: %.c
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)
	@echo "Executable '$@' has been created."

# Target to compile all .c files into assembly files
asm: $(SRC:.c=.s)
	@echo "Assembly files have been generated."

# Pattern rule to compile .c to .s (assembly)
%.s: %.c
	$(CC) -S $< -o $@

# Target to clean up all executables and assembly files
clean:
	rm -f $(EXEC) *.s
	@echo "All compiled executables and assembly files have been removed."

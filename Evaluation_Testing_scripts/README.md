# Testing suite
This folder contains the files used to test the results enmasse for re-compilability and re-executability

## LLM4Decompile testing
The basic tests for LLM4Decompile's re-compilability are in the llm4decompile_testing.py this requires a model path for llm4decompile from hugging face and sub directories for the results files.

In the end it should print the number of files that llm4 produced that compiled and didn't

## Re-exe-tester.py
This file contains functions to test recompilability and reexecutability on files as well as a function to add common C includes to the top of every c file in a given directory.
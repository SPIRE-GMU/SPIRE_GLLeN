# Library imports
import os

# ADD IMPORTS FOR OTHER LIBRARIES
from Neo4J import gccToCFG
from imports import exeBenchToC

def main():
    print("=========================================================================")
    input_file = str(input("Enter file path ('x' to cancel): "))
    # CATCH STATEMENT
    if input_file == 'x':
        print("Cancelled.\nExiting...")
        return
    while os.path.isdir(input_file) != True:
        input_file = str(input("Does not exist.\nEnter file path ('x' to cancel): "))
        if input_file == 'x':
            print("Cancelled.\nExiting...")
            print("=========================================================================")
            break


    if (os.path.isdir(input_file)):
        print('\nExists. Proceeding...\n')
        if (os.path.isdir(input_file)):
            # MAKE THE IMPORTS RUN ON THE FILE
            pass
        elif (os.path.isfile(input_file)):
            # MAKE THE IMPORTS RUN ON A WHOLE DIRECTORY
            pass
        output = input('Enter output path: ')
        #LOOPING CATCH STATEMENT FOR INCORRECT PATHING
        while (os.path.isdir(output) != True):
            output = input("Enter output path ('x' to cancel): ")
            if output == 'x':
                break
        
        print('\nDONE\n')
    
main()
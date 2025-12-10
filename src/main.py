
import os
import sys

from train_decpomdp import train_decpomdp
from test_baselines import test_baselines
from tom_training_testing import tom_training_testing

if __name__ == "__main__":
    run_main = True
    while run_main:
        choice = None
        while choice not in ["1", "2", "3", "4", "5"]:
            print("1. Train DecPOMDP Agents")
            print("2. Test Baseline Agents")
            print("3. Train Theory of Mind (ToM) Agents")
            print("4. Do All!")
            print("5. Exit")
            choice = input("Select an option (1-4): ")
            print()
        if choice == "1":
            train_decpomdp()
        elif choice == "2":
            test_baselines()
        elif choice == "3":
            tom_training_testing()
        elif choice =="4":
            train_decpomdp()
            test_baselines()
            tom_training_testing()
            run_main = False
        elif choice == "5":
            run_main = False
        else:
            print("Invalid choice. Please try again.")
            input("Press Enter to continue...")
    print("Exiting the program.")

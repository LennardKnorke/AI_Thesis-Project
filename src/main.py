# main.py

import os
import sys

from hypersearch_baselines import hypersearch_baselines
from train_test_baselines import train_test_baselines
from train_worldmodel import train_worldmodel
from tom_training_testing import tom_training_testing

if __name__ == "__main__":
    run_main = True
    while run_main:
        choice = None
        while choice not in ["1", "2", "3", "4", "5"]:
            print("1. Hyperparameter Search - Baselines Agents")
            print("2. Train/Test Baseline Agents")
            print("3. Train World-Model")
            print("4. (Hyperparameter Search + )Train/Test ToM-Agents")
            print("5. Do All!")
            print("6. Exit")
            choice = input("Select an option (1-5): ")
            print()
        # Execute the selected option
        if choice == "1":
            hypersearch_baselines()
        elif choice == "2":
            train_test_baselines()
        elif choice == "3":
            train_worldmodel()
        elif choice == "4":
            tom_training_testing()
        elif choice =="5":
            hypersearch_baselines()
            train_test_baselines()
            train_worldmodel()
            tom_training_testing()
            run_main = False
        elif choice == "6":
            run_main = False
        else:
            print("Invalid choice. Please try again.")
            input("Press Enter to continue...")
    print("Exiting the program.")
###################################
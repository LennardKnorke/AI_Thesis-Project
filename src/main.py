# main.py
from hypersearch_baselines import hypersearch_baselines
from hypersearch_tom import hypersearch_tom

from train_worldmodel import train_worldmodel

from train_test_baselines import train_test_baselines
from train_test_tom import train_test_tom

if __name__ == "__main__":
    run_main = True
    while run_main:
        choice = None
        while choice not in ["1", "2", "3", "4", "5", "6", "7"]:
            print("1. Hyperparameter Search - Baselines Agents")
            print("2. Train/Test - Baseline Agents")
            print("3. Hyperparameter Search - ToM World-Model (+ Train/Test)")
            print("4. Hyperparameter Search - ToM-Agent")
            print("5. Train/Test - ToM-Agent")
            print("6. Do All!")
            print("7. Exit")
            choice = input("Select an option (1-7): ")
        # Baselines
        if choice == "1":
            hypersearch_baselines() 
        elif choice == "2":
            train_test_baselines()
        # World Model
        elif choice == "3":
            train_worldmodel()
        # ToM Agent
        elif choice == "4":
            hypersearch_tom()
        elif choice == "5":
            train_test_tom()
        # EVERYTHING
        elif choice =="6":
            # Baselines
            hypersearch_baselines()
            train_test_baselines()
            # World Model
            train_worldmodel()
            # ToM Agent
            hypersearch_tom()
            train_test_tom()
            run_main = False
        elif choice == "7":
            run_main = False
        else:
            print("Invalid choice. Please try again.")
            input("Press Enter to continue...")
    print("Done.")
###################################
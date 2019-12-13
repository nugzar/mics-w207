# Quick
Download the following files from Nugzar’s github url https://github.com/nugzar/mics-w207/tree/master/final
- w207_project_data_engineering.ipynb
- w207_project_bruteforce.py
- w207_project.ipynb
- bruteforced_engineered.csv

Place the files in the same folder
Run w207_project.ipynb


# Full process
**Step 1:** Download the following files from kaggle - https://www.kaggle.com/c/microsoft-malware-prediction/data
- train.csv
- test.csv

**Step 2:** Download the following files from Nugzar’s github url https://github.com/nugzar/mics-w207/tree/master/final
- w207_project_data_engineering.ipynb
- w207_project_bruteforce.py
- w207_project.ipynb

**Step 3:** Place the files download in step 1 & step 2 in the same folder

**Step 4:** Run the file w207_project_data_engineering.ipynb
The file bruteforced_engineered.csv will be generated. This file is used as an input for the file "w207_project.ipynb"

**Step 5:** Optional - Run the file w207_project_bruteforce.py
Run "w207_project_bruteforce.py regularized.csv". 

This command will start bruteforcing the regularized.csv dataset. It will try dropping the columns one after another and finding the optimal columns. The process will be logged in w207_project_bruteforce.log file. The columns and the accuracy score will be logged to the file and when the process ends, it will be possible to select the best score and the list of columns to be used for getting that score

These columns are already included in w207_project_data_engineering.ipynb file at the bottom for information purposes.

**Step 6:** Run the file w207_project.ipynb
This notebook takes "bruteforced_engineered.csv" and applies different models to make predictions. The details can be seen in the comments inside the file.

import pandas as pd
from subprocess import Popen, PIPE, DEVNULL, STDOUT

DEBUG = 1

def execute_command(cmd):
    """Wrapper for executing CLI commands.
    Args:
        cmd (str): Command to execute.
    """
    if (DEBUG):
        print(cmd)
        process = Popen(cmd, shell=True, stdout=PIPE)
    else:
        process = Popen(cmd, shell=True, stdout=DEVNULL, stderr=STDOUT)
    process.communicate()
    
    return cmd

list_of_dfs = []
for i in range(1, 23):
    chr_df = pd.read_csv(f"results_csv/hicdiffusion/chr{i}.csv")
    chr_df["chr"] = f"chr{i}"
    list_of_dfs.append(chr_df)
    
df = pd.concat(list_of_dfs)

for index, row in df.iterrows():
    execute_command(f"sbatch corigami_predict.slurm --chr {row['chr']} --start {row['pos']}")
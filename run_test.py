import argparse
import os
import shutil
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


parser = argparse.ArgumentParser(description='HiCDiffusion test pipeline.')

parser.add_argument('-v', '--val_chr', required=True)
parser.add_argument('-t', '--test_chr', required=True)
parser.add_argument('-f', '--file', required=False)

args = parser.parse_args()

print("Running HiCDiffusion test pipeline. The configuration:")
print(args)
print()


hic_filename = args.file
if(hic_filename != ""):
    filename_prefix = "_"+hic_filename
else:
    filename_prefix = ""

execute_command(f"sbatch --dependency=singleton --job-name=HiCDiffusion{filename_prefix}_test_{args.test_chr}_val_{args.val_chr} --output=models/hicdiffusion{filename_prefix}_test_{args.test_chr}_val_{args.val_chr}/test_hicdiffusion.log test_hicdiffusion.slurm -t {args.test_chr} -v {args.val_chr} -m models/hicdiffusion{filename_prefix}_test_{args.test_chr}_val_{args.val_chr}/best_val_loss_hicdiffusion.ckpt -f {hic_filename}")
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

args = parser.parse_args()

print("Running HiCDiffusion test pipeline. The configuration:")
print(args)
print()

main_folder = "models/hicdiffusion_test_%s_val_%s" % (args.test_chr, args.val_chr)

execute_command(f"sbatch --dependency=singleton --job-name=HiCDiffusion_test_{args.test_chr}_val_{args.val_chr} --output=models/hicdiffusion_test_{args.test_chr}_val_{args.val_chr}/test_hicdiffusion.log test_hicdiffusion.slurm -t {args.test_chr} -v {args.val_chr} -m models/hicdiffusion_test_{args.test_chr}_val_{args.val_chr}/best_val_loss_hicdiffusion.ckpt")
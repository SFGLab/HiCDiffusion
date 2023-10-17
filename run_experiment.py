import argparse
import os
import shutil
from subprocess import Popen, PIPE, DEVNULL, STDOUT
import math

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


parser = argparse.ArgumentParser(description='HiCDiff pipeline.')

parser.add_argument('-v', '--val_chr', required=True)
parser.add_argument('-t', '--test_chr', required=True)

args = parser.parse_args()

print("Running HiCDiff pipeline. The configuration:")
print(args)
print()

main_folder = "models/hicdiff_test_%s_val_%s" % (args.test_chr, args.val_chr)

if os.path.exists(main_folder) and os.path.isdir(main_folder):
    shutil.rmtree(main_folder)
os.makedirs(main_folder)


execute_command(f"sbatch --dependency=singleton --job-name=HiCDiff_test_{args.test_chr}_val_{args.val_chr} --output=models/hicdiff_test_{args.test_chr}_val_{args.val_chr}/train_hicdiff_encoder_decoder.log train_hicdiff_encoder_decoder.slurm -t {args.test_chr} -v {args.val_chr}")
execute_command(f"sbatch --dependency=singleton --job-name=HiCDiff_test_{args.test_chr}_val_{args.val_chr} --output=models/hicdiff_test_{args.test_chr}_val_{args.val_chr}/train_hicdiff.log train_hicdiff.slurm -t {args.test_chr} -v {args.val_chr} -m models/hicdiff_test_{args.test_chr}_val_{args.val_chr}/best_val_loss_encoder_decoder.ckpt")
execute_command(f"sbatch --dependency=singleton --job-name=HiCDiff_test_{args.test_chr}_val_{args.val_chr} --output=models/hicdiff_test_{args.test_chr}_val_{args.val_chr}/test_hicdiff.log test_hicdiff.slurm -t {args.test_chr} -v {args.val_chr} -m models/hicdiff_test_{args.test_chr}_val_{args.val_chr}/best_val_loss_hicdiff.ckpt")
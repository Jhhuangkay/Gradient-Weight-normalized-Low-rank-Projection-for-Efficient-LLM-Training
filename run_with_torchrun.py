import sys
import subprocess

# Prepare the torchrun command
command = [
    "torchrun",
    "--standalone",
    "--nproc_per_node=1",
    "test_train.py"
] + sys.argv[1:]

# Execute the command
subprocess.run(command)

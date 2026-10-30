#!/bin/bash
# Runs a command file with P concurrent processes on the job's GPU (one A100 fits ~8 of these models).
# Usage (inside run.sbatch, cwd = src): bash ../experiments/run_commands.sh cmds.txt [P]
P=${2:-8}
xargs -P $P -I{} bash -c {} < $1
echo "RUN_COMMANDS_DONE $1"

#!/bin/bash -l
#
# Generic CSCS data-transfer job for the dedicated `xfer` Slurm partition.
# Adapted from the CSCS docs "Internal Transfer" section:
#   https://docs.cscs.ch/storage/transfer/
#
# The xfer partition is purpose-built for moving data between the internal CSCS
# file systems (scratch <-> store). Use it instead of copying on a login node or
# burning a GPU/compute node on a plain `cp`.
#
# Usage:
#   sbatch stage.sbatch <source> <destination>
#   sbatch stage.sbatch <source> <destination> <next_script.sbatch>
#
#   <source>        file or directory to copy
#   <destination>   where to copy it to
#   <next_script>   (optional) a Slurm script to submit ONLY IF this copy
#                   succeeds, via --dependency=afterok. Lets you chain a
#                   stage-in -> production -> stage-out pipeline.
#
# Examples:
#   # one-off copy from scratch to store
#   sbatch stage.sbatch \
#     /iopsstor/scratch/cscs/djanjetovic/tabular_ablation/tokenized \
#     /capstor/store/cscs/swissai/a139/djanjetovic/tabular_ablation/tokenized
#
#   # stage-in, then run training only if the copy succeeded
#   sbatch stage.sbatch $STORE/data $SCRATCH/data submit_tabular_ablation.sh

#SBATCH --time=06:00:00
#SBATCH --ntasks=1
#SBATCH --partition=xfer
#SBATCH --job-name=stage
#SBATCH --output=stage-%j.out
#SBATCH --error=stage-%j.err

# rsync -a preserves timestamps/permissions and is restartable (re-running skips
# files already copied). The trailing-slash convention matters for directories:
# pass source WITHOUT a trailing slash to copy the directory itself.
command="rsync -av"

SRC="$1"
DEST="$2"
NEXT="$3"

if [ -z "$SRC" ] || [ -z "$DEST" ]; then
    echo "ERROR: need <source> and <destination>."
    echo "Usage: sbatch stage.sbatch <source> <destination> [next_script.sbatch]"
    exit 1
fi

# Make sure the parent of the destination exists.
mkdir -p "$(dirname "$DEST")"

echo -e "$SLURM_JOB_NAME started on $(date):\n  $command $SRC -> $DEST\n"
srun -n $SLURM_NTASKS $command "$SRC" "$DEST"
RC=$?
echo -e "\n$SLURM_JOB_NAME finished on $(date) with exit code $RC"

if [ $RC -ne 0 ]; then
    echo "Transfer FAILED — not submitting any dependent job."
    exit $RC
fi

# Optionally chain a follow-up job that runs only if this transfer succeeded.
if [ -n "$NEXT" ]; then
    # xfer partition sets a per-CPU memory constraint; unset it before submitting
    # a job destined for a different partition.
    unset SLURM_MEM_PER_CPU
    echo "Submitting dependent job: $NEXT"
    sbatch --dependency=afterok:$SLURM_JOB_ID "$NEXT"
fi

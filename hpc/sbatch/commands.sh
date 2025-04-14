ls
pwd
whoami
id
groups
echo $HOME
# check OS
uname -a 
# print volume info
df -h
# Shows filesystem type and mount options
df -T /ceph       

# Show all node information
sinfo -s
# more detail
sinfo -o "%P %l %D %t %N"
# Show information about nodes on the gpu partition
sinfo -p gpu
# Show Job History
sacct -u $USER
# Show Completed Jobs With Details
sacct -u $USER --format=JobID,JobName,Partition,Account,AllocCPUS,State,ExitCode,Elapsed

cd src/
sbatch run_b_act.sbatch

squeue --me
watch -n 30 squeue -u $USER
# Show jobs running on gpu node gn35
squeue --nodelist=gn35
# Show (job) node details
scontrol show job 56539837
# Show all partitions info
scontrol show partition 
# Show gpu nodes 
scontrol show nodes | grep gpu
# Show priorities in squeue - the Q param
#  larger numbers indicate higher priority in Slurm
squeue -o "%.7i %.9P %.8j %.8u %.2t %.10M %.6D %9Q" -p gpu 
# Cancle job
scancel 56539837

# check disk usage
du -sh /ceph/hpc/data/FRI/dp8949/data
du -sh /ceph/hpc/data/FRI/dp8949/huggingface/

# run nvidia-smi on gpu nodes
srun --nodelist=nsc-vfp002 --gres=gpu:1 -p e7 nvidia-smi
srun -p gpu --gres=gpu:4 -n 1 -c 1 nvidia-smi

# print latest log files
cat $(ls -t src/logs/*.out | head -n 1) $(ls -t src/logs/*.err | head -n 1) 2>/dev/null

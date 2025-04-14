#!/bin/bash
#SBATCH --job-name=llama3_70b_monitor1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4
#SBATCH --mem=250G
#SBATCH --partition=gpu
#SBATCH --time=48:00:00
#SBATCH --output=logs/build_act_%j.out
#SBATCH --error=logs/build_act_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=x@gmail.com

echo "=================================================="
echo "Running on host: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo "=================================================="

# Load necessary modules
module load Python/3.11.5-GCCcore-13.2.0

# Activate virtual environment
source ../test_llama/venv/bin/activate

# Load environment variables from .env file, if available
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
else
    echo "WARNING: .env file not found! Make sure HF_TOKEN and HF_HOME are set."
fi

# Ensure HF_HOME is correctly set
export HF_HOME="${HF_HOME:-/ceph/hpc/data/FRI/dp8949/huggingface}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

echo "HF Token (masked): ${HF_TOKEN:0:8}******"
echo "HF Home: $HF_HOME"
echo "Output Directory: /ceph/hpc/data/FRI/dp8949/data/activations_data/"

# Set monitoring directory (adjust if needed)
MONITOR_DIR=$HOME/monitoring
BIN_DIR=$MONITOR_DIR/bin
LOG_DIR=$MONITOR_DIR/logs

pkill node_exporter prometheus dcgm_exporter grafana
echo "Starting monitoring services..."

# Start Node Exporter (listens on port 9101)
nohup $BIN_DIR/node_exporter --web.listen-address=:9101 > $LOG_DIR/node_exporter.log 2>&1 &
NODE_EXPORTER_PID=$!

# Start NVIDIA DCGM Exporter (listens on port 9401) (need root privilages)
nohup $BIN_DIR/dcgm_exporter --address=:9401 -f $BIN_DIR/default-counters.csv > $LOG_DIR/dcgm_exporter.log 2>&1 &
DCGM_EXPORTER_PID=$!

# Start Prometheus (listens on port 9090)
nohup $BIN_DIR/prometheus --config.file=$BIN_DIR/prometheus.yml --web.listen-address=:9090 --storage.tsdb.path=$LOG_DIR/prom_data > $LOG_DIR/prometheus.log 2>&1 &
PROMETHEUS_PID=$!

# Start Grafana (listens on port 3000)
nohup $BIN_DIR/grafana server --homepath $BIN_DIR/grafana-v11.6.0 --config $BIN_DIR/custom.ini > $LOG_DIR/grafana.log 2>&1 &
GRAFANA_PID=$!

echo "Monitoring services started:"
echo "  Node Exporter PID: $NODE_EXPORTER_PID"
echo "  DCGM Exporter PID: $DCGM_EXPORTER_PID"
echo "  Prometheus PID: $PROMETHEUS_PID"
echo "  Grafana PID: $GRAFANA_PID"

# Give the monitoring services a few seconds to initialize
sleep 10

echo "Starting deep learning job..."

# Run LLaMA activation extraction job
srun python build_activations.py \
    --model_name "meta-llama/Llama-3.3-70B-Instruct" \
    --cache_dir "$HF_HOME" \
    --layer_index 40 \
    --data_len 300000 \
    --batch_size 8 \
    --file_size 81920 \
    --vector_size 8192 \
    --output_dir "/ceph/hpc/data/FRI/dp8949/data/activations_data/"

echo "Deep learning job finished. Shutting down monitoring services..."

# Kill monitoring services
kill $NODE_EXPORTER_PID $DCGM_EXPORTER_PID $PROMETHEUS_PID $GRAFANA_PID

echo "=================================================="
echo "Job finished at: $(date)"
echo "=================================================="


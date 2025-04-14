mkdir -p $HOME/monitoring/{bin,logs}

# make prometheus.yml
touch $HOME/monitoring/bin/prometheus.yml
# copy the file content

# Install Node Exporter
cd $HOME/monitoring/bin
wget https://github.com/prometheus/node_exporter/releases/download/v1.9.0/node_exporter-1.9.0.linux-amd64.tar.gz
tar -xzvf node_exporter-1.9.0.linux-amd64.tar.gz
mv node_exporter-1.9.0.linux-amd64/node_exporter .
rm -rf node_exporter-1.9.0.linux-amd64*

# Install Prometheus
wget https://github.com/prometheus/prometheus/releases/download/v2.53.4/prometheus-2.53.4.linux-amd64.tar.gz
tar -xzvf prometheus-2.53.4.linux-amd64.tar.gz
mv prometheus-2.53.4.linux-amd64/prometheus .
rm -rf prometheus-2.53.4.linux-amd64*

# Download the specified version of DCGM Exporter (DCGM on hpc is 3.3.8)
wget https://github.com/NVIDIA/dcgm-exporter/archive/refs/tags/3.3.8-3.6.0.tar.gz -O dcgm-exporter-3.3.8-3.6.0.tar.gz
# Extract the downloaded tarball
tar -xzvf dcgm-exporter-3.3.8-3.6.0.tar.gz
# Navigate into the extracted directory
cd dcgm-exporter-3.3.8-3.6.0
# Load the Go module (adjust the version as needed)
module load Go/1.22.1
# Verify the Go installation
go version
# Build the dcgm_exporter binary and place it in the specified directory
go build -o $HOME/monitoring/bin/dcgm_exporter ./cmd/dcgm-exporter
# Navigate out of the source directory
cd ..
# Clean up by removing the downloaded and extracted files
rm -rf dcgm-exporter-3.3.8-3.6.0*

# Install Grafana
# cd $HOME/monitoring/bin
wget https://dl.grafana.com/enterprise/release/grafana-enterprise-11.6.0.linux-amd64.tar.gz
tar -zxvf grafana-enterprise-11.6.0.linux-amd64.tar.gz
mv grafana-v11.6.0/bin/grafana .
cp grafana-v11.6.0/conf/sample.ini ./custom.ini
sed -i "s@^;data = /var/lib/grafana@data = $HOME/monitoring/logs/grafana_data@" custom.ini
mkdir -p $HOME/monitoring/logs/grafana_data
# rm -rf grafana-v11.6.0*
rm grafana-enterprise-11.6.0.linux-amd64.tar.gz
# export PATH=$HOME/monitoring/bin:$PATH

#rm -rf /ceph/hpc/data/FRI/dp8949/data/activations_data/
# Run the monitoring services and the deep learning job
cd $HOME/src/
sbatch run_bact_graf.sh

# Check the logs
tail -n 10 monitoring/logs/*

# Check the monitoring services 
curl gn04:9101/metrics # Node Exporter
curl gn04:9401/metrics # DCGM Exporter

# Run locally - to access Prometheus web interface
ssh -L 9090:gn47:9090 -L 3000:gn47:3000 dp8949@login.vega.izum.si
#pkill -f "ssh -N -f -L 9090:gn04:9090"

# Then open http://localhost:9090 in your browser.
# And try the following queries:
#    up
#    node_cpu_seconds_total
#    DCGM_FI_DEV_GPU_UTIL
# Or open http://localhost:9090/targets to see the status of the monitoring services.

# Then open http://localhost:3000 in your browser to access Grafana.
# Login with the default credentials: admin/admin
# Go to Connections > Add new connection > Select Prometheus > Set the URL to http://localhost:9090 > Save & Test
# Upper right corner click + > Import Dashboard > Import 
#    > Paste the dashboard url https://grafana.com/grafana/dashboards/1860-node-exporter-full/ > Load > Select the Prometheus data source > Import
# Then again click + > Import Dashboard > Import
#    > Paste the dashboard url https://grafana.com/grafana/dashboards/12239-nvidia-dcgm-exporter-dashboard/ > Load > Select the Prometheus data source > Import

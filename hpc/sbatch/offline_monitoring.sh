scp dp8949@login.vega.izum.si:/ceph/hpc/home/dp8949/monitoring/logs/grafana_data/* ./Desktop/monlogs/grafana_data/

sudo apt-get install -y adduser libfontconfig1 musl
wget https://dl.grafana.com/enterprise/release/grafana-enterprise_11.6.0_amd64.deb
sudo dpkg -i grafana-enterprise_11.6.0_amd64.deb

sudo systemctl start grafana-server
sudo systemctl enable grafana-server

# try: http://localhost:3000
#    (Default login: admin / admin)

sudo systemctl stop grafana-server
sudo cp /var/lib/grafana/grafana.db /var/lib/grafana/grafana.db.bak
sudo cp ./Desktop/monlogs/grafana_data/grafana.db /var/lib/grafana/grafana.db
sudo chown grafana:grafana /var/lib/grafana/grafana.db

# open localhost:3000 again


scp -r dp8949@login.vega.izum.si:/ceph/hpc/home/dp8949/monitoring/logs/prom_data/* ./Desktop/monlogs/prom_data/
#scp -r dp8949@login.vega.izum.si:/ceph/hpc/home/dp8949/src/data* ./Desktop/monlogs/prom_data/

sudo apt install -y prometheus
sudo systemctl stop prometheus

sudo mv /var/lib/prometheus /var/lib/prometheus_backup
sudo cp -r ~/Desktop/monlogs/prom_data /var/lib/prometheus
sudo chown -R prometheus:prometheus /var/lib/prometheus

sudo mv /etc/prometheus/prometheus.yml /etc/prometheus/prometheus_back.yml
sudo nano /etc/prometheus/prometheus.yml
# global:
#   scrape_interval: 5s
# scrape_configs:
#   - job_name: 'node_exporter'
#     static_configs:
#       - targets: ['localhost:9101']
#   - job_name: 'dcgm_exporter'
#     static_configs:
#       - targets: ['localhost:9401']

sudo nano /etc/systemd/system/prometheus.service
# [Unit]
# Description=Prometheus Service
# After=network.target

# [Service]
# Type=simple
# ExecStart=/usr/bin/prometheus \
#   --config.file=/etc/prometheus/prometheus.yml \
#   --storage.tsdb.path=/var/lib/prometheus/ \
#   --storage.tsdb.retention.time=30d

# [Install]
# WantedBy=multi-user.target

sudo systemctl daemon-reload
sudo systemctl start prometheus
#sudo systemctl restart prometheus
sudo systemctl status prometheus
#sudo journalctl -u prometheus



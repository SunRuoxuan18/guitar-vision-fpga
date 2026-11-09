#!/bin/sh
set -eu

dropin_dir=/etc/systemd/system/fretlens-vision.service.d
backup_dir=/home/xilinx/camera_service_backups
stamp=$(date +%Y%m%d_%H%M%S)

mkdir -p "$backup_dir"
cp "$dropin_dir/camera-yuyv.conf" "$backup_dir/camera-yuyv.conf.$stamp"
install -o root -g root -m 0644 \
  /tmp/fretlens-vision-c922-mjpg.conf \
  "$dropin_dir/camera-c922-mjpg.conf"
rm -f "$dropin_dir/camera-yuyv.conf"
cp /home/xilinx/vision_server.py \
  "/home/xilinx/camera_service_backups/vision_server.py.$stamp"
install -o root -g root -m 0644 \
  /tmp/vision_server.py \
  /home/xilinx/vision_server.py
rm -f /tmp/fretlens-vision-c922-mjpg.conf
rm -f /tmp/vision_server.py

systemctl daemon-reload
systemctl restart fretlens-vision.service
systemctl show fretlens-vision.service -p ActiveState -p SubState -p MainPID
cat "$dropin_dir/camera-c922-mjpg.conf"

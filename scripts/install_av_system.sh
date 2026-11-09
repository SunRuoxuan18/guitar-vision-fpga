#!/bin/sh
# Run as root after extracting the reviewed release on the PYNQ board.
set -eu
cd /home/xilinx/av_system
test -f av_system.bit
test -f av_system.hwh
test -f manifest.json
mkdir -p /home/xilinx/av_system/previous_units
for unit in fretlens-pc-pl.service fretlens-pc-flow.service fretlens-vision-pl.service; do
    if test -f "/etc/systemd/system/$unit"; then
        cp -n "/etc/systemd/system/$unit" previous_units/ || true
    fi
    systemctl stop "$unit" || true
done
# A legacy audio HTTP process is not owned by these units. Refuse to overwrite
# PL until its owner has stopped it; do not kill arbitrary programs by pattern.
if pgrep -f '^.*python[0-9.]* .*audio_music_server.py' >/dev/null; then
    echo 'Stop the existing legacy audio server before installing AVS1' >&2
    exit 1
fi
systemctl stop fretlens-av.target || true
systemctl stop fretlens-av.service || true
gcc -O3 -ffp-contract=off -shared -fPIC -o libaudio_fit.so.new native/audio_fit.c -lm
mv -f libaudio_fit.so.new libaudio_fit.so
if test -f native/neck_region.cpp; then
    g++ -O3 -std=c++11 -Wno-psabi -ffp-contract=off -shared -fPIC -o libneck_region.so.new native/neck_region.cpp
    mv -f libneck_region.so.new libneck_region.so
fi
if test -f native/neck_color.cpp; then
    g++ -O3 -std=c++11 -Wno-psabi -ffp-contract=off -shared -fPIC -o libneck_color.so.new native/neck_color.cpp native/neck_color_ridges.cpp native/neck_color_candidates.cpp $(pkg-config --cflags --libs opencv4)
    mv -f libneck_color.so.new libneck_color.so
fi
if test -f native/hand_q16.c; then
    gcc -O3 -mfpu=neon -ffp-contract=off -pthread -shared -fPIC -o libhand_q16.so.new native/hand_q16.c -lm
    mv -f libhand_q16.so.new libhand_q16.so
fi
if test -f native/fast_jpeg.c; then
    gcc -O3 -shared -fPIC -o libfast_jpeg.so.new native/fast_jpeg.c -ljpeg
    mv -f libfast_jpeg.so.new libfast_jpeg.so
fi
if test -f native/vision_transport.c; then
    gcc -O3 -mfpu=neon -shared -fPIC -o libvision_transport.so.new native/vision_transport.c
    mv -f libvision_transport.so.new libvision_transport.so
fi
for unit in fretlens-av.service fretlens-av.target; do
    install -m 0644 "$unit" "/etc/systemd/system/$unit"
done
systemctl daemon-reload
systemd-analyze verify fretlens-av.target fretlens-av.service
systemctl enable fretlens-av.target
systemctl start fretlens-av.target
systemctl is-active fretlens-av

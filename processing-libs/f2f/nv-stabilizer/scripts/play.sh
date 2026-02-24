#!/bin/bash

# GStreamer pipeline to play 17_11_47 mwir shaking.avi

VIDEO_FILE="scripts/mwir_shaking_2.mp4"

if [ ! -f "$VIDEO_FILE" ]; then
    echo "Error: Video file not found: $VIDEO_FILE"
    exit 1
fi

# GStreamer pipeline with proper error handling
gst-launch-1.0 \
    filesrc location="$VIDEO_FILE" ! \
    decodebin ! \
    videoconvert ! \
    textoverlay text="shaky" valignment=top halignment=left font-desc="Sans, 28" shaded-background=true ! \
    nvvidconv ! \
    nv3dsink &
    
# GStreamer pipeline with proper error handling
gst-launch-1.0 \
    filesrc location="$VIDEO_FILE" ! \
    decodebin ! \
    nvvidconv ! \
    myf2f config-path=/opt/project-system/gst-plugins/plugins/gstmyf2f/config/config-nv.yaml ! \
    nv3dsink

exit $?

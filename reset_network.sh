#!/bin/bash

# Replace 'enp0sX' with your actual Ethernet interface name
INTERFACE="enp7s0"

# Remove the specific IP address
sudo ip addr flush dev $INTERFACE

# Bring the interface down and up to reset
sudo ip link set $INTERFACE down
sudo ip link set $INTERFACE up

echo "Network configuration reset to default."

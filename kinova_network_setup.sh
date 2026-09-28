#!/bin/bash

set -e
# Replace 'enp0sX' with your actual Ethernet interface name
INTERFACE="enp7s0"
IP_ADDRESS="192.168.1.11"
PREFIX=24

sudo ip addr flush dev "$INTERFACE"
sudo ip addr add "$IP_ADDRESS/$PREFIX" dev "$INTERFACE"
sudo ip link set "$INTERFACE" up

echo "Network configuration applied for Kinova Gen3 robot:"
ip -br addr show "$INTERFACE"
ip route get 192.168.1.10

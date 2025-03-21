#!/bin/bash

# Replace 'enp0sX' with your actual Ethernet interface name
INTERFACE="enp7s0"
IP_ADDRESS="192.168.1.11"
SUBNET_MASK="255.255.255.0"

# Configure the IP address and subnet mask
sudo ip addr add $IP_ADDRESS/$SUBNET_MASK dev $INTERFACE

# Bring the interface up
sudo ip link set $INTERFACE up

echo "Network configuration applied for Kinova Gen3 robot."

#!/bin/bash

# Starting script for armassist with watchdog

# Get name of first relay
relay_1=$(/home/tecnalia/usbrelay/usbrelay/usbrelay 2>/dev/null | grep _1 | cut -d"=" -f1)

# Turn relay on
(cd '/home/tecnalia/usbrelay/usbrelay/';./usbrelay $relay_1=1 2>/dev/null)

sleep 1s

# Start watchdog
(cd /home/tecnalia/code/ismore/tubingen/scripts/; ./watchdog_armassist) & 

# Start armassist app
(cd /home/tecnalia/code/armassist/; echo "2" | ./IsMore)

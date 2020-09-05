#!/bin/bash

# Starting script for armassist with watchdog

# Get name of second relay
relay_2=$(/home/tecnalia/usbrelay/usbrelay/usbrelay 2>/dev/null | grep _2 | cut -d"=" -f1)

# Turn relay on
(cd '/home/tecnalia/usbrelay/usbrelay/';./usbrelay $relay_2=1 2>/dev/null)

sleep 1s

# Start watchdog
(cd /home/tecnalia/code/ismore/tubingen/scripts/; ./watchdog_rehand) & 

# Start rehand app
(cd /home/tecnalia/code/rehand/rehand_app/; sudo ./ReHandController_IsMore)

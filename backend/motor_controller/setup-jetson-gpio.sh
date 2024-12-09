#!/bin/sh

sudo pip3 install Jetson.GPIO

sudo groupadd -f -r gpio
sudo usermod -a -G gpio `whoami`

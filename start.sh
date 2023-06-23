#!/bin/bash
docker run -d -v /home/akagi/Documents/Projects/retro-analytics/data/outputs:app/data/outputs --name update-container daily-update


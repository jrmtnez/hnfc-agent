#!/bin/bash

cd /root/vsc/hnfc-dataset-agent/
source /root/miniconda3/bin/activate agent
python -m agent.launchers.hnfc_launch_2_sentence_mgmt
conda deactivate
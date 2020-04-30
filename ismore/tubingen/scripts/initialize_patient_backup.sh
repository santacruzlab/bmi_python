#!/bin/bash

# Run this file to initialize a new patient backup. All folders according to the folder structure definition will be created on the server.

read -p "Patient ID: " patient_id

# Config
base_folder=/mnt/cinstorage/hybrid-bmi-2018/$patient_id/data

# Create folders:
mkdir -p $base_folder/pre1/bmi-paretic/header_files
mkdir -p $base_folder/pre1/bmi-paretic/supp_hdf
mkdir -p $base_folder/pre1/bmi-healthy/header_files
mkdir -p $base_folder/pre1/bmi-healthy/supp_hdf
mkdir -p $base_folder/pre1/eeg-left-right/header_files
mkdir -p $base_folder/pre1/eeg-left-right/supp_hdf
mkdir -p $base_folder/pre1/eeg-resting-state/header_files
mkdir -p $base_folder/pre1/eeg-resting-state/supp_hdf

mkdir -p $base_folder/pre2/bmi-paretic/header_files
mkdir -p $base_folder/pre2/bmi-paretic/supp_hdf
mkdir -p $base_folder/pre2/bmi-healthy/header_files
mkdir -p $base_folder/pre2/bmi-healthy/supp_hdf
mkdir -p $base_folder/pre2/eeg-left-right/header_files
mkdir -p $base_folder/pre2/eeg-left-right/supp_hdf
mkdir -p $base_folder/pre2/eeg-resting-state/header_files
mkdir -p $base_folder/pre2/eeg-resting-state/supp_hdf

mkdir -p $base_folder/pre3/bmi-paretic/header_files
mkdir -p $base_folder/pre3/bmi-paretic/supp_hdf
mkdir -p $base_folder/pre3/bmi-healthy/header_files
mkdir -p $base_folder/pre3/bmi-healthy/supp_hdf
mkdir -p $base_folder/pre3/eeg-left-right/header_files
mkdir -p $base_folder/pre3/eeg-left-right/supp_hdf
mkdir -p $base_folder/pre3/eeg-resting-state/header_files
mkdir -p $base_folder/pre3/eeg-resting-state/supp_hdf

mkdir -p $base_folder/post1/bmi-paretic/header_files
mkdir -p $base_folder/post1/bmi-paretic/supp_hdf
mkdir -p $base_folder/post1/bmi-healthy/header_files
mkdir -p $base_folder/post1/bmi-healthy/supp_hdf
mkdir -p $base_folder/post1/eeg-left-right/header_files
mkdir -p $base_folder/post1/eeg-left-right/supp_hdf
mkdir -p $base_folder/post1/eeg-resting-state/header_files
mkdir -p $base_folder/post1/eeg-resting-state/supp_hdf

mkdir -p $base_folder/training/

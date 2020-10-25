# -*- coding: utf-8 -*-
"""
Created on Sun Oct 11 11:09:34 2020

@author: kathr
"""
import os
import mne
#import numpy as np
#import pandas as pd
#from matplotlib import pyplot as plt
#path="C:\\Users\\kathr\\OneDrive\\Documents\\EEG Specialkursus"
path="C:\\Users\\kathr\\OneDrive\\Documents\\GitHub\\EEG---Special-course"
os.chdir(path)

##########################
## Loading saved epochs ##
##########################


epochs_a_resampled = mne.read_epochs('epochs_a_resampled-epo.fif')
epochs_231_resampled = mne.read_epochs('epochs_231_resampled-epo.fif')
epochs_224a_resampled = mne.read_epochs('epochs_224a_resampled-epo.fif')
epochs_225a_resampled = mne.read_epochs('epochs_225a_resampled-epo.fif')
epochs_233_resampled = mne.read_epochs('epochs_233_resampled-epo.fif')
epochs_224b_resampled = mne.read_epochs('epochs_224b_resampled-epo.fif')
epochs_225b_resampled = mne.read_epochs('epochs_225b_resampled-epo.fif')

#############################
#%% Defining channel names ##
#############################

picks_a = []
picks_b = []
picks_a_eog = []
picks_b_eog = []
channels = epochs_231_resampled.info.ch_names

for i in range(len(channels)):
    if channels[i].startswith('1-A') or channels[i].startswith('1-B'):
        picks_a.append(channels[i])

print(picks_a)

for i in range(len(channels)):
    if channels[i].startswith('2-A') or channels[i].startswith('2-B'):
        picks_b.append(channels[i])

print(picks_b)

# List of channels with eog channels
for i in range(len(channels)):
    if channels[i].startswith('1-A') or channels[i].startswith('1-B') or channels[i].startswith('1-E'):
        picks_a_eog.append(channels[i])

print(picks_a_eog)

for i in range(len(channels)):
    if channels[i].startswith('2-A') or channels[i].startswith('2-B') or channels[i].startswith('2-E'):
        picks_b_eog.append(channels[i])

print(picks_b_eog)

############################
#%% Setting channel names ##
############################

montage = mne.channels.make_standard_montage("biosemi64")
#montage.plot()
new_ch_names = montage.ch_names

for i in range(len(new_ch_names)):
    
    epochs_a_resampled.rename_channels(mapping = {picks_a[i]:new_ch_names[i]})

print(epochs_a_resampled.info.ch_names)

for i in range(len(new_ch_names)):
    
    epochs_231_resampled.rename_channels(mapping = {picks_a[i]:new_ch_names[i]})

print(epochs_231_resampled.info.ch_names)

for i in range(len(new_ch_names)):
    
    epochs_224a_resampled.rename_channels(mapping = {picks_a[i]:new_ch_names[i]})

print(epochs_224a_resampled.info.ch_names)

for i in range(len(new_ch_names)):
    
    epochs_225a_resampled.rename_channels(mapping = {picks_a[i]:new_ch_names[i]})

print(epochs_225a_resampled.info.ch_names)


epochs_231_resampled['Angry1','Angry2'].plot(picks = new_ch_names[20:30],n_epochs = 10)


############################
#%% Removing bad channels ##
############################

epochs_231_resampled.info['bads'].append('PO3')
#epochs_231_resampled.interpolate_bads()

#####################
#%% Re-referencing ##
#####################

epochs_a_resampled.set_eeg_reference('average')
epochs_231_resampled.set_eeg_reference('average')
epochs_224a_resampled.set_eeg_reference('average')
epochs_225a_resampled.set_eeg_reference('average')




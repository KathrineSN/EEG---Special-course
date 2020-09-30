# -*- coding: utf-8 -*-
"""
Created on Sun Sep 13 14:26:15 2020

@author: kathr
"""

import os
import mne
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
path="C:\\Users\\kathr\\OneDrive\\Documents\\EEG Specialkursus"
os.chdir(path)

######################
#%% Loading the data #
######################

mne.set_log_level("WARNING")
raw1 = mne.io.read_raw_bdf("sj0013a_unshared.bdf", preload=True)
raw2 = mne.io.read_raw_bdf("sj0013ab_shared.bdf", preload=True)
raw3 = mne.io.read_raw_bdf("sj0013b_unshared.bdf", preload=True)

#raw1.plot(n_channels=6, scalings={"eeg": 600e-7}, start=100, block = True)

rawa = mne.concatenate_raws([raw1,raw2])
rawb = mne.concatenate_raws([raw2,raw3])
print(rawa.info)

########################
#%% Filtering the data #
########################

#Applying high pass filter at 0.1 Hz and low pass filter at 40 Hz 

f_rawa = rawa.filter(l_freq=0.1, h_freq=40, picks="eeg") 
f_rawb = rawb.filter(l_freq=0.1, h_freq=40, picks="eeg")
#f_rawa.plot(n_channels=32, scalings={"eeg": 600e-8}, start=100, block = True)
#f_rawa.plot(n_channels=32, scalings={"eeg": 600e-8}, start=100, block = True)

mne.viz.plot_raw(f_rawa, n_channels = 100)


print(rawa.info.ch_names)

#######################
#%% Epoching the data #
#######################

## Dividing channels ##
picks_a = []
picks_b = []
picks_a_eog = []
picks_b_eog = []
channels = rawa.info.ch_names

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




## Finding unique events

eventsa = mne.find_events(f_rawa, stim_channel = 'Status')
eventsb = mne.find_events(f_rawb, stim_channel = 'Status')
print(mne.find_events(f_rawa))
print('Number of events:', len(eventsa))
print('Unique event codes:', np.unique(eventsa[:, 2]))

events_231 = eventsa[0:600,:]
events_224_a = eventsa[601:1257,:]
events_225_a = eventsa[1257:1901,:]

events_233 = eventsb[0:656,:]
events_224_b = eventsb[657:1301,:]
events_225_b = eventsb[1302:1901,:]

## Epochs for person A
# Epochs for A unshared condition 

event_dict = {'Angry1': 40, 'Angry2': 41, 'Happy1': 50,'Happy2': 51, 'Neutral1': 60, 'Neutral2':61 }

epochs_231 = mne.Epochs(f_rawa, events_231, event_id = event_dict, tmin=-0.1, tmax=1,
                    baseline=(None, 0), picks = picks_a, preload=True, detrend = 0)

epochs_231.plot(n_epochs=10)

#Epochs for AB shared with feedback

epochs_224_a = mne.Epochs(f_rawa, events_224_a, event_id = event_dict, tmin=-0.1, tmax=1,
                    baseline=(None, 0), preload=True, detrend = 0)

epochs_224_a.plot(n_epochs=10)

#Epochs for AB shared without feedback

epochs_225_a = mne.Epochs(f_rawa, events_225_a, event_id = event_dict, tmin=-0.1, tmax=1,
                    baseline=(None, 0), preload=True, detrend = 0)

epochs_225_a.plot(n_epochs=10)

## Epochs for person B

epochs_233 = mne.Epochs(f_rawa, events_233, event_id = event_dict, tmin=-0.1, tmax=1,
                    baseline=(None, 0), preload=True, detrend = 0)

epochs_233.plot(n_epochs=10)

#Epochs for AB shared with feedback

epochs_224_b = mne.Epochs(f_rawa, events_224_b, event_id = event_dict, tmin=-0.1, tmax=1,
                    baseline=(None, 0), preload=True, detrend = 0)

epochs_224_b.plot(n_epochs=10)

#Epochs for AB shared without feedback

epochs_225_b = mne.Epochs(f_rawa, events_225_b, event_id = event_dict, tmin=-0.1, tmax=1,
                    baseline=(None, 0), preload=True, detrend = 0)

epochs_225_b.plot(n_epochs=10)

###########################
#%% Downsampling the data #
###########################

#Downsampling person A
epochs_231_resampled = epochs_231.copy().resample(256, npad = 'auto')
epochs_224a_resampled = epochs_224_a.copy().resample(256, npad = 'auto')
epochs_225a_resampled = epochs_225_a.copy().resample(256, npad = 'auto')

#Downsampling person B
epochs_233_resampled = epochs_233.copy().resample(256, npad = 'auto')
epochs_224b_resampled = epochs_224_b.copy().resample(256, npad = 'auto')
epochs_225b_resampled = epochs_225_b.copy().resample(256, npad = 'auto')

#Plotting example of downsampling
plt.figure(figsize=(7, 3))
n_samples_to_plot = int(0.5 * epochs_231.info['sfreq'])  # plot 0.5 seconds of data
plt.plot(epochs_231.times[:n_samples_to_plot],
         epochs_231.get_data()[0, 0, :n_samples_to_plot], color='black')

n_samples_to_plot = int(0.5 * epochs_231_resampled.info['sfreq'])
plt.plot(epochs_231_resampled.times[:n_samples_to_plot],
         epochs_231_resampled.get_data()[0, 0, :n_samples_to_plot],
         '-o', color='red')

plt.xlabel('time (s)')
plt.legend(['original', 'downsampled'], loc='best')
plt.title('Effect of downsampling')
mne.viz.tight_layout()


##############################
#%% Identifying bad channels #
##############################

#Power spectral density

## Person A ##
# Three conditions for angry faces 
epochs_231_resampled['Angry1','Angry2'].plot_psd(picks = picks_a, color = 'red')

epochs_224a_resampled['Angry1','Angry2'].plot_psd(picks = picks_a)
epochs_225a_resampled['Angry1','Angry2'].plot_psd(picks = picks_a)

# Three conditions for happy faces 
epochs_231_resampled['Happy1','Happy2'].plot_psd(picks = picks_a)
epochs_224a_resampled['Happy1','Happy2'].plot_psd(picks = picks_a)
epochs_225a_resampled['Happy1','Happy2'].plot_psd(picks = picks_a)

# Three conditions for neutral faces
epochs_231_resampled['Neutral1','Neutral2'].plot_psd(picks = picks_a)
epochs_224a_resampled['Neutral1','Neutral2'].plot_psd(picks = picks_a)
epochs_225a_resampled['Neutral1','Neutral2'].plot_psd(picks = picks_a)


print(epochs_231_resampled.info.ch_names)

# Plotting one channel at a time
for i in range(len(picks_a)):
    print('plotting channel:'+ epochs_231_resampled.info.ch_names[i] )
    epochs_231_resampled['Angry1','Angry2'].plot_psd(picks = picks_a[i])

print(epochs_231_resampled.info.ch_names[20])
print(epochs_231_resampled.info.ch_names[25])

# Channel 1-A21 and 1-A26 seems to be the most noisy channels
epochs_231_resampled['Angry1','Angry2'].plot_psd(picks = '1-A21')
epochs_231_resampled['Angry1','Angry2'].plot_psd(picks = '1-A26')

epochs_231_resampled['Angry1','Angry2'].plot(picks = '1-A21')
epochs_231_resampled['Angry1','Angry2'].plot(picks = picks_a[20:30],n_epochs = 10)


###################################
#%% Setting correct channel names # 
###################################

montage = mne.channels.make_standard_montage("biosemi64")
#montage.plot()
new_ch_names = montage.ch_names

for i in range(len(new_ch_names)):
    
    epochs_231_resampled.rename_channels(mapping = {picks_a[i]:new_ch_names[i]})

print(epochs_231_resampled.info.ch_names)

epochs_231_resampled['Angry1','Angry2'].plot(picks = new_ch_names[20:30],n_epochs = 10)


# Note to self: jeg har kun sat pick_a på epochs fra 231

##########################
#%% Marking bad channels #
##########################

epochs_231_resampled.info['bads'].append('PO3')

# Should I interpolate?

########################
#%% Channel Statistics #
########################

df_231 = epochs_231_resampled['Angry1','Angry2'].to_data_frame(picks = picks_a)

ch_stat = df_231.describe()

###########################################
#%% Artifact rejection using EOG Channels #
###########################################

f_rawa.rename_channels(mapping = {'1-EXG1':'1-EOG'})
f_rawa.rename_channels(mapping = {'1-EXG2':'2-EOG'})
f_rawa.rename_channels(mapping = {'1-EXG3':'3-EOG'})

eog_events = mne.preprocessing.find_eog_events(f_rawa, ch_name = '1-EXG1')
eog_events = mne.preprocessing.find_eog_events(f_rawa, ch_name = '1-EXG2')
eog_events = mne.preprocessing.find_eog_events(f_rawa, ch_name = '1-EXG3')

eog_epochs = mne.preprocessing.create_eog_epochs(f_rawa, ch_name='1-EXG1', event_id=998)
eog_epochs = mne.preprocessing.create_eog_epochs(f_rawa, ch_name='1-EXG2', event_id=998)
eog_epochs = mne.preprocessing.create_eog_epochs(f_rawa, ch_name='1-EXG3', event_id=998)

print(f_rawa.info.ch_names)

eog = f_rawa.pick_types(eog=True)
print(eog)


print(mne.find_events(f_rawa))

print(eog_epochs)

print(eog_events)
print(f_rawa.info.ch_names)





# Testing things






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

#%% Epochs for person A and B
# Epochs for A unshared condition AND removing eog artifacts with rejection_criteria

#Defining rejection criterias

f_rawa.set_channel_types({'1-A1':'ecg'})
f_rawa.set_channel_types({'1-EXG1':'eog'})
f_rawa.get_channel_types()

reject_criteria = dict(eeg = 80e-6,
                       ecg = 60e-6,
                       eog = 90e-6)
                    

event_dict = {'Angry1': 40, 'Angry2': 41, 'Happy1': 50,'Happy2': 51, 'Neutral1': 60, 'Neutral2':61 }

# Epochs for subject a all conditions
epochs_a = mne.Epochs(f_rawa, eventsa, event_id = event_dict, tmin=-0.1, tmax=1,
                    baseline=(None, 0), picks = picks_a_eog, preload=True, detrend = 0, reject = reject_criteria)

epochs_a.save('epochs_a-epo.fif', overwrite = True)

epochs_a.plot_drop_log()
print(epochs_a.drop_log)

# Epochs for separate conditions for subject a
epochs_231 = mne.Epochs(f_rawa, events_231, event_id = event_dict, tmin=-0.1, tmax=1,
                    baseline=(None, 0), picks = picks_a_eog, preload=True, detrend = 0, reject = reject_criteria)

epochs_231.plot_drop_log()
print(epochs_231.drop_log)

epochs_231.save('epochs_231-epo.fif', overwrite = True)

epochs_231.plot(n_epochs=10)

#Epochs for AB shared with feedback

epochs_224_a = mne.Epochs(f_rawa, events_224_a, event_id = event_dict, tmin=-0.1, tmax=1,
                    baseline=(None, 0), preload=True, detrend = 0, reject = reject_criteria)

epochs_224_a.save('epochs_224_a-epo.fif', overwrite = True)

epochs_224_a.plot(n_epochs=10)

#Epochs for AB shared without feedback

epochs_225_a = mne.Epochs(f_rawa, events_225_a, event_id = event_dict, tmin=-0.1, tmax=1,
                    baseline=(None, 0), preload=True, detrend = 0, reject = reject_criteria)

epochs_225_a.save('epochs_225_a-epo.fif', overwrite = True)

epochs_225_a.plot(n_epochs=10)

## Epochs for person B

epochs_233 = mne.Epochs(f_rawa, events_233, event_id = event_dict, tmin=-0.1, tmax=1,
                    baseline=(None, 0), preload=True, detrend = 0, reject = reject_criteria)

epochs_233.save('epochs_233-epo.fif', overwrite = True)

epochs_233.plot(n_epochs=10)

#Epochs for AB shared with feedback

epochs_224_b = mne.Epochs(f_rawa, events_224_b, event_id = event_dict, tmin=-0.1, tmax=1,
                    baseline=(None, 0), preload=True, detrend = 0, reject = reject_criteria)

epochs_224_b.save('epochs_224_b-epo.fif', overwrite = True)

epochs_224_b.plot(n_epochs=10)

#Epochs for AB shared without feedback

epochs_225_b = mne.Epochs(f_rawa, events_225_b, event_id = event_dict, tmin=-0.1, tmax=1,
                    baseline=(None, 0), preload=True, detrend = 0, reject = reject_criteria)
epochs_225_b.save('epochs_225_b-epo.fif', overwrite = True)

epochs_225_b.plot(n_epochs=10)


####################################
#%% Reading in epochs as fif files #
####################################
# To be used after epochs have been loaded the first time

#epochs_a = mne.read_epochs('epochs_a-epo.fif')
#epochs_231 = mne.read_epochs('epochs_231-epo.fif')
#epochs_224_a = mne.read_epochs('epochs_224_a-epo.fif')
#epochs_225_a = mne.read_epochs('epochs_225_a-epo.fif')
#epochs_233 = mne.read_epochs('epochs_233-epo.fif')
#epochs_224_b = mne.read_epochs('epochs_224_b-epo.fif')
#epochs_225_b = mne.read_epochs('epochs_225_b-epo.fif')

###########################
#%% Downsampling the data #
###########################

#Downsampling person A
epochs_a_resampled = epochs_a.copy().resample(256, npad = 'auto')
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

#Saving resampled epochs
epochs_a_resampled.save('epochs_a_resampled-epo.fif', overwrite = True)
epochs_231_resampled.save('epochs_231_resampled-epo.fif', overwrite = True)
epochs_224a_resampled.save('epochs_224a_resampled-epo.fif', overwrite = True)
epochs_225a_resampled.save('epochs_225a_resampled-epo.fif', overwrite = True)

epochs_233_resampled.save('epochs_233_resampled-epo.fif', overwrite = True)
epochs_224a_resampled.save('epochs_224a_resampled-epo.fif', overwrite = True)
epochs_225b_resampled.save('epochs_225b_resampled-epo.fif', overwrite = True)

##############################
# Reading downsampled epochs #
##############################
# To be used after epochs have been loaded the first time

#epochs_231_resampled = mne.read_epochs('epochs_231_resampled-epo.fif')
#epochs_224_a_resampled = mne.read_epochs('epochs_224_a_resampled-epo.fif')
#epochs_225_a_resampled = mne.read_epochs('epochs_225_a_resampled-epo.fif')
#epochs_233_resampled = mne.read_epochs('epochs_233_resampled-epo.fif')
#epochs_224_b_resampled = mne.read_epochs('epochs_224_b_resampled-epo.fif')
#epochs_225_b_resampled = mne.read_epochs('epochs_225_b_resampled-epo.fif')

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
# not working but have worked previously...
#for i in range(len(picks_a)):
#    print('plotting channel:' + epochs_231_resampled.info.ch_names[i])
#    epochs_231_resampled['Angry1','Angry2'].plot_psd(picks = picks_a[i])

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


# Note to self: jeg har kun sat pick_a på epochs fra 231

########################
#%% Channel Statistics #
########################

df_231 = epochs_231_resampled['Angry1','Angry2'].to_data_frame(picks = new_ch_names)

ch_stat = df_231.describe()

###########################################
#%% Dropping bad channels & interpolating #
###########################################

#epochs_231_resampled.info['bad'] = ['PO3']

epochs_a_resampled.info['bads'].append('PO3')
epochs_231_resampled.info['bads'].append('PO3')
epochs_224a_resampled.info['bads'].append('PO3')
epochs_225a_resampled.info['bads'].append('PO3')
#epochs_231_resampled.interpolate_bads()
print(epochs_231_resampled.info.ch_names)

print(epochs_231_resampled.info['bads'])

#####################
#%% Re-referencing  #
#####################


#print(epochs_224a_resampled.info['dig'])

epochs_a_resampled.set_eeg_reference('average')
epochs_231_resampled.set_eeg_reference('average')
epochs_224a_resampled.set_eeg_reference('average')
epochs_225a_resampled.set_eeg_reference('average')

#Creating average plot

print(new_ch_names)

epochs_231_resampled.set_channel_types({'Fp1':'eeg'})
evoked_231_Angry = epochs_231_resampled['Angry1','Angry2'].average(picks = new_ch_names)
evoked_231_Happy = epochs_231_resampled['Happy1','Happy2'].average(picks = new_ch_names)
evoked_231_Neutral = epochs_231_resampled['Neutral1','Neutral2'].average(picks = new_ch_names)
evoked_231_Angry.plot()
evoked_231_Happy.plot()
evoked_231_Neutral.plot()

# Testing things






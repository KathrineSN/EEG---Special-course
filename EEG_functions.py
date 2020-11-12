# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 09:13:05 2020

@author: kathr
"""

import os
import mne
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
#path="C:\\Users\\kathr\\OneDrive\\Documents\\EEG Specialkursus"
path="C:\\Users\\kathr\\OneDrive\\Documents\\GitHub\\EEG---Special-course"
os.chdir(path)
#from preprocessing.py import *
#from re_load import *

def loading_raw(file1, file2, file3, pair):
    
    # loading the raw file
    mne.set_log_level("WARNING")
    raw1 = mne.io.read_raw_bdf(file1, preload=True)
    raw2 = mne.io.read_raw_bdf(file2, preload=True)
    raw3 = mne.io.read_raw_bdf(file3, preload=True)
    
    if file1 == "sj0016a_unshared.bdf":
        raw1.drop_channels(['3-A1', '3-A2', '3-A3', '3-A4', '3-A5', '3-A6', '3-A7', '3-A8', '3-A9', '3-A10', '3-A11', '3-A12','3-A13', '3-A14', '3-A15', '3-A16', '3-A17', '3-A18', '3-A19', '3-A20', '3-A21', '3-A22', '3-A23', '3-A24', '3-A25', '3-A26', '3-A27', '3-A28', '3-A29', '3-A30', '3-A31', '3-A32', '3-B1', '3-B2', '3-B3', '3-B4', '3-B5', '3-B6', '3-B7', '3-B8', '3-B9', '3-B10', '3-B11', '3-B12', '3-B13', '3-B14', '3-B15', '3-B16', '3-B17', '3-B18', '3-B19', '3-B20', '3-B21', '3-B22', '3-B23', '3-B24', '3-B25', '3-B26', '3-B27', '3-B28', '3-B29', '3-B30', '3-B31', '3-B32', '3-EXG1', '3-EXG2', '3-EXG3', '3-EXG4', '3-EXG5', '3-EXG6', '3-EXG7', '3-EXG8', '4-A1', '4-A2', '4-A3', '4-A4', '4-A5', '4-A6', '4-A7', '4-A8', '4-A9', '4-A10', '4-A11', '4-A12', '4-A13', '4-A14', '4-A15', '4-A16', '4-A17', '4-A18', '4-A19', '4-A20', '4-A21', '4-A22', '4-A23', '4-A24','4-A25', '4-A26', '4-A27','4-A28','4-A29', '4-A30', '4-A31','4-A32', '4-B1','4-B2', '4-B3','4-B4', '4-B5', '4-B6','4-B7','4-B8','4-B9','4-B10','4-B11','4-B12','4-B13','4-B14', '4-B15', '4-B16','4-B17','4-B18','4-B19','4-B20','4-B21','4-B22','4-B23','4-B24','4-B25','4-B26','4-B27','4-B28','4-B29','4-B30','4-B31','4-B32', '4-EXG1','4-EXG2','4-EXG3','4-EXG4','4-EXG5','4-EXG6','4-EXG7', '4-EXG8'])  
    
     #raw1.plot(n_channels=6, scalings={"eeg": 600e-7}, start=100, block = True)
    raw2.plot(n_channels=6, scalings={"eeg": 600e-7}, start=100, block = True)
    raw2.plot(n_channels=6, scalings={"eeg": 600e-7}, start=100, block = True)
    
    rawa = mne.concatenate_raws([raw1,raw2])
    rawb = mne.concatenate_raws([raw2,raw3])
    print(rawa.info)
    
    #filtering
    
    f_rawa = rawa.filter(l_freq=0.1, h_freq=40, picks="eeg") 
    f_rawb = rawb.filter(l_freq=0.1, h_freq=40, picks="eeg")
    
    #diving channels
        
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
    
    #Finding events
    
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
    
    #Defining rejection criteria
    
    f_rawa.set_channel_types({'1-A1':'ecg'})
    f_rawa.set_channel_types({'1-EXG1':'eog'})
    f_rawb.set_channel_types({'2-A1':'ecg'})
    f_rawb.set_channel_types({'2-EXG1':'eog'})
    f_rawa.get_channel_types()
    
    reject_criteria = dict(eeg = 80e-6,
                           ecg = 60e-6,
                           eog = 90e-6)
                        
    
    event_dict = {'Angry1': 40, 'Angry2': 41, 'Happy1': 50,'Happy2': 51, 'Neutral1': 60, 'Neutral2':61 }
    
    # Epochs for subject a all conditions
    
    
    epochs_a = mne.Epochs(f_rawa, eventsa, event_id = event_dict, tmin=-0.5, tmax=1,
                        baseline=(None, 0), picks = picks_a_eog, preload=True, detrend = 0, reject = reject_criteria)
    
    
    epochs_b = mne.Epochs(f_rawb, eventsb, event_id = event_dict, tmin=-0.5, tmax=1,
                        baseline=(None, 0), picks = picks_b_eog, preload=True, detrend = 0, reject = reject_criteria)
    
    
    epochs_a.plot_drop_log()
    print(epochs_a.drop_log)
    
    # Epochs for separate conditions for subject a
    epochs_231 = mne.Epochs(f_rawa, events_231, event_id = event_dict, tmin=-0.5, tmax=1,
                        baseline=(None, 0), picks = picks_a_eog, preload=True, detrend = 0, reject = reject_criteria)
    
    epochs_231.plot_drop_log()
    print(epochs_231.drop_log)
    
    
    epochs_231.plot(n_epochs=10)
    
    #Epochs for AB shared with feedback
    
    epochs_224_a = mne.Epochs(f_rawa, events_224_a, event_id = event_dict, tmin=-0.5, tmax=1,
                        baseline=(None, 0), picks = picks_a_eog, preload=True, detrend = 0, reject = reject_criteria)
    
    
    epochs_224_a.plot(n_epochs=10)
    
    #Epochs for AB shared without feedback
    
    epochs_225_a = mne.Epochs(f_rawa, events_225_a, event_id = event_dict, tmin=-0.5, tmax=1,
                        baseline=(None, 0), picks = picks_a_eog, preload=True, detrend = 0, reject = reject_criteria)
    
    
    epochs_225_a.plot(n_epochs=10)
    
    ## Epochs for person B
    
    epochs_233 = mne.Epochs(f_rawb, events_233, event_id = event_dict, tmin=-0.5, tmax=1,
                        baseline=(None, 0), picks = picks_b_eog, preload=True, detrend = 0, reject = reject_criteria)
    
    
    epochs_233.plot(n_epochs=10)
    
    #Epochs for AB shared with feedback
    
    epochs_224_b = mne.Epochs(f_rawb, events_224_b, event_id = event_dict, tmin=-0.5, tmax=1,
                        baseline=(None, 0), picks = picks_b_eog, preload=True, detrend = 0, reject = reject_criteria)
    
    
    epochs_224_b.plot(n_epochs=10)
    
    #Epochs for AB shared without feedback
    
    epochs_225_b = mne.Epochs(f_rawb, events_225_b, event_id = event_dict, tmin=-0.5, tmax=1,
                        baseline=(None, 0), picks = picks_b_eog, preload=True, detrend = 0, reject = reject_criteria)
    
    epochs_225_b.plot(n_epochs=10)
    
    #Downsampling
    
    #Downsampling person A
    epochs_a_resampled = epochs_a.copy().resample(256, npad = 'auto')
    epochs_231_resampled = epochs_231.copy().resample(256, npad = 'auto')
    epochs_224a_resampled = epochs_224_a.copy().resample(256, npad = 'auto')
    epochs_225a_resampled = epochs_225_a.copy().resample(256, npad = 'auto')
    
    #Downsampling person B
    epochs_b_resampled = epochs_b.copy().resample(256, npad = 'auto')
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
    
    montage = mne.channels.make_standard_montage("biosemi64")
    new_ch_names = montage.ch_names
    print(new_ch_names)
    print(picks_b)
    print(picks_a)

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
    
    for i in range(len(new_ch_names)):
        
        epochs_b_resampled.rename_channels(mapping = {picks_b[i]:new_ch_names[i]})
    
    print(epochs_b_resampled.info.ch_names)
    
    for i in range(len(new_ch_names)):
        
        epochs_233_resampled.rename_channels(mapping = {picks_b[i]:new_ch_names[i]})
    
    print(epochs_233_resampled.info.ch_names)
    
    for i in range(len(new_ch_names)):
        
        epochs_224b_resampled.rename_channels(mapping = {picks_b[i]:new_ch_names[i]})
    
    print(epochs_224b_resampled.info.ch_names)
    
    for i in range(len(new_ch_names)):
        
        epochs_225b_resampled.rename_channels(mapping = {picks_b[i]:new_ch_names[i]})
    
    print(epochs_225b_resampled.info.ch_names)
    
    epochs_a_resampled.info['bads'].append('PO3')
    epochs_231_resampled.info['bads'].append('PO3')
    epochs_224a_resampled.info['bads'].append('PO3')
    epochs_225a_resampled.info['bads'].append('PO3')
    epochs_b_resampled.info['bads'].append('PO3')
    epochs_233_resampled.info['bads'].append('PO3')
    epochs_224b_resampled.info['bads'].append('PO3')
    epochs_225b_resampled.info['bads'].append('PO3')
    #epochs_231_resampled.interpolate_bads()
    
    
    
    # Re-referencing #
    epochs_a_resampled.set_eeg_reference('average')
    epochs_231_resampled.set_eeg_reference('average')
    epochs_224a_resampled.set_eeg_reference('average')
    epochs_225a_resampled.set_eeg_reference('average')
    epochs_b_resampled.set_eeg_reference('average')
    epochs_233_resampled.set_eeg_reference('average')
    epochs_224b_resampled.set_eeg_reference('average')
    epochs_225b_resampled.set_eeg_reference('average')
    
    
    #Saving resampled epochs
    epochs_a_resampled.save(pair + '_epochs_a_resampled-epo.fif', overwrite = True)
    epochs_231_resampled.save(pair + '_epochs_231_resampled-epo.fif', overwrite = True)
    epochs_224a_resampled.save(pair + '_epochs_224a_resampled-epo.fif', overwrite = True)
    epochs_225a_resampled.save(pair + '_epochs_225a_resampled-epo.fif', overwrite = True)
    
    epochs_b_resampled.save(pair + '_epochs_b_resampled-epo.fif', overwrite = True)
    epochs_233_resampled.save(pair + '_epochs_233_resampled-epo.fif', overwrite = True)
    epochs_224a_resampled.save(pair + '_epochs_224b_resampled-epo.fif', overwrite = True)
    epochs_225b_resampled.save(pair + '_epochs_225b_resampled-epo.fif', overwrite = True)

    return


def create_evoked(epochs, component, cond, pair, plot=0):
    
    chan = []
    
    #Creating evoked object 
    evoked = epochs.average()
    print(evoked.info['ch_names'])
    
    # Defining channels
    if component == 'N170':
        chan = ['PO7','PO8','P7','P8','P9','P10']
    
    if component == 'LLP':
        chan = ['CP1', 'CP3', 'P3', 'CP2', 'CP4', 'P4']
        
    idx1 = mne.pick_channels(evoked.info['ch_names'], chan)
    idx_dict = dict(idx = idx1)
    evoked_combined = mne.channels.combine_channels(evoked, idx_dict, method = 'mean')
    #evoked_231_N170.plot()
    title = 'Average of: ' + component + 'cond:' + cond
    if plot:
        evoked_combined.plot(titles = dict(eeg = title))
    evoked_combined.save(str(pair) + '_evoked_' + cond + '_' + component + '_combined-ave.fif')
    return evoked_combined

def save_evoked(pairs):
    
    for i in pairs:
        p_a = mne.read_epochs(str(i) + '_epochs_a_resampled-epo.fif')
        p_231 = mne.read_epochs(str(i) + '_epochs_231_resampled-epo.fif')
        p_224a = mne.read_epochs(str(i) + '_epochs_224a_resampled-epo.fif')
        p_225a = mne.read_epochs(str(i) + '_epochs_225a_resampled-epo.fif')
        p_b = mne.read_epochs(str(i) + '_epochs_b_resampled-epo.fif')
        p_233 = mne.read_epochs(str(i) + '_epochs_233_resampled-epo.fif')
        p_224b = mne.read_epochs(str(i) + '_epochs_224b_resampled-epo.fif')
        p_225b = mne.read_epochs(str(i) + '_epochs_224b_resampled-epo.fif')
        
        # Social cond. for person A
        evoked_a_angryN170 = create_evoked(p_a['Angry1','Angry2'], 'N170', 'Angry', str(i))
        evoked_a_angryLPP = create_evoked(p_a['Angry1','Angry2'], 'LPP', 'Angry', str(i))
        evoked_a_happyN170 = create_evoked(p_a['Happy1','Happy2'], 'N170', 'Happy', str(i))
        evoked_a_happyLPP = create_evoked(p_a['Happy1','Happy2'], 'LPP', 'Happy', str(i))
        evoked_a_neutralN170 = create_evoked(p_a['Neutral1','Neutral2'], 'N170', 'Neutral', str(i))
        evoked_a_neutralLPP = create_evoked(p_a['Neutral1','Neutral2'], 'LPP', 'Neutral', str(i))
        
        evoked_231 = create_evoked(p_231, 'N170', '231', str(i))
        evoked_231 = create_evoked(p_231, 'LPP', '231', str(i))
        evoked_224a = create_evoked(p_224a, 'N170', '224a', str(i))
        evoked_224a = create_evoked(p_224a, 'LPP', '224a', str(i))
        evoked_225a = create_evoked(p_225a, 'N170', '225a', str(i))
        evoked_225a = create_evoked(p_225a, 'LPP', '225a', str(i))
        
        # Social cond. for person 
        evoked_b_angryN170 = create_evoked(p_b['Angry1','Angry2'], 'N170', 'Angry', str(i))
        evoked_b_angryLPP = create_evoked(p_b['Angry1','Angry2'], 'LPP', 'Angry', str(i))
        evoked_b_happyN170 = create_evoked(p_b['Happy1','Happy2'], 'N170', 'Happy', str(i))
        evoked_b_happyLPP = create_evoked(p_b['Happy1','Happy2'], 'LPP', 'Happy', str(i))
        evoked_b_neutralN170 = create_evoked(p_b['Neutral1','Neutral2'], 'N170', 'Neutral', str(i))
        evoked_b_neutralLPP = create_evoked(p_b['Neutral1','Neutral2'], 'LPP', 'Neutral', str(i))
        
        evoked_233 = create_evoked(p_233, 'N170', '233', str(i))
        evoked_233 = create_evoked(p_233, 'LPP', '233', str(i))
        evoked_224b = create_evoked(p_224b, 'N170', '224b', str(i))
        evoked_224b = create_evoked(p_224b, 'LPP', '224b', str(i))
        evoked_225b = create_evoked(p_225b, 'N170', '225b', str(i))
        evoked_225b = create_evoked(p_225b, 'LPP', '225b', str(i))
    
    return 

save_evoked([13,14,15,16,17,18])
#save_evoked([13])
      
def grand_average(pairs, cond, component, plot = 0):
    
    evokeds = []
    
    for i in pairs:
        
        for root, dirs, files in os.walk(path):
            
            for f in files:
                
                # If statement to merge 233 and 231
                if cond == '233' or cond == '231':
        
                    if f.startswith(str(i)+ '_evoked_') and f.endswith('233_' + component + '_combined-ave.fif'):
                        
                        evokeds.append(mne.read_evokeds(f))
                    
                    if f.startswith(str(i)+ '_evoked_') and f.endswith('231_' + component + '_combined-ave.fif'):
                        
                        evokeds.append(mne.read_evokeds(f))
                
                elif f.startswith(str(i)+ '_evoked_') and f.endswith(cond + '_' + component + '_combined-ave.fif'):
                    
                    evokeds.append(mne.read_evokeds(f))
                    
                elif f.startswith(str(i)+ '_evoked_') and f.endswith(cond + 'a_' + component + '_combined-ave.fif'):
                    evokeds.append(mne.read_evokeds(f))
                
                elif f.startswith(str(i)+ '_evoked_') and f.endswith(cond + 'b_' + component + '_combined-ave.fif'):
                    evokeds.append(mne.read_evokeds(f))
    
    for i in range(len(evokeds)):
        evokeds[i] = evokeds[i][0]
    
    grand_average = mne.grand_average(evokeds)
    if plot:
        grand_average.plot()
    
   
    return grand_average


def create_subject_list(epoch_file, sub, social, component):
    
    epoch = mne.read_epochs(epoch_file)
    
    if component == 'N170' or component == 'EPN':
        chan = ['PO7','PO8','P7','P8','P9','P10']
        
    if component == 'Early LPP' or component == 'Late LPP':
        chan = ['CP1', 'CP3', 'P3', 'CP2', 'CP4', 'P4']
    
    df = epoch.to_data_frame(picks = chan)
    
    if component == 'N170':
        df_temp = df.loc[df['time']>170]
        df_temp1 = df_temp.loc[df['time']<230]
        col = df_temp1.loc[: , "PO7":"P10"]
        df_temp1[component] = col.mean(axis=1)
    
    if component == 'EPN':
        df_temp = df.loc[df['time']>230]
        df_temp1 = df_temp.loc[df['time']<350]
        col = df_temp1.loc[: , "PO7":"P10"]
        df_temp1[component] = col.mean(axis=1)
    
    if component == 'Early LPP':
        df_temp = df.loc[df['time']>400]
        df_temp1 = df_temp.loc[df['time']<700]
        col = df_temp1.loc[: , "CP1":"P4"]
        df_temp1[component] = col.mean(axis=1)
        
    if component == 'Late LPP':
        df_temp = df.loc[df['time']>700]
        df_temp1 = df_temp.loc[df['time']<1000]
        col = df_temp1.loc[: , "CP1":"P4"]
        df_temp1[component] = col.mean(axis=1)
    
    data = []
    angry = ['Angry1','Angry2']
    happy = ['Happy1','Happy2']
    neutral = ['Neutral1','Neutral2']
    df_angry = df_temp1.loc[df['condition'].isin(angry)]
    df_happy = df_temp1.loc[df['condition'].isin(happy)]
    df_neutral = df_temp1.loc[df['condition'].isin(neutral)]
    
    data.append([sub, social,'Angry',df_angry[component].mean()])
    data.append([sub, social,'Happy',df_happy[component].mean()])
    data.append([sub, social,'Neutral',df_neutral[component].mean()])
    
    return data

def create_dataframe(pairs, component):
    
    epoch_files = []
    
    data = []
    
    for i in pairs:
        
        for root, dirs, files in os.walk(path):
            
            for f in files:
                
                if f.startswith(str(i) + '_epochs_'):
                    
                    epoch_files.append(f)
        
    
        for epoch in epoch_files:
            name = str(epoch)
            if name.endswith('224a_resampled-epo.fif'):
                data.append(create_subject_list(epoch, str(i) + 'a', '224', component))
                
            if name.endswith('224b_resampled-epo.fif'):
                data.append(create_subject_list(epoch, str(i) + 'b', '224', component))  
            
            if name.endswith('225a_resampled-epo.fif'):
                data.append(create_subject_list(epoch, str(i) + 'a', '225', component))
            
            if name.endswith('225b_resampled-epo.fif'):
                data.append(create_subject_list(epoch, str(i) + 'b', '225', component))
            
            if name.endswith('231_resampled-epo.fif'):
                data.append(create_subject_list(epoch, str(i) + 'a', '231', component))
            
            if name.endswith('233_resampled-epo.fif'):
                data.append(create_subject_list(epoch, str(i) + 'b', '233', component))
    
    print(data)
    
    new_data = []
    
    for i in range(len(data)):
        for j in range(3):
            new_data.append(data[i][j])
    
    df = pd.DataFrame(new_data, columns = ['Subject', 'Social condition', 'Emotional condition', 'Avg. amplitude'])
    
    return df
        
def grand_average_with_CI(overall_cond, component, plot = 1):
    
    
    pairs = [13,14,15,16,17,18]
    
    if overall_cond == 'Social':
        
        dfs_231 = []
        dfs_224 = []
        dfs_225 = []
        
        for i in pairs:
            
            for root, dirs, files in os.walk(path):
                
                for f in files:
                    
                    if f.startswith(str(i)+ '_evoked_231') and f.endswith(component + '_combined-ave.fif'):
                    
                        evoked_231 = mne.read_evokeds(f)
                    
                        dfs_231.append(evoked_231[0].to_data_frame())
                        
                    if f.startswith(str(i)+ '_evoked_233') and f.endswith(component + '_combined-ave.fif'):
                    
                        evoked_233 = mne.read_evokeds(f)
                    
                        dfs_231.append(evoked_233[0].to_data_frame())
                    
                    if f.startswith(str(i)+ '_evoked_224a') and f.endswith(component + '_combined-ave.fif'):
                    
                        evoked_224a = mne.read_evokeds(f)
                    
                        dfs_224.append(evoked_224a[0].to_data_frame())
                    
                    if f.startswith(str(i)+ '_evoked_224b') and f.endswith(component + '_combined-ave.fif'):
                    
                        evoked_224b = mne.read_evokeds(f)
                    
                        dfs_224.append(evoked_224b[0].to_data_frame())
                    
                    if f.startswith(str(i)+ '_evoked_225a') and f.endswith(component + '_combined-ave.fif'):
                    
                        evoked_225a = mne.read_evokeds(f)
                    
                        dfs_225.append(evoked_225a[0].to_data_frame())
                    
                    if f.startswith(str(i)+ '_evoked_225b') and f.endswith(component + '_combined-ave.fif'):
                    
                        evoked_225b = mne.read_evokeds(f)
                    
                        dfs_225.append(evoked_225b[0].to_data_frame())
        
    
        df_social_erp_231 = pd.concat(dfs_231, axis = 0, sort = False)
        l_231 = df_social_erp_231.shape
        l_231 = l_231[0]
        social_cond = ['Unshared'] * l_231
        df_social_erp_231['Social Condition'] = social_cond
        

        df_social_erp_224 = pd.concat(dfs_224, axis = 0, sort = False) 
        l_224 = df_social_erp_224.shape
        l_224 = l_224[0]
        social_cond = ['Shared with feedback'] * l_224
        df_social_erp_224['Social Condition'] = social_cond
        
        df_social_erp_225 = pd.concat(dfs_225, axis = 0, sort = False) 
        l_225 = df_social_erp_225.shape
        l_225 = l_225[0]
        social_cond = ['Shared without feedback'] * l_225
        df_social_erp_225['Social Condition'] = social_cond
        
        full_df = pd.concat([df_social_erp_231, df_social_erp_224, df_social_erp_225], axis = 0, sort = False)
        full_df = full_df.rename(columns={'idx':'Microvolt'})
        full_df = full_df.rename(columns={'time':'Time'})

        
        if plot:
            plt.figure()
            sns.lineplot(x = 'Time', y = 'Microvolt', hue = 'Social Condition', data = full_df)
            #sns.lineplot(x = 'Time', y = 'Microvolt', data = full_df)
            
    if overall_cond == 'Emotional':
        
        dfs_happy = []
        dfs_angry = []
        dfs_neutral = []
        
        for i in pairs:
            
            for root, dirs, files in os.walk(path):
                
                for f in files:
                    
                    if f.startswith(str(i)+ '_evoked_Happy') and f.endswith(component + '_combined-ave.fif'):
                    
                        evoked_happy = mne.read_evokeds(f)
                    
                        dfs_happy.append(evoked_happy[0].to_data_frame())
                        
                    if f.startswith(str(i)+ '_evoked_Angry') and f.endswith(component + '_combined-ave.fif'):
                    
                        evoked_angry = mne.read_evokeds(f)
                    
                        dfs_angry.append(evoked_angry[0].to_data_frame())
                    
                    if f.startswith(str(i)+ '_evoked_Neutral') and f.endswith(component + '_combined-ave.fif'):
                    
                        evoked_neutral = mne.read_evokeds(f)
                    
                        dfs_neutral.append(evoked_neutral[0].to_data_frame())
                            
        
        df_emotional_erp_happy = pd.concat(dfs_happy, axis = 0, sort = False)
        l_happy = df_emotional_erp_happy.shape
        l_happy = l_happy[0]
        emotional_cond = ['Happy'] * l_happy
        df_emotional_erp_happy['Emotion'] = emotional_cond

        df_emotional_erp_angry = pd.concat(dfs_angry, axis = 0, sort = False) 
        l_angry = df_emotional_erp_angry.shape
        l_angry = l_angry[0]
        emotional_cond = ['Angry'] * l_angry
        df_emotional_erp_angry['Emotion'] = emotional_cond
        
        df_emotional_erp_neutral = pd.concat(dfs_neutral, axis = 0, sort = False) 
        l_neutral = df_emotional_erp_neutral.shape
        l_neutral = l_neutral[0]
        emotional_cond = ['Neutral'] * l_neutral
        df_emotional_erp_neutral['Emotion'] = emotional_cond

        full_df = pd.concat([df_emotional_erp_happy, df_emotional_erp_angry, df_emotional_erp_neutral], axis = 0, sort = False)
        full_df = full_df.rename(columns={'idx':'Microvolt'})
        full_df = full_df.rename(columns={'time':'Time'})

        
        if plot:
            plt.figure()
            sns.lineplot(x = 'Time', y = 'Microvolt', hue = 'Emotion', data = full_df)
            #sns.lineplot(x = 'Time', y = 'Microvolt', data = full_df)
    
    return full_df

#grand_average_with_CI('Social', 'N170')
#grand_average_with_CI('Emotional', 'N170')    
    

######## Denne del skal flyttes til serparat fil

#grand_average([13,14], '225', 'N170', plot = 1)
#grand_average([13,14,15],'Happy','N170', plot=1)
#grand_average([13,14], 'Neutral', 'N170', plot = 1)
#grand_average([13,14,15,17], '233', 'N170', plot = 1)
#grand_average([13,14], '225', 'N170', plot = 1)
#grand_average([13], '233', 'N170', plot = 1)
#grand_average([13,14,15,17,18], 'Angry', 'N170', plot = 1)
#ga_224 = grand_average([13,14,15,16,17,18], '224', 'N170', plot = 1)
#ga_225 = grand_average([13,14,15,16,17,18], '225', 'N170', plot = 1)
#ga_231 = grand_average([13,14,15,16,17,18], '231', 'N170', plot = 1)

#ga_Angry = grand_average([13,14,15,16,17,18], 'Angry', 'N170', plot = 1)
#ga_Happy = grand_average([13,14,15,16,17,18], 'Happy', 'N170', plot = 1)
#ga_Neutral = grand_average([13,14,15,16,17,18], 'Neutral', 'N170', plot = 1)


#mne.viz.plot_compare_evokeds(dict(shared_with_feedback = ga_224, shared_without_feedback = ga_225, unshared = ga_231))
#mne.viz.plot_compare_evokeds(dict(Angry = ga_Angry, Happy = ga_Happy, Neutral = ga_Neutral))



#create_evoked(epochs_231_resampled, component = 'N170', cond = '231', plot=1)

#loading_raw("sj0013a_unshared.bdf", "sj0013ab_shared.bdf", "sj0013b_unshared.bdf", pair = '13')
#loading_raw("sj0014a_unshared.bdf", "sj0014ab_shared.bdf", "sj0014b_unshared.bdf", pair = '14')
#loading_raw("sj0015a_unshared.bdf", "sj0015ab_shared.bdf", "sj0015b_unshared.bdf", pair = '15')
#loading_raw("sj0016a_unshared.bdf", "sj0016ab_shared.bdf", "sj0016b_unshared.bdf", pair = '16')
#loading_raw("sj0017a_unshared.bdf", "sj0017ab_shared.bdf", "sj0017b_unshared.bdf", pair = '17')
#loading_raw("sj0018a_unshared.bdf", "sj0018ab_shared.bdf", "sj0018b_unshared.bdf", pair = '18')


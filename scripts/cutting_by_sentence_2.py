# -*- coding: utf-8 -*-
"""
Created on Thu Dec  8 14:44:56 2022

@author: arnab
"""

import os
import numpy as np
import scipy.io.wavfile as wavfile
import pickle
import pandas as pd
from scipy.io import savemat
from scipy.io import loadmat
from functools import reduce 
#from utils import load_transcript, load_label, load_audio

def load_pickle(file_path):
    
    pickle_file = open(file_path, "rb")
    objects = []

    i=0

    while True:
        print('i',i)
        try:

            objects.append(pickle.load(pickle_file))

        except EOFError:

            break

    pickle_file.close()

    a=objects[0]
    
    return a

def load_label(filepath):

    with open(filepath, "rb") as f:
        full_labels = pickle.load(f)
        labels_df = pd.DataFrame(full_labels["labels"])

    labels_df["audio_onset"] = ((labels_df.onset + 3000) / 512)
    labels_df["audio_offset"] = ((labels_df.offset + 3000) / 512)

    labels_df = labels_df.dropna(subset=["audio_onset", "audio_offset"])

    return labels_df


def load_audio(filepath):
    fs, audio = wavfile.read(filepath)
    print(f"Sampling rate: {fs}")
    print(f"Audio Length (s): {len(audio) / fs}")
    return fs, audio


pickle_file_path='C:\Princeton\Research\whisper project\whisper-decoder\data\podcast\\777_full_labels.pkl'


def cut_audio_sentence(pickle_file_path,audio_path,saving_dir, Saving=True):   

    df=load_label(pickle_file_path)

    df['punc']=0

    for i in df.index:
    
        w=df.word[i]
    
        if w[-1]=='.':
            df.punc[i]=1

    segment_onset_id=[]
    segment_offset_id=[]
    segment_onset_id.append([df.audio_onset[df.index[0]], df.index[0],0])


    for i in range(len(df.index)):
    
        k=df.index[i]
    
        if df.punc[k]==1:
            segment_offset_id.append([df.audio_offset[k],k,i])
            segment_onset_id.append([df.audio_onset[df.index[i+1]],df.index[i+1],i+1])
    


    segment_offset_id.append([df.audio_offset[df.index[len(df.index)-1]],df.index[len(df.index)-1],len(df.index)-1])


   
    # audio_path='C:\Princeton\Research\whisper project\whisper-decoder\data\podcast\Podcast.wav'    
    # saving_dir='C:\\Princeton\Research\\whisper project\\whisper-decoder\\data\\podcast\\audio_sentence'


    fs, full_audio = load_audio(audio_path)

    sentence_onset_offset=[]

    for i in range(len(segment_onset_id)):    
    
        start_onset=segment_onset_id[i][0]
        end_onset=segment_offset_id[i][0]
        
        if (end_onset-start_onset)>0.1:
     
            chunk_data = full_audio[int(start_onset * fs) : int(end_onset * fs)]
            chunk_name = os.path.join(saving_dir, f"sentence_{i}.wav")
            
            if Saving:
                wavfile.write(chunk_name, fs, chunk_data)
        
            sentence_onset_offset.append([start_onset*512-3000,end_onset*512-3000])


    sentence_label=[]

    for i in range(len(segment_onset_id)):    
    
        a1=segment_onset_id[i][2]
    
    
        a2=segment_offset_id[i][2]+1
    
        indices=df.index[a1:a2]

        word=df.word[indices[0]]
        word=word+' '
        for j in range(1,len(indices)):
            word=word+df.word[indices[j]]
        
            if j<len(indices)-1:
                word=word+' '
            
            sentence_label.append(word)
            del word
            
    return sentence_label, sentence_onset_offset




# savemat('sentence_feature_label.mat',{'sentence_label': sentence_label})
# savemat('sentence_onset_offset_brain.mat',{'sentence_onset_offset': sentence_onset_offset})


### ECoG data cut

elec_csv_path='C:\Princeton\Research\whisper project\whisper-decoder\my codes\\717_elec_data.csv'
path_elec="C:\Princeton\Research\Hyper_alignment\significant electrode files\Patient 717\All significant electrode"


# ifg_significant=[4,9,10,18,27,66,71,74,75,78,79,80,82,86,87,88,95,108]
# stg_significant=[36,37,38,39,46,47,112,113,114,116,117,119,120,121,122,126]

def ecog_cut(elec_csv_path,path_elec,subject,sentence_onset_offset,shift_ms=300):

    df1 = pd.read_csv(elec_csv_path)

    ifg_significant=df1['ifg']
    stg_significant=df1['stg']


    electrodes=[ifg_significant,stg_significant]
    electrodes = reduce(lambda a, b: a+b, electrodes)



    
    ecogs=[]

    for i in electrodes:
        filename='NY'+str(subject)+'_111_Part1_conversation1_electrode_preprocess_file_'+str(i)+'.mat'
    
        path_elec_file=path_elec+'\\'+ filename
    
        e=loadmat(path_elec_file)['p1st'].squeeze().astype(np.float32)
        ecogs.append(e)

    ecogs = np.asarray(ecogs)    
    ecogs_ifg=ecogs[0:len(ifg_significant),:]    
    ecogs_stg=ecogs[len(ifg_significant): len(stg_significant)+len(ifg_significant) ,:]    

    ecogs = np.asarray(ecogs).T    
    ecogs_ifg=np.asarray(ecogs_ifg).T    
    ecogs_stg=np.asarray(ecogs_stg).T  

    bin_ms=62.5
    
    window_ms=625
    fs=512

    bin_fs = int(bin_ms / 1000 * fs)
    shift_fs = int(shift_ms / 1000 * fs)
    window_fs = int(window_ms / 1000 * fs)
    half_window = window_fs // 2
    # n_bins = window_ms // bin_ms
    n_bins = window_fs // bin_fs
    
    elec_data=[]
    elec_data_ifg=[]
    elec_data_stg=[]
    
    for i in range(len(sentence_onset_offset)):
    
    # if (i>5883):
    #     break
    
        onset = int(sentence_onset_offset[i][0])
        offset = int(sentence_onset_offset[i][1])
    # datum.append(df.word[i])
    # word_number.append(i)
    
        start = (onset) - half_window + shift_fs
        end = (offset) + half_window + shift_fs

        w=ecogs[int(start):int(end), :].mean(axis=1)
        w_ifg=ecogs_ifg[int(start):int(end), :].mean(axis=1)
        w_stg=ecogs_stg[int(start):int(end), :].mean(axis=1)
    
    # elec_data.append(w)
    # elec_data_ifg.append(w_ifg)
    # elec_data_stg.append(w_stg)


        elec_data.append(w)
        elec_data_ifg.append(w_ifg)
        elec_data_stg.append(w_stg)
        
    return elec_data, elec_data_ifg, elec_data_stg

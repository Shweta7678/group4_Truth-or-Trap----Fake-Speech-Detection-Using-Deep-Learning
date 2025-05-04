import os
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
import librosa
from torch.utils.data import Dataset
from RawBoost import ISD_additive_noise,LnL_convolutive_noise,SSI_additive_noise,normWav
from random import randrange
import random



___author__ = "Hemlata Tak"
__email__ = "tak@eurecom.fr"


def genSpoof_list( dir_meta,is_train=False,is_eval=False):
    
    d_meta = {}
    file_list=[]
    with open(dir_meta, 'r') as f:
         l_meta = f.readlines()

    if (is_train):
        for line in l_meta:
             _,key,_,label = line.strip().split()
             
             file_list.append(key)
             d_meta[key] = 1 if label == 'bonafide' else 0
        return d_meta,file_list
    
    elif(is_eval):
        for line in l_meta:
            key= line.strip()
            file_list.append(key)
        return file_list
    else:
        print("inside test dataloader")
        for line in l_meta:
             key,label = line.strip().split()
            #  print(f"key: {key}")
            #  print(f"label: {label}")
             file_list.append(key)
             d_meta[key] = label
        return d_meta,file_list



def pad(x, max_len=64600):
    x_len = x.shape[0]
    if x_len >= max_len:
        return x[:max_len]
    # need to pad
    num_repeats = int(max_len / x_len)+1
    padded_x = np.tile(x, (1, num_repeats))[:, :max_len][0]
    return padded_x	
			

from torch.utils.data.dataloader import default_collate

class Dataset_ASVspoof2019_train(Dataset):
    def __init__(self, args, list_IDs, labels, base_dir, algo):
        self.args     = args
        self.base_dir = base_dir
        self.algo     = algo
        self.cut      = 64600
        self.trim_top_db = args.trim_top_db

        # 1) Filter out any utt_id whose waveform would be empty
        valid_ids = []
        for utt_id in list_IDs:
            wav_path = os.path.join(base_dir, 'wav', utt_id + '.wav')
            try:
                X, _ = librosa.load(wav_path, sr=16000, mono=True)
            except Exception:
                continue
            if X.size > 0:
                valid_ids.append(utt_id)

        self.list_IDs = valid_ids
        # rebuild labels so indexing by utt_id works
        self.labels = {utt: labels[utt] for utt in valid_ids}

    def __len__(self):
        # return number of valid utterances
        return len(self.list_IDs)

    def __getitem__(self, index):
        utt_id = self.list_IDs[index]
        wav_path = os.path.join(self.base_dir, 'wav', utt_id + '.wav')

        # load & process
        X, sr = librosa.load(wav_path, sr=16000, mono=True)
        Y = process_Rawboost_feature(X, sr, self.args, self.algo)
        X_pad = pad(Y, self.cut)
        x_inp = torch.Tensor(X_pad)

        target = self.labels[utt_id]
        return x_inp, target


            

class Dataset_ASVspoof2021_eval(Dataset):
    def __init__(self, list_IDs, base_dir,labels):
        self.base_dir = base_dir
        self.cut = 64600
        self.labels = labels
        self.list_IDs = list_IDs
        # # 1) filter out any utt whose file is missing or empty
        # valid_ids = []
        # for utt in list_IDs:
        #     wav_path = os.path.join(self.base_dir, 'wav', utt + '.wav')
        #     try:
        #         X, _ = librosa.load(wav_path, sr=16000, mono=True)
        #     except Exception:
        #         continue
        #     if X.size > 0:
        #         valid_ids.append(utt)
        # self.list_IDs = valid_ids

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        
        utt_id = self.list_IDs[index]
        # wav_path = os.path.join(self.base_dir, utt_id)
        X, _ = librosa.load(f"{self.base_dir}/{utt_id}", sr=16000, mono=True)
        X_pad = pad(X, self.cut)
        # target = self.labels[utt_id]
        target = int(self.labels[utt_id])           # Convert string "0"/"1" to int 0/1
        target = torch.tensor(target)
        return torch.Tensor(X_pad),target





#--------------RawBoost data augmentation algorithms---------------------------##

def process_Rawboost_feature(feature, sr,args,algo):
    
    # Data process by Convolutive noise (1st algo)
    if algo==1:

        feature =LnL_convolutive_noise(feature,args.N_f,args.nBands,args.minF,args.maxF,args.minBW,args.maxBW,args.minCoeff,args.maxCoeff,args.minG,args.maxG,args.minBiasLinNonLin,args.maxBiasLinNonLin,sr)
                            
    # Data process by Impulsive noise (2nd algo)
    elif algo==2:
        
        feature=ISD_additive_noise(feature, args.P, args.g_sd)
                            
    # Data process by coloured additive noise (3rd algo)
    elif algo==3:
        
        feature=SSI_additive_noise(feature,args.SNRmin,args.SNRmax,args.nBands,args.minF,args.maxF,args.minBW,args.maxBW,args.minCoeff,args.maxCoeff,args.minG,args.maxG,sr)
    
    # Data process by all 3 algo. together in series (1+2+3)
    elif algo==4:
        
        feature =LnL_convolutive_noise(feature,args.N_f,args.nBands,args.minF,args.maxF,args.minBW,args.maxBW,
                 args.minCoeff,args.maxCoeff,args.minG,args.maxG,args.minBiasLinNonLin,args.maxBiasLinNonLin,sr)                         
        feature=ISD_additive_noise(feature, args.P, args.g_sd)  
        feature=SSI_additive_noise(feature,args.SNRmin,args.SNRmax,args.nBands,args.minF,
                args.maxF,args.minBW,args.maxBW,args.minCoeff,args.maxCoeff,args.minG,args.maxG,sr)                 

    # Data process by 1st two algo. together in series (1+2)
    elif algo==5:
        
        feature =LnL_convolutive_noise(feature,args.N_f,args.nBands,args.minF,args.maxF,args.minBW,args.maxBW,
                 args.minCoeff,args.maxCoeff,args.minG,args.maxG,args.minBiasLinNonLin,args.maxBiasLinNonLin,sr)                         
        feature=ISD_additive_noise(feature, args.P, args.g_sd)                
                            

    # Data process by 1st and 3rd algo. together in series (1+3)
    elif algo==6:  
        
        feature =LnL_convolutive_noise(feature,args.N_f,args.nBands,args.minF,args.maxF,args.minBW,args.maxBW,
                 args.minCoeff,args.maxCoeff,args.minG,args.maxG,args.minBiasLinNonLin,args.maxBiasLinNonLin,sr)                         
        feature=SSI_additive_noise(feature,args.SNRmin,args.SNRmax,args.nBands,args.minF,args.maxF,args.minBW,args.maxBW,args.minCoeff,args.maxCoeff,args.minG,args.maxG,sr) 

    # Data process by 2nd and 3rd algo. together in series (2+3)
    elif algo==7: 
        
        feature=ISD_additive_noise(feature, args.P, args.g_sd)
        feature=SSI_additive_noise(feature,args.SNRmin,args.SNRmax,args.nBands,args.minF,args.maxF,args.minBW,args.maxBW,args.minCoeff,args.maxCoeff,args.minG,args.maxG,sr) 
   
    # Data process by 1st two algo. together in Parallel (1||2)
    elif algo==8:
        
        feature1 =LnL_convolutive_noise(feature,args.N_f,args.nBands,args.minF,args.maxF,args.minBW,args.maxBW,
                 args.minCoeff,args.maxCoeff,args.minG,args.maxG,args.minBiasLinNonLin,args.maxBiasLinNonLin,sr)                         
        feature2=ISD_additive_noise(feature, args.P, args.g_sd)

        feature_para=feature1+feature2
        feature=normWav(feature_para,0)  #normalized resultant waveform
 
    # original data without Rawboost processing           
    else:
        
        feature=feature
    
    return feature

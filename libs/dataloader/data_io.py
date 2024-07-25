import torch
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader
import subprocess, librosa
import soundfile as sf
from .customize_collate_fn import customize_collate
from .customize_sampler import SamplerBlockShuffleByLen
from tqdm import tqdm
__author__ = "Junyan Wu"
__email__ = "wujy298@mail2.sysu.edu.cn"

class ASVspoof2019PS(Dataset):
    def __init__(self, path ,part='train', rso = 20):
        self.part = part
        self.path = os.path.join(path, self.part)
        self.protocol_rso = 20
        protocol_path="%s/segment_labels/%s_seglab_0.02.npy"%(path,self.part)
        self.protocol = np.load(protocol_path, allow_pickle=True).item()
        self.filelist,self.labels=[], []
        self.gt_dict={}
        self.rso=rso
        for k in self.protocol.keys():
            self.filelist.append(k)
            self.gt_dict[k]=np.array(self.label2gt(self.protocol[k]),dtype=float)
            scale_label = self.lab_in_scale(self.protocol[k])
            self.labels.append(scale_label)
        self.rso=rso
    def __getitem__(self, idx):
        filename = self.filelist[idx]
        label=self.protocol[filename]
        filepath = os.path.join(self.path, 'con_wav',filename + ".wav")
        featureTensor = self.torchaudio_load(filepath)
        featureTensor = featureTensor.float()
        label = np.array(label,dtype=int)
        label = self.lab_in_scale(label)
        label = torch.tensor(label, dtype=torch.float32)
        featureTensor = torch.squeeze(featureTensor,dim=0)
        return featureTensor, filename, label
    
    def __len__(self):
        return len(self.filelist)
    
    def torchaudio_load(self,filepath):
        wave, sr = sf.read(filepath)
        # wave, sr = librosa.load(filepath)
        waveform = torch.Tensor(np.expand_dims(wave, axis=0))
        return waveform
    
    def lab_in_scale(self, labvec):
        shift=int(self.rso/self.protocol_rso)
        num_frames = int(len(labvec) / shift)
        if(num_frames==0):
            num_frames=1
            new_lab = np.zeros(num_frames, dtype=int)
            new_lab[0] = min(labvec) 
        else:
            new_lab = np.zeros(num_frames, dtype=int)
            for idx in np.arange(num_frames):
                st, et  = int(idx * shift), int((idx+1)*shift)
                new_lab[idx] = min(labvec[st:et])
        return new_lab


    def f_get_seq_len_list(self):
        return [len(x) for x in self.labels]
    
    def label2gt(self, rsolabel):
        rso=self.protocol_rso
        fake_segments = []
        prev_label = None
        current_start = 0
        fake_label=0
        true_label=1
        for i, label in enumerate(rsolabel):
            label = int(label)
            time = i * rso
            if label == fake_label and prev_label == true_label: # mark fake start
                current_start = time
                if i == len(rsolabel) - 1:  # the end
                    fake_segments.append((current_start, time + rso))
            elif label == true_label and prev_label == fake_label:# mark fake end
                fake_segments.append((current_start, time))
            elif label == fake_label and i == len(rsolabel) - 1: # the end
                fake_segments.append((current_start, time + rso))
            prev_label = label
        gt = [[float(start / 1000), float(end / 1000)] for start, end in fake_segments]
        return gt
    def get_gt_dict(self):
        return self.gt_dict
    

class HAD(ASVspoof2019PS):
    def __init__(self, path,  part='train',rso=20):
        self.part = part
        self.path = os.path.join(path, 'HAD_'+self.part)
        protocol_path=os.path.join(self.path, 'HAD_%s_label.txt'%self.part)
        self.protocol = np.loadtxt(protocol_path,dtype=str).reshape(-1,3)
        self.filelist = self.protocol[:,0]
        self.seg_labels=self.protocol[:,1]
        self.utt_labels=self.protocol[:,2]
        self.labels=[]
        self.gt_dict={}
        self.rso=rso
        self.protocol_rso = 20
        for filename,label_str in zip(self.filelist,self.seg_labels):
            label=self.parse_seglabel(label_str)
            scale_label=self.lab_in_scale(label)
            label_split = [item.split('-') for item in label_str.split('/')]
            fake_region = [sublist[:2] for sublist in label_split if sublist[2] == 'F']
            self.gt_dict[filename]=np.array(fake_region,dtype=float)
            self.labels.append(scale_label)

    def __len__(self):
        return len(self.filelist)
    
    def __getitem__(self, idx):
        filename = self.filelist[idx]
        filepath = os.path.join(self.path, 'conbine', filename + ".wav")
        if self.part=='test':
            filepath = os.path.join(self.path, 'test', filename + ".wav")
        featureTensor = self.torchaudio_load(filepath)
        featureTensor = featureTensor.float()
        label = self.labels[idx]
        label = torch.tensor(label, dtype=torch.float32)
        featureTensor = torch.squeeze(featureTensor,dim=0)
        return featureTensor, filename, label
    
    def parse_seglabel(self,label_str):
        labelstr_split=label_str.split('/')
        labels=[]
        for item in labelstr_split:
            s,e,y=item.split("-") # [start, end, T/F]
            label,s_idx,e_idx=self._parse_seglabel(s,e,y)
            labels.extend(label)
        label_np=np.array(labels,dtype=int).reshape(-1)
        return label_np
    
    def _parse_seglabel(self, s,e,y):
        label_dict={'T':1,'F':0}
        labels=label_dict[y]
        s_ms=float(s)*1000
        e_ms=float(e)*1000
        s_idx=int(s_ms/self.rso)
        e_idx=int(e_ms/self.rso)
        total=e_idx-s_idx
        return_label=total*[labels]
        return return_label,s_idx,e_idx
        
    def f_get_seq_len_list(self):
        return [len(x) for x in self.labels]
    
    def get_gt_dict(self):
        return self.gt_dict

    
class LAVDF(HAD):
    def __init__(self, path,  part='train',rso=20):
        self.part = part
        self.path = path
        self.protocol_path = os.path.join(path, 'metadata.min.json')
        protocol_json = self.readjson(self.protocol_path)
        self.protocol_json=self.split_part(protocol_json,part)
        self.rso=rso
        self.protocol_rso = 20

    def __getitem__(self, idx):
        filename = self.protocol_json[idx]['file']
        mp4path = os.path.join(self.path, filename)
        """
            too slow
            video, audio, info = torchvision.io.read_video(mp4path, pts_unit="sec")
        """
        filepath = mp4path.replace(".mp4",".wav")
        cmd="ffmpeg -y -i '%s' -acodec pcm_s16le -f s16le -ac 1 -ar 16000 -f wav '%s'"%(mp4path,filepath)
        if not os.path.exists(filepath):
            cmd="ffmpeg -y -i '%s' -acodec pcm_s16le -f s16le -ac 1 -ar 16000 -f wav '%s'"%(mp4path,filepath)
            subprocess.run(cmd, shell=True, stdout=True, stderr=subprocess.DEVNULL)
        audio= self.torchaudio_load(filepath)
        dur = audio.shape[-1]/16000
        seg_label, utt_label, fake_region=self.parse_label(self.protocol_json[idx],dur)
        featureTensor = audio.float()
        featureTensor = torch.squeeze(featureTensor,dim=0)
        scale_label=self.lab_in_scale(seg_label)
        label = torch.tensor(scale_label, dtype=torch.float32)
        return featureTensor, filename, label

    
    def readjson(self,jsonfile):
        import json
        with open(jsonfile, 'r') as file:
            data = json.load(file)
        return data

    def split_part(self,json_data,part):
        json_data = [item for item in json_data if item['split'] == part]
        return json_data

    def parse_label(self,item,dur):
        utt_label='bonafide' if item['modify_audio']==False else 'spoof'
        fake_region=[] if utt_label=='bonafide' else item['fake_periods']
        seg_label=self.parse_seglabel(fake_region, dur)
        return seg_label, utt_label, fake_region
    
    def __len__(self):
        return len(self.protocol_json)

    def fill_intervals(self, dur, fake_intervals):
        fake_intervals.sort(key=lambda x: x[0])
        filled_intervals = []
        last_end = 0
        for interval in fake_intervals:
            start, end = interval
            if start > last_end:
                filled_intervals.append([last_end, start, 'T'])
            filled_intervals.append([start, end, 'F'])
            last_end = end
        if last_end < dur:
            filled_intervals.append([last_end, dur, 'T'])
        return filled_intervals
    
    def parse_seglabel(self,fake_regions,dur):
        label_fills=self.fill_intervals(dur, fake_regions)
        labels=[]
        for label_fill in label_fills:
            s,e,y=label_fill 
            label2num,s_idx,e_idx=self._parse_seglabel(s,e,y)
            labels.extend(label2num)
        labels_np=np.array(labels,dtype=int).reshape(-1)
        return labels_np
    
        
    def f_get_seq_len_list(self):
        return [x['audio_frames'] for x in self.protocol_json]
    
    def get_gt_dict(self):
        gt_dict={}
        for idx in range(len(self.protocol_json)):
            filename = self.protocol_json[idx]['file']
            if self.protocol_json[idx]['modify_audio']==True:
                gt_dict[filename]=np.array(self.protocol_json[idx]['fake_periods'],dtype=float)
            else:
                gt_dict[filename]=np.array([],dtype=float)
        return gt_dict


########CHANGE########PATH##########
def get_dataloader(batch_size,part,dn,rso):
    assert part in ['train', 'dev','test']
    if dn=="PS":
        part=part.replace("test","eval")
        dst=ASVspoof2019PS(path="/data/wujy/audio/ps",part=part,rso=rso)######
    elif dn=='HAD':
        dst=HAD(path='/data/wujy/audio/HAD', part=part, rso=rso)####
    elif dn=='LAVDF':
        dst=LAVDF(path="/data/wujy/audio/LAV-DF/", part=part, rso=rso)
    gt_dict=dst.get_gt_dict()
    if batch_size > 1:
        collate_fn = customize_collate
        tmp_sampler = SamplerBlockShuffleByLen(dst.f_get_seq_len_list(), batch_size=batch_size)
        dlr=torch.utils.data.DataLoader(dst, collate_fn=collate_fn,batch_size=batch_size,sampler=tmp_sampler,num_workers=8)
    else:
        dlr=torch.utils.data.DataLoader(dst, batch_size=1,num_workers=8,shuffle=True if part=='train' else False)
    del dst
    return gt_dict,dlr

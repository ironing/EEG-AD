import os
import numpy as np
from numpy import unravel_index
import array
import mne
from mne.io import concatenate_raws, read_raw_edf, read_raw_fif, RawArray
import struct
import scipy.io as io

'''
chb01	F	11
chb02	M	11
chb03	F	14
chb04	M	22
chb05	F	 7
chb06	F	 1.5
chb07	F	14.5
chb08	M	 3.5
chb09	F	10
chb10	M	 3
chb11	F	12
chb12	F	 2
chb13	F	 3
chb14	F	 9
chb15	M	16
chb16	F	 7
chb17	F	12
chb18	F	18
chb19	F	19
chb20	F	 6
chb21	F	13
chb22	F	 9
chb23	F	 6
'''



# pick abnormal EEG
root_dir = '/data/CHB/files/chbmit/1.0.0/dataset_chb'
Dirs = ['chb01', 'chb02', 'chb03', 'chb04', 'chb05',
        'chb06', 'chb07', 'chb08', 'chb09', 'chb10',
        'chb11', 'chb12', 'chb13', 'chb14', 'chb15',
        'chb16', 'chb17', 'chb18', 'chb19', 'chb20',
        'chb21', 'chb22', 'chb23'
        ]
total = 0
for Dir in Dirs:
    Path = os.path.join(root_dir, Dir)
    save_dir = os.path.join('./data/dataset_chb/pick_abnormal', Dir)
    seizure_files = []
    for root, dirs, files in os.walk(Path):
        for file in files:
            path = os.path.join(root, file)
            if ".seizures" in path:
                seizure_files.append(file[:-9])
            if "summary.txt" in path:
                with open(path, "r") as f:
                    annotations = f.readlines()

    for seizure in seizure_files:
        for i, anno in enumerate(annotations):
            if seizure in anno:
                break
        file_name = annotations[i].split(':')[1].replace(" ", "").replace("\n", "")
        file_path = os.path.join(Path, file_name)
        save_name = os.path.join(save_dir, file_name[:-4])
        if "File Start Time" in annotations[i+1]:
            i += 3
        else:
            i += 1
       
        nums = int(annotations[i].split(':')[1])
        raw = read_raw_edf(file_path)
        for j in range(nums):
            anno_start = annotations[i + 1]
            anno_end = annotations[i + 2]
            s = int(anno_start.split(':')[1].split('seconds')[0].replace(" ", ""))
            e = int(anno_end.split(':')[1].split('seconds')[0].replace(" ", ""))
            new_raw = raw.copy()
            new_raw.crop(tmin=s, tmax=e)
            new_raw.info['meas_date'] = None
            save_path = save_name + '_a{}raw.fif'.format(total)
            new_raw.save(save_path)
            total += 1
            i += 2
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
# pick normal EEG          
Dirs = ['chb01', 'chb06', 'chb09', 'chb11',
        'chb20', 'chb21', 'chb23',

        'chb02', 'chb03', 'chb04', 'chb05',
        'chb07', 'chb08', 'chb10',
        'chb14', 'chb15', 'chb16', 'chb17', 'chb18', 'chb19',
        'chb22'
        ]
total = 0
for Dir in Dirs:
    Path = os.path.join(root_dir, Dir)
    save_dir = os.path.join('./data/dataset_chb/pick_normal', Dir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    normal_files = []
    for root, dirs, files in os.walk(Path):
        for file in files:
            if "summary.txt" in file:
                continue
            if "seizures" in file:
                continue
            if "html" in file:
                continue
            if "{}.seizures".format(file) in files:
                continue
            path = os.path.join(root, file)
            normal_files.append(path)

    for path in normal_files:
        print(path)
        raw = read_raw_edf(path, verbose=False)
        data = raw.get_data()
        length = data.shape[1]
        new_raw = raw.copy()
        if raw.times[-1] >= 3599:
            new_raw.crop(tmin=0, tmax=3599)
            
        new_raw.info['meas_date'] = None
        file_name = path.split('/')[-1][:-4]
        save_path = os.path.join(save_dir, file_name + "_{}_raw.fif".format(total))
        new_raw.save(save_path)
        total += 1






selection = ['FP1-F7', 'F7-T7', 'T7-P7', 'P7-O1', 'FP1-F3', 'F3-C3',
             'C3-P3', 'P3-O1', 'FP2-F4', 'F4-C4', 'C4-P4', 'P4-O2',
             'FP2-F8', 'F8-T8', 'T8-P8', 'T8-P8-0', 'P8-O2', 'FZ-CZ', 'CZ-PZ']

selection_1 = ['FP1', 'F7', 'T7', 'P7', 'FP1', 'F3',
               'C3', 'P3', 'FP2', 'F4', 'C4', 'P4',
               'FP2', 'F8', 'T8', 'P8', 'FZ', 'CZ']

selection_2 = ['FP1-CS2', 'F7-CS2', 'T7-CS2', 'P7-CS2', 'FP1-CS2', 'F3-CS2',
               'C3-CS2', 'P3-CS2', 'FP2-CS2', 'F4-CS2', 'C4-CS2', 'P4-CS2',
               'FP2-CS2', 'F8-CS2', 'T8-CS2', 'P8-CS2', 'FZ-CS2', 'CZ-CS2']


Dirs = ['chb01', 'chb06', 'chb09', 'chb11',
        'chb20', 'chb21', 'chb23',

        'chb02', 'chb03', 'chb04', 'chb05',
        'chb07', 'chb08', 'chb10',
        'chb14', 'chb15', 'chb16', 'chb17', 'chb18', 'chb19',
        'chb22'
        ]



# generate 3s normal EEG
save_dir_normal = './data/dataset_chb/MAT/normal/3s769'
if not os.path.exists(save_dir_normal):
  os.makedirs(save_dir_normal)
total = 0
for Dir in Dirs:
    save_dir = os.path.join(save_dir_normal, Dir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    Path = os.path.join('./data/dataset_chb/pick_normal', Dir)
    for root, dirs, files in os.walk(Path):
        for file in files:
            path = os.path.join(root, file)
            raw = read_raw_fif(path, preload=True, verbose=False)
            ch_names = raw.ch_names
            for ch in ch_names:
                if ch.upper() not in selection:
                    raw.drop_channels(ch)

            if len(raw.ch_names) != 18:
                continue
            data = raw.get_data()
            length = data.shape[1]
            nums = int(length / 769)
            for j in range(nums):
                if (j + 1) * 769 > length:
                    break
                data_ = data[:, j * 769:(j + 1) * 769]
                savepath = os.path.join(save_dir, file[:-4] + '_a{}_t{}.mat'.format(j, total))
                dict = {}
                dict['data'] = data_
                io.savemat(savepath, dict)
                total += 1


                
                
                
                
# generate 3s abnormal EEG
save_dir_abnormal = './data/dataset_chb/MAT/abnormal/3s769'
if not os.path.exists(save_dir_abnormal):
  os.makedirs(save_dir_abnormal)
total = 0
for Dir in Dirs:
    save_dir = os.path.join(save_dir_abnormal, Dir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    abnormal_files = []
    for root, dirs, files in os.walk('./data/dataset_chb/pick_abnormal'):
        for file in files:
            if Dir in file:
                abnormal_files.append(os.path.join('./data/dataset_chb/pick_abnormal', file))

    for path in abnormal_files:
        raw = read_raw_fif(path, verbose=False)
        ch_names = raw.ch_names

        if selection[0] in ch_names:
            sel = selection
        elif selection_1[0] in ch_names:
            sel = selection_1
        else:
            sel = selection_2

        for ch in ch_names:
            if ch.upper() not in sel:
                raw.drop_channels(ch)

        data = raw.get_data()
        length = data.shape[1]
        nums = int(length / 769)
        for j in range(nums):
            if (j + 1) * 769 > length:
                break
            data_ = data[:, j * 769:(j + 1) * 769]
            savepath = os.path.join(save_dir, path.split('/')[-1][:-4] + '_a{}_t{}.mat'.format(j, total))
            dict = {}
            dict['data'] = data_
            io.savemat(savepath, dict)
            total += 1



            
            
            
            
# EEG path ——> txt
root_dir = './data/dataset_chb/MAT/normal/3s769/'
normal_paths = []

for root, dirs, files in os.walk(root_dir):
    for file in files:
        path = os.path.join(root, file)
        normal_paths.append(path) 
random.shuffle(normal_paths)

abnormal_paths = []
for root, dirs, files in os.walk('./data/dataset_chb/MAT/abnormal/3s769/'):
    for file in files:
        path = os.path.join(root, file)
        abnormal_paths.append(path)
random.shuffle(abnormal_paths)

# print(len(normal_paths), len(abnormal_paths))


save_dir = './data/all_TXT_3s'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

with open('{}/train.txt'.format(save_dir), 'w+') as f:
    for file in normal_paths:
        f.write(file + "\r\n")

with open('{}/test.txt'.format(save_dir), 'w+') as f:
    for file in abnormal_paths:
        f.write(file + "\r\n")










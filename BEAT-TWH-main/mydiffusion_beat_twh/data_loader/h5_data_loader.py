import torch
import h5py
import numpy as np
from torch.utils.data import DataLoader
import pdb
import os


speaker_id_dict = {
    2: 0,
    10: 1,
}


class SpeechGestureDataset(torch.utils.data.Dataset):
    def __init__(self, h5file, motion_dim, style_dim, sequence_length=30*5, npy_root="../../process", 
                 version='v0', dataset='TWH', in_memory=True):
        self.h5_path = h5file
        self.h5 = None
        with h5py.File(h5file, "r") as h5:
            self.len = len(h5.keys())
        self.motion_dim = motion_dim
        self.style_dim = style_dim
        self.version = version
        
        self.gesture_mean = np.load(os.path.join(npy_root, "gesture_" + dataset + "_mean_" + self.version + ".npy"))
        self.gesture_std = np.load(os.path.join(npy_root, "gesture_" + dataset + "_std_" + self.version + ".npy"))

        # Speaker ids are small; keep them in memory.
        with h5py.File(h5file, "r") as h5:
            if dataset == 'BEAT':
                self.id = [speaker_id_dict[int(h5[str(i)]["speaker_id"][:][0])] for i in range(self.len)]
            else:
                self.id = [int(h5[str(i)]["speaker_id"][:][0]) for i in range(self.len)]

        self.in_memory = bool(in_memory)
        self.audio = None
        self.text = None
        self.gesture = None

        if self.in_memory:
            # WARNING: This can use a lot of RAM for large datasets.
            with h5py.File(h5file, "r") as h5:
                self.audio = [h5[str(i)]["audio"][:] for i in range(self.len)]
                self.text = [h5[str(i)]["text"][:] for i in range(self.len)]
                self.gesture = [
                    (h5[str(i)]["gesture"][:] - self.gesture_mean) / self.gesture_std
                    for i in range(self.len)
                ]
        self.sequence_length = sequence_length

        print("Total clips:", self.len, "| in_memory:", self.in_memory)
        self.segment_length = sequence_length

    def __len__(self):
        return self.len

    def _get_h5(self):
        if self.h5 is None:
            # Lazily open per-process/per-worker.
            self.h5 = h5py.File(self.h5_path, "r")
        return self.h5

    def __getitem__(self, idx):
        if self.in_memory:
            total_frame_len = self.audio[idx].shape[0]
        else:
            h5 = self._get_h5()
            total_frame_len = h5[str(idx)]["audio"].shape[0]
        start_frame = np.random.randint(0, total_frame_len - self.segment_length)
        end_frame = start_frame + self.segment_length

        if self.in_memory:
            audio = self.audio[idx][start_frame:end_frame]
            text = self.text[idx][start_frame:end_frame]
            posrat = self.gesture[idx][start_frame:end_frame]
        else:
            h5 = self._get_h5()
            grp = h5[str(idx)]
            audio = grp["audio"][start_frame:end_frame]
            text = grp["text"][start_frame:end_frame]
            posrat = (grp["gesture"][start_frame:end_frame] - self.gesture_mean) / self.gesture_std

        textaudio = np.concatenate((audio, text), axis=-1)
        textaudio = torch.FloatTensor(textaudio)

        # Compute vel/acc on the fly for the segment to avoid huge RAM usage.
        if "v0" in self.version:
            vel = np.concatenate(
                (np.zeros([1, self.motion_dim], dtype=posrat.dtype), posrat[1:] - posrat[:-1]),
                axis=0,
            )
            acc = np.concatenate(
                (np.zeros([1, self.motion_dim], dtype=posrat.dtype), vel[1:] - vel[:-1]),
                axis=0,
            )
            gesture = np.concatenate((posrat, vel, acc), axis=-1)   #so the motion feature is motion_dim * 3
        else:
            gesture = posrat
        
        gesture = torch.FloatTensor(gesture)
        speaker = np.zeros([self.style_dim])
        # speaker[0] = 1      # dummy speaker
        speaker[self.id[idx]] = 1
        speaker = torch.FloatTensor(speaker)
        return textaudio, gesture, speaker


class RandomSampler(torch.utils.data.Sampler):
    def __init__(self, min_id, max_id):
        self.min_id = min_id
        self.max_id = max_id
    def __len__(self):
        return self.max_id - self.min_id
    def __iter__(self):
        while True:
            yield np.random.randint(self.min_id, self.max_id)


class SequentialSampler(torch.utils.data.Sampler):
    def __init__(self, min_id, max_id):
        self.min_id = min_id
        self.max_id = max_id
    def __iter__(self):
        return iter(range(self.min_id, self.max_id))


if __name__ == '__main__':
    '''
    cd ./BEAT-main/mydiffusion_beat/data_loader
    python h5_data_loader.py
    '''
    # Get data, data loaders and collate function ready
    print("Loading dataset into memory ...")
    trn_dataset = SpeechGestureDataset("../../process/speaker_2_10_v0.h5", motion_dim=684, style_dim=2)

    train_loader = DataLoader(trn_dataset, num_workers=4,
                              sampler=RandomSampler(0, len(trn_dataset)),
                              batch_size=128,
                              pin_memory=True,
                              drop_last=False)

    for batch_i, batch in enumerate(train_loader, 0):
        textaudio, gesture, speaker = batch     # (128, 150, 1435), (128, 150, 744), (128, 17)
        print(batch_i)
        pdb.set_trace()

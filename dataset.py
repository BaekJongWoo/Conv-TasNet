import tensorflow as tf
import numpy as np
import musdb
import gc
from tqdm import tqdm


class DecodedTrack:

    def __init__(self, length, mixed, stems):
        self.length = length
        self.mixed = mixed
        self.stems = stems

class Dataset:
    
    def __init__(self, E, M, C, W, STEMS, dataset_path, subsets:str="train", memory_limit: bool=False):

        self.E, self.M, self.C, self.W = E, M, C, W
        self.STEMS = STEMS
        self.subsets = subsets
        self.dataset_path = dataset_path

        self.tracks_train = list(musdb.DB(root=dataset_path, subsets=subsets))

        if memory_limit == memory_limit:
            self.tracks_train = np.random.choice(self.tracks_train, size=40, replace=False)

        self.num_tracks = len(self.tracks_train)
        print("Decoding Tracks: {} Tracks".format(self.num_tracks))
        for i in tqdm(range(self.num_tracks)):
            self.tracks_train[i] = self.decode(self.tracks_train[i])
    
    def shuffle(self):
        self.tracks_train = list(musdb.DB(root=self.dataset_path, subsets=self.subsets))
        self.tracks_train = np.random.choice(self.tracks_train, size=50, replace=False)
        self.num_tracks = len(self.tracks_train)
        print("Decoding Tracks: {} Tracks".format(self.num_tracks))
        for i in tqdm(range(self.num_tracks)):
            self.tracks_train[i] = self.decode(self.tracks_train[i])

    def decode(self, track):
        mixed = (track.audio[:, 0].flatten(), track.audio[:, 1].flatten())
        length = mixed[0].shape[-1]
        stems = {}
        for key in self.STEMS:
            stems[key] = (track.targets[key].audio[:, 0].flatten(), 
                          track.targets[key].audio[:, 1].flatten())
        gc.collect()
        return DecodedTrack(length=length, mixed=mixed, stems=stems)

    def get_dataset(self):
        indices = np.random.choice(list(range(self.num_tracks)), self.num_tracks, replace=False)

        duration = self.W
        x_batch = np.zeros((self.E * self.M * 2, self.W))
        y_batch = np.zeros((self.E * self.M * 2, self.C, self.W))

        for i in range(self.E * self.M):
            track = self.tracks_train[np.random.choice(indices)]
            start = np.random.randint(0, track.length - duration)

            left = i * 2
            right = left + 1
            
            x_batch[left] = track.mixed[0][start:start+duration]
            x_batch[right] = track.mixed[1][start:start+duration]

            for c, stem in enumerate(self.STEMS):
                y_batch[left][c] = track.stems[stem][0][start:start+duration]
                y_batch[right][c] = track.stems[stem][1][start:start+duration]
        
        return np.array(x_batch), np.array(y_batch)

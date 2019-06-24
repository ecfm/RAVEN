import re
from collections import defaultdict

import torch
import torch.utils.data as Data
import numpy as np
from consts import global_consts as gc
import sys

if gc.SDK_PATH is None:
    print("SDK path is not specified! Please specify first in constants/paths.py")
    exit(0)
else:
    print("Added gc.SDK_PATH")
    import os

    print(os.getcwd())
    sys.path.append(gc.SDK_PATH)

import mmsdk.mmdatasdk.dataset.standard_datasets.CMU_MOSI.cmu_mosi_std_folds as std_folds

from mmsdk import mmdatasdk as md

from consts import global_consts as gc

DATASET = md.cmu_mosi

# obtain the train/dev/test splits - these splits are based on video IDs
trainvid = std_folds.standard_train_fold
testvid = std_folds.standard_test_fold
validvid = std_folds.standard_valid_fold


def mid(a):
    return (a[0] + a[1]) / 2.0


class MOSISubdata():
    def __init__(self, name="train"):
        self.name = name
        self.covarepInput = []
        self.covarepLength = []
        self.wordInput = []
        self.wordLength = []
        self.facetInput = []
        self.facetLength = []
        self.labelOutput = []


def normalize_len(feature_array_list, dim):
    lengths_to_append = []
    to_append = []
    for feature_array in feature_array_list:
        if len(feature_array) > gc.shift_padding_len:
            feature_array = feature_array[:gc.shift_padding_len]
        lengths_to_append.append(len(feature_array))
        if len(feature_array) == 0:
            feature_array = np.zeros((gc.shift_padding_len , dim))
        elif len(feature_array) < gc.shift_padding_len:
            padding = np.zeros((gc.shift_padding_len - len(feature_array), dim))
            feature_array = np.vstack([feature_array, padding])
        to_append.append(feature_array[:])
    return lengths_to_append, to_append


class MOSIDataset(Data.Dataset):
    trainset = MOSISubdata("train")
    testset = MOSISubdata("test")
    validset = MOSISubdata("valid")

    def __init__(self, root, cls="train", src="csd", save=False):
        self.root = root
        self.cls = cls
        if len(MOSIDataset.trainset.labelOutput) != 0 and cls != "train":
            print("Data has been previously loaded, fetching from previous lists.")
        else:
            self.load_data()

        if self.cls == "train":
            self.dataset = MOSIDataset.trainset
        elif self.cls == "test":
            self.dataset = MOSIDataset.testset
        elif self.cls == "valid":
            self.dataset = MOSIDataset.validset

        self.covarepInput = self.dataset.covarepInput[:]
        self.covarepLength = self.dataset.covarepLength[:]
        self.wordLength = self.dataset.wordLength[:]
        self.wordInput = self.dataset.wordInput[:]
        self.facetInput = self.dataset.facetInput[:]
        self.facetLength = self.dataset.facetLength[:]
        self.labelOutput = self.dataset.labelOutput[:]

    def load_data(self):
        DATASET = md.cmu_mosi
        try:
            md.mmdataset(DATASET.highlevel, gc.data_path)
        except RuntimeError:
            print("High-level features have been downloaded previously.")

        try:
            md.mmdataset(DATASET.raw, gc.data_path)
        except RuntimeError:
            print("Raw data have been downloaded previously.")

        try:
            md.mmdataset(DATASET.labels, gc.data_path)
        except RuntimeError:
            print("Labels have been downloaded previously.")

        facet_field = 'CMU_MOSI_VisualFacet_4.1'
        covarep_field = 'CMU_MOSI_COVAREP'
        word_field = 'CMU_MOSI_TimestampedWordVectors_1.1'

        features = [
            word_field,
            facet_field,
            covarep_field,
        ]

        recipe = {feat: os.path.join(gc.data_path, feat) + '.csd' for feat in features}
        dataset = md.mmdataset(recipe)
        dataset.impute(word_field)
        vid0 = list(dataset[word_field].keys())[0]
        gc.wordDim = len(dataset[word_field][vid0]['features'][0])
        gc.facetDim = len(dataset[facet_field][vid0]['features'][0])
        gc.covarepDim = len(dataset[covarep_field][vid0]['features'][0])

        label_field = 'CMU_MOSI_Opinion_Labels'

        # we add and align to lables to obtain labeled segments
        # this time we don't apply collapse functions so that the temporal sequences are preserved
        label_recipe = {label_field: os.path.join(gc.data_path, label_field + '.csd')}
        dataset.add_computational_sequences(label_recipe, destination=None)
        dataset.impute(label_field)
        dataset.align(label_field)
        dataset.align(word_field)

        labels = defaultdict(lambda: [])
        words = defaultdict(lambda: [])
        facets = defaultdict(lambda: [])
        covareps = defaultdict(lambda: [])
        segment_field_pairs = [(labels, label_field), (words, word_field), (facets, facet_field),
                               (covareps, covarep_field)]
        field_with_nan = [label_field, facet_field, covarep_field]
        for vid_label_word in dataset[label_field].keys():
            # define a regular expression to extract the video ID out of the keys
            pattern = re.compile('(.*\[\d+\])\[\d+\]')
            # get the video ID and the features out of the aligned dataset
            vid_label = re.search(pattern, vid_label_word).group(1)
            for segment_features, field in segment_field_pairs:
                features = []
                if vid_label_word in dataset[field].keys():
                    if field == label_field or field == word_field:
                        # labels and words are references for alignment. 'features' only contains 1 word
                        assert len(dataset[field][vid_label_word]['features']) == 1
                        features = dataset[field][vid_label_word]['features'][0]
                    else:
                        features = dataset[field][vid_label_word]['features']
                else:
                    print('Segment %s not found in sequence %s' % (vid_label_word, field))
                if field in field_with_nan:
                    features = np.nan_to_num(features)
                segment_features[vid_label].append(features)

        for vid_label in labels.keys():
            pattern = re.compile('(.*)\[\d+\]')
            vid = re.search(pattern, vid_label).group(1)

            if vid in trainvid:
                dataset = MOSIDataset.trainset
            elif vid in validvid:
                dataset = MOSIDataset.validset
            elif vid in testvid:
                dataset = MOSIDataset.testset
            else:
                print(f"Found video that doesn't belong to any splits: {vid}")
                continue
            num_words = len(words[vid_label])
            if num_words > gc.padding_len or num_words == 0:
                continue
            dataset.wordLength.append(num_words)
            dataset.wordInput.append(words[vid_label])
            # all the labels within the same segment are the same
            dataset.labelOutput.append(labels[vid_label][0])

            lengths_to_append, to_append = normalize_len(facets[vid_label], gc.facetDim)
            dataset.facetInput.append(to_append[:])
            dataset.facetLength.append(lengths_to_append[:])

            lengths_to_append, to_append = normalize_len(covareps[vid_label], gc.covarepDim)
            dataset.covarepInput.append(to_append[:])
            dataset.covarepLength.append(lengths_to_append[:])

    def __getitem__(self, index):
        inputLen = self.wordLength[index]
        return torch.cat((torch.tensor(self.wordInput[index], dtype=torch.float32),
                          torch.zeros((gc.padding_len - len(self.wordInput[index]), gc.wordDim))), 0), \
               torch.cat((torch.tensor(self.covarepInput[index], dtype=torch.float32), torch.zeros(
                   (gc.padding_len - len(self.covarepInput[index]), gc.shift_padding_len, gc.covarepDim))), 0), \
               torch.cat((torch.tensor(self.covarepLength[index], dtype=torch.long),
                          torch.zeros(gc.padding_len - len(self.covarepLength[index]), dtype=torch.long)), 0), \
               torch.cat((torch.tensor(self.facetInput[index], dtype=torch.float32), torch.zeros(
                   (gc.padding_len - len(self.facetInput[index]), gc.shift_padding_len, gc.facetDim))), 0), \
               torch.cat((torch.tensor(self.facetLength[index], dtype=torch.long),
                          torch.zeros(gc.padding_len - len(self.facetLength[index]), dtype=torch.long)), 0), \
               inputLen, torch.tensor(self.labelOutput[index]).squeeze()

    def __len__(self):
        return len(self.labelOutput)


if __name__ == "__main__":
    dataset = MOSIDataset(gc.data_path, src="csd", save=False)

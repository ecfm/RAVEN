from consts import global_consts as gc

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformer import Models

import layer

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.normcovarep = nn.BatchNorm2d(gc.padding_len, track_running_stats=False)
        self.dropcovarep = nn.Dropout(p=gc.dropProb)
        self.fc_rszcCovarep = nn.Linear(gc.covarepDim, gc.normDim)

        # self.covarepTransformer = Models.TransformerEncoder(gc.shift_padding_len, gc.normDim,
        #                                                     gc.n_layers, gc.n_head, gc.normDim, gc.normDim,
        #                                                     gc.normDim, gc.ff_iner_dim)
        self.covarepTemporalTransformer = Models.TransformerEncoder(gc.padding_len, gc.covarepDim,
                                                                    gc.n_layers, gc.n_head, gc.covarepDim, gc.covarepDim,
                                                                    gc.covarepDim, gc.ff_iner_dim)

        self.wordTemporalTransformer = Models.TransformerEncoder(gc.padding_len, gc.wordDim, gc.n_layers_large,
                                                                 gc.n_head_large, gc.wordDim, gc.wordDim,
                                                                 gc.wordDim, gc.ff_iner_dim)

        # self.normFacet = nn.BatchNorm2d(gc.padding_len, track_running_stats=False)
        self.dropFacet = nn.Dropout(p=gc.dropProb)
        self.fc_rszFacet = nn.Linear(gc.facetDim, gc.normDim)
        # self.facetTransformer = Models.TransformerEncoder(gc.shift_padding_len, gc.normDim,
        #                           gc.n_layers, gc.n_head, gc.normDim, gc.normDim,
        #                                                   gc.normDim, gc.ff_iner_dim)
        self.facetTemporalTransformer = Models.TransformerEncoder(gc.padding_len, gc.facetDim,
                                                                  gc.n_layers, gc.n_head, gc.facetDim, gc.facetDim,
                                                                  gc.facetDim, gc.ff_iner_dim)
        multiModelDim = gc.wordDim + gc.covarepDim + gc.facetDim
        self.multiModelTransformer = Models.TransformerEncoder(gc.padding_len, multiModelDim, gc.n_layers_large,
                                                               gc.n_head_large, multiModelDim, multiModelDim,
                                                               multiModelDim, gc.ff_iner_dim)
        self.finalW = nn.Linear(gc.padding_len * multiModelDim, 1)

    def forward(self, words, covarep, facet, inputLens):
        batch = covarep.size()[0]
        covarepState_pos = torch.LongTensor(np.array([[j+1 if torch.abs(covarep[i][j]).sum() > 0 else 0
                                                       for j in range(gc.padding_len)]
                                                      for i in range(batch)])).to(gc.device)
        covarepState = self.covarepTemporalTransformer(covarep, covarepState_pos)[0]


        facetState_pos = torch.LongTensor(np.array([[j + 1 if torch.abs(facet[i][j]).sum() > 0 else 0
                                                     for j in range(gc.padding_len)]
                                                    for i in range(batch)])).to(gc.device)
        facetState = self.facetTemporalTransformer(facet, facetState_pos)[0]

        word_pos = torch.LongTensor(np.array([[i + 1 if i < len else 0 for i in range(gc.padding_len)]
                                              for len in inputLens])).to(gc.device)
        wordState = self.wordTemporalTransformer(words, word_pos)[0]

        multiModelState = torch.cat([covarepState, facetState, wordState], 2)
        multiModelState_pos = torch.LongTensor(np.array([[j + 1 if torch.abs(multiModelState[i][j]).sum() > 0 else 0
                                                     for j in range(gc.padding_len)]
                                                    for i in range(batch)])).to(gc.device)
        multiModelState = self.multiModelTransformer(multiModelState, multiModelState_pos)[0].view(batch, -1)

        return self.finalW(multiModelState).squeeze()

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

        self.covarepTransformer = Models.TransformerEncoder(gc.shift_padding_len, gc.normDim,
                                                            gc.n_layers, gc.n_head, gc.normDim, gc.normDim,
                                                            gc.normDim, gc.ff_iner_dim)
        self.covarepTemporalTransformer = Models.TransformerEncoder(gc.padding_len, gc.normDim,
                                                                    gc.n_layers, gc.n_head, gc.normDim, gc.normDim,
                                                                    gc.normDim, gc.ff_iner_dim)

        self.wordTemporalTransformer = Models.TransformerEncoder(gc.padding_len, gc.wordDim, gc.n_layers_large,
                                                                 gc.n_head_large, gc.wordDim, gc.wordDim,
                                                                 gc.wordDim, gc.ff_iner_dim)

        # self.normFacet = nn.BatchNorm2d(gc.padding_len, track_running_stats=False)
        self.dropFacet = nn.Dropout(p=gc.dropProb)
        self.fc_rszFacet = nn.Linear(gc.facetDim, gc.normDim)
        self.facetTransformer = Models.TransformerEncoder(gc.shift_padding_len, gc.normDim,
                                  gc.n_layers, gc.n_head, gc.normDim, gc.normDim,
                                                          gc.normDim, gc.ff_iner_dim)
        self.facetTemporalTransformer = Models.TransformerEncoder(gc.padding_len, gc.normDim,
                                                                  gc.n_layers, gc.n_head, gc.normDim, gc.normDim,
                                                                  gc.normDim, gc.ff_iner_dim)
        multiModelDim = gc.wordDim + 2 * gc.normDim
        self.multiModelTransformer = Models.TransformerEncoder(gc.padding_len, multiModelDim, gc.n_layers_large,
                                                               gc.n_head_large, multiModelDim, multiModelDim,
                                                               multiModelDim, gc.ff_iner_dim)
        self.finalW = nn.Linear(gc.padding_len * multiModelDim, 1)

    def forward(self, words, covarep, covarepLens, facet, facetLens, inputLens):
        batch = covarep.size()[0]
        inputs = None
        covarep = self.normcovarep(covarep)
        covarepInput = self.fc_rszcCovarep(self.dropcovarep(covarep))
        covarepFlat = covarepInput.data.contiguous().view(-1, gc.shift_padding_len, gc.normDim)
        # covarepLensFlat.shape = batch * gc.padding_len
        covarepLensFlat = covarepLens.data.contiguous().view(-1)
        coverp_pos = torch.LongTensor(np.array([[i+1 if i < len else 0 for i in range(gc.shift_padding_len)]
                                                for len in covarepLensFlat])).to(gc.device)
        # output.shape = [batch * gc.padding_len, gc.shift_padding_len, gc.normDim]
        output = self.covarepTransformer(covarepFlat, coverp_pos)[0]
        # output.shape = [batch * gc.padding_len, gc.shift_padding_len + 1, gc.normDim]
        output = torch.cat([torch.zeros(batch * gc.padding_len, 1, gc.normDim).to(gc.device), output], 1)
        # covarepSelector.shape = [batch * gc.padding_len, 1, gc.shift_padding_len + 1]
        covarepSelector = torch.zeros(batch * gc.padding_len, 1, gc.shift_padding_len + 1).to(gc.device).scatter_(2, covarepLensFlat.unsqueeze(1).unsqueeze(1), 1.0)
        # covarepState.shape = [batch, gc.padding_len, gc.normDim]
        covarepState = torch.matmul(covarepSelector, output).squeeze().view(batch, gc.padding_len, gc.normDim)
        covarepState_pos = torch.LongTensor(np.array([[j+1 if torch.abs(covarepState[i][j]).sum() > 0 else 0
                                                       for j in range(gc.padding_len)]
                                                      for i in range(batch)])).to(gc.device)
        covarepState = self.covarepTemporalTransformer(covarepState, covarepState_pos)[0]


        #facet = self.normFacet(facet)
        facetInput = self.fc_rszFacet(self.dropFacet(facet))
        facetFlat = facetInput.data.contiguous().view(-1, gc.shift_padding_len, gc.normDim)
        facetLensFlat = facetLens.data.contiguous().view(-1)
        facet_pos = torch.LongTensor(np.array([[i + 1 if i < len else 0 for i in range(gc.shift_padding_len)]
                                                for len in facetLensFlat])).to(gc.device)
        output = self.facetTransformer(facetFlat, facet_pos)[0]
        output = torch.cat([torch.zeros(batch * gc.padding_len, 1, gc.normDim).to(gc.device), output], 1)
        facetSelector = torch.zeros(batch * gc.padding_len, 1, gc.shift_padding_len + 1).to(gc.device).scatter_(2, facetLensFlat.unsqueeze(1).unsqueeze(1), 1.0)
        # facetState.shape = [batch, gc.padding_len, gc.normDim]
        facetState = torch.matmul(facetSelector, output).view(batch, gc.padding_len, gc.normDim)
        facetState_pos = torch.LongTensor(np.array([[j + 1 if torch.abs(facetState[i][j]).sum() > 0 else 0
                                                     for j in range(gc.padding_len)]
                                                    for i in range(batch)])).to(gc.device)
        facetState = self.covarepTemporalTransformer(facetState, facetState_pos)[0]

        word_pos = torch.LongTensor(np.array([[i + 1 if i < len else 0 for i in range(gc.padding_len)]
                                              for len in inputLens])).to(gc.device)
        wordState = self.wordTemporalTransformer(words, word_pos)[0]

        multiModelState = torch.cat([covarepState, facetState, wordState], 2)
        multiModelState_pos = torch.LongTensor(np.array([[j + 1 if torch.abs(multiModelState[i][j]).sum() > 0 else 0
                                                     for j in range(gc.padding_len)]
                                                    for i in range(batch)])).to(gc.device)
        multiModelState = self.multiModelTransformer(multiModelState, multiModelState_pos)[0].view(batch, -1)

        return self.finalW(multiModelState).squeeze()

# -*- coding: utf-8 -*-

import torch
import torch.nn as nn

# This is an improved version over https://github.com/kmkurn/pytorch-crf and is faster.

class CRF(nn.Module):
    def __init__(self, M, batch_first = False):
        if M <= 0:
            raise ValueError(f'invalid number of tags: {M}')
        super().__init__()
        self.M = M
        self.batch_first = batch_first
        # parameters
        self.transitions = nn.Parameter(torch.zeros(M, M))
        self.start_transitions = nn.Parameter(torch.zeros(M), requires_grad=False)
        self.end_transitions = nn.Parameter(torch.zeros(M), requires_grad=False)
        
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.zeros_(self.transitions)
        nn.init.zeros_(self.start_transitions)
        nn.init.zeros_(self.end_transitions)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(M={self.M})'

    def forward(self, emissions, tags, weight=None, mask=None, reduction='sum') -> torch.Tensor:
        # if batch_first: emissions: [B,T,M]; tags: [B,T]; weight & mask: [B,T]
        # if not batch_first: emissions: [T,B,M]; tags: [T,B]; weight & mask: [T,B]
        # reduction: none|sum|mean|token_mean
        self._validate(emissions, tags=tags, mask=mask)
        if reduction not in ('none', 'sum', 'mean', 'token_mean'):
            raise ValueError(f'invalid reduction: {reduction}')
        if weight is None:
            weight = torch.ones_like(tags, device=emissions.device).float()
        if mask is None:
            mask = torch.ones_like(tags, device=emissions.device).int()

        if self.batch_first:
            emissions = emissions.transpose(0, 1)
            tags = tags.transpose(0, 1)
            mask = mask.transpose(0, 1)
            weight = weight.transpose(0, 1)
            
        score = self._compute_score(emissions, tags, weight, mask) # -> [B,]
        log_Z = self._compute_normalizer(emissions, weight, mask) # -> [B,]
        llh = score - log_Z # -> [B,]
        nllh = - llh
        
        if reduction == 'none':
            return nllh
        elif reduction == 'sum':
            return nllh.sum()
        elif reduction == 'mean':
            return nllh.mean()
        else:
            assert reduction == 'token_mean'
            return  nllh.sum() / mask.float().sum()
    

    def decode(self, emissions, weight=None, mask=None):
        # if batch_first: emissions: [B,T,M]; tags: [B,T]; weight & mask: [B,T]
        # if not batch_first: emissions: [T,B,M]; tags: [T,B]; weight & mask: [T,B]
        self._validate(emissions, mask=mask)
        if weight is None:
            weight = emissions.new_ones(emissions.shape[:2], device=emissions.device).float()
        if mask is None:
            mask = emissions.new_ones(emissions.shape[:2], device=emissions.device).int()

        if self.batch_first:
            emissions = emissions.transpose(0, 1)
            mask = mask.transpose(0, 1)
            weight = weight.transpose(0, 1)

        return self._viterbi_decode(emissions, weight, mask)

    def _validate(self, emissions, tags=None, mask=None) -> None:
        if emissions.dim() != 3:
            raise ValueError(f'emissions must have dimension of 3, got {emissions.dim()}')
        if emissions.size(2) != self.M:
            raise ValueError(
                f'expected last dimension of emissions is {self.M}, '
                f'got {emissions.size(2)}')

        if tags is not None:
            if emissions.shape[:2] != tags.shape:
                raise ValueError(
                    'the first two dimensions of emissions and tags must match, '
                    f'got {tuple(emissions.shape[:2])} and {tuple(tags.shape)}')

        if mask is not None:
            if emissions.shape[:2] != mask.shape:
                raise ValueError(
                    'the first two dimensions of emissions and mask must match, '
                    f'got {tuple(emissions.shape[:2])} and {tuple(mask.shape)}')
            no_empty_seq = not self.batch_first and mask[0].all()
            no_empty_seq_bf = self.batch_first and mask[:, 0].all()
            if not no_empty_seq and not no_empty_seq_bf:
                raise ValueError('mask of the first timestep must all be on')


    def _compute_score(self, emissions, tags, weight, mask) -> torch.Tensor:
        # emissions: [T,B,M], tags: [T,B],weight &  mask: [T,B]
        assert emissions.dim() == 3 and tags.dim() == 2
        assert emissions.shape[:2] == tags.shape
        assert emissions.size(2) == self.M
        assert mask.shape == tags.shape
        assert mask[0].all()

        T, B = tags.shape
        tags = torch.where(mask, tags, 1)
        mask = mask.float()

        # Start transition score
        score = self.start_transitions[tags[0]] # [B,]
        
        # Transition scores
        score_transitions = self.transitions[tags[:-1], tags[1:]] * mask[1:] / weight[:-1] # [T-1,B] * [T-1,B] / [T-1,B] -> [T-1,B]
        score += score_transitions.sum(dim=0) # -> [B,]

        # Emission scores
        score_emissions = torch.gather(emissions, index=tags.unsqueeze(-1), dim=-1).squeeze(-1) * mask # [T,B,M] -gather-> [T,B,1] -unsq-> [T,B]
        score += score_emissions.sum(dim=0) # -> [B,]
        
        # End transition score
        seq_ends = mask.long().sum(dim=0) - 1 # -> [B,]
        last_tags = tags[seq_ends, torch.arange(B)] # -> [B,]
        score += self.end_transitions[last_tags] # -> [B,]

        return score

    def _compute_normalizer(self, emissions, weight, mask) -> torch.Tensor:
        # emissions: [T,B,M], weight & mask: [T,B]
        assert emissions.dim() == 3 and mask.dim() == 2
        assert emissions.shape[:2] == mask.shape
        assert emissions.size(2) == self.M
        assert mask[0].all()

        # Start transition and emission score, score[i][j] = score at tag-j of sample i
        score = self.start_transitions + emissions[0] # -> [B,M]
        
        # Transition and Emission score at each step
        T, B = mask.size()
        transitions_weighted = self.transitions[None,None,...].repeat(T-1,B,1,1) / weight[:-1][...,None,None] # -> [T-1,B,M,M]
        score_steps = transitions_weighted + emissions[1:].unsqueeze(-2) # [T-1,B,M,M] + [T-1,B,1,M] -> [T-1,B,M,M]
        
        # Set the score of paddings log(eye(M)), keep score unchanged at padding:
        score_steps[~mask[1:]] = torch.log(torch.eye(self.M, device=score_steps.device)) # [M,M]

        for sc_step in score_steps: # steps (T - 1)
            score = score.unsqueeze(-1) + sc_step # [B,M,1] + [B,M,M] -> [B,M,M]
            score = torch.logsumexp(score, dim=1) # -> [B,M]
            
        # End transition score
        score = score + self.end_transitions  # [B,M] + [M,] -> [B,M]
        
        # Sum (log-sum-exp) over all possible tags
        return torch.logsumexp(score, dim=1) # [B,]

    def _viterbi_decode(self, emissions, weight, mask):
        # emissions: [T,B,M], mask: [T,B]
        assert emissions.dim() == 3 and mask.dim() == 2
        assert emissions.shape[:2] == mask.shape
        assert emissions.size(2) == self.M
        assert mask[0].all()

        T, B = mask.shape
        # Start transition and first emission, score[i][j] = score of best tag sequence ends with tag-j of sample i
        score = self.start_transitions + emissions[0] # [B,M]
        
        # Transition and Emission score at each step
        T, B = mask.size()
        transitions_weighted = self.transitions[None,None,...].repeat(T-1,B,1,1) / weight[:-1][...,None,None] # -> [T-1,B,M,M]
        score_steps = transitions_weighted + emissions[1:].unsqueeze(-2) # [T-1,B,M,M] + [T-1,B,1,M] -> [T-1,B,M,M]
        score_steps[~mask[1:]] = torch.log(torch.eye(self.M, device=score_steps.device)) # [M,M]
        
        # Viterbi algorithm: compute the score of the best tag sequence for every possible next tag
        history = [] # the best tags candidate transitioned from
        for sc_step in score_steps: # steps (T - 1)
            score = score.unsqueeze(-1) + sc_step # [B,M,1] + [B,M,M] -> [B,M,M]

            # Find the maximum score over all possible current tag
            score, indices = score.max(dim=1) # [B,M]
            history.append(indices)

        # End transition score
        score += self.end_transitions # [B,M]

        # Compute the best path for each sample for the last timestep
        _, best_last_tags = score.max(dim=-1) # -> [B,M] -> _,[B,]
        best_tags_seq = [best_last_tags]
    
        # We trace back where the best last tag comes from, append that to our best tag
        # sequence, and trace it back again, and so on
        for hist in reversed(history):
            best_last_tags = hist[torch.arange(B), best_tags_seq[-1]] # hist: [B,M], last_tags: [B,]
            best_tags_seq.append(best_last_tags) # -> T * [B,] (reversed)
    
        # Reverse the order because we start from the last timestep
        best_tags_seq.reverse() # T * [B,] -> T * [B,]
        best_tags_seq = torch.stack(best_tags_seq) # -> [T,B]
        
        
        seq_lengths = mask.long().sum(dim=0) # [B,]
        best_tags_list = []
        for iseq, lenseq in enumerate(seq_lengths):
            tags = list(best_tags_seq[:lenseq, iseq].cpu())
            best_tags_list.append(tags)

        return best_tags_list


    

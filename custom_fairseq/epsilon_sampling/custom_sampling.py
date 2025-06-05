""" Modified fairseq code to include epsilon sampling  """
import torch
from fairseq import search

from typing import List, Optional

from torch import Tensor

class CustomSampling(search.Sampling):

    def __init__(self, tgt_dict, sampling_topk=-1, sampling_topp=-1.0, sampling_epsilon=-1.0):
        super().__init__(tgt_dict, sampling_topk, sampling_topp)
        self.sampling_epsilon = float(sampling_epsilon)
        self.min_tokens_to_keep = 1
        self.filter_value = 0

    def _sample_epsilon(self, lprobs):
        """ Perform epsilon sampling

        Args:
            lprobs: (bsz x input_beam_size x vocab_size)
                the model's log-probabilities over the vocabulary at the current step

        Return: A tuple of (trimed_probs, truncated_indices) where:
            trimed_probs: (bsz x input_beam_size x ?)
                the model's probabilities over the elements selected to sample from. The
                width of the third dimension is determined by epsilon.
            truncated_indices: (bsz x input_beam_size x ?)
                the indices of the chosen elements. """

        #Get real probabilities by raising e to the power of the logprobs
        probs = lprobs.exp_()
        
        # sort the last dimension (vocab dimension) in descending order
        sorted_probs, sorted_indices = probs.sort(descending=True)

        #Create mask, where 1 shows that the token should be discarded
        indices_to_remove = sorted_probs < self.sampling_epsilon
        
        # safety check
        # if min_tokens_to_keep is 1, this will be either 0 or 1
        top_k = min(self.min_tokens_to_keep, sorted_probs.size(-1))
        #Include the self.min_tokens_to_keep tokens with the highest prob, if they fall below the epsilon threshold
        indices_to_remove = indices_to_remove & (sorted_probs < torch.topk(sorted_probs, top_k)[0])

        #Need to find the maximum number of tokens at a current beam, in order to have a consisten tensor size
        cumsum_mask = indices_to_remove.cumsum(dim=2)
        #Get the cumulative sum of the mask at the last position, i.e. how many elements that are to be discarded
        last_included = cumsum_mask[:, :, -1:]
        
        # Ensure valid indices by clamping the value between 0 and vocab_size -1
        #last_included.clamp_(0, mask.size()[2] - 1)
        
        #Since we mask positively the tokens to discard, max_dim is vocab_size - the lowest number of tokens discarded
        max_dim = cumsum_mask.size()[2] - last_included.min()
   
        truncated_mask = indices_to_remove[:, :, : max_dim + 1]
        truncated_probs = sorted_probs[:, :, : max_dim + 1]
        truncated_indices = sorted_indices[:, :, : max_dim + 1]


        # Set probs under cutoff to filter_value (0) to give them 0 proba
        trimmed_probs = truncated_probs.masked_fill_(truncated_mask, self.filter_value)

        return trimmed_probs, truncated_indices

    
    @torch.jit.export
    def step(
        self,
        step: int,
        lprobs,
        scores,
        prev_output_tokens: Optional[Tensor] = None,
        original_batch_idxs: Optional[Tensor] = None,
    ):
        bsz, beam_size, vocab_size = lprobs.size()

        if step == 0:
            # at the first step all hypotheses are equally likely, so use
            # only the first beam
            lprobs = lprobs[:, ::beam_size, :].contiguous()

        if self.sampling_topp > 0:
            # only sample from the smallest set of words whose cumulative probability mass exceeds p
            probs, top_indices = self._sample_topp(lprobs)
        elif self.sampling_topk > 0:
            # only sample from top-k candidates
            lprobs, top_indices = lprobs.topk(self.sampling_topk)
            probs = lprobs.exp_()
            
        #MY addition
        elif self.sampling_epsilon > 0:
            probs, top_indices = self._sample_epsilon(lprobs)
            
        else:
            probs = lprobs.exp_()

            # dummy data to be consistent with true branch for type check
            top_indices = torch.empty(0).to(probs)
        # sample
        if step == 0:
            indices_buf = torch.multinomial(
                probs.view(bsz, -1),
                beam_size,
                replacement=True,
            ).view(bsz, beam_size)
        else:
            indices_buf = torch.multinomial(
                probs.view(bsz * beam_size, -1),
                1,
                replacement=True,
            ).view(bsz, beam_size)

        if step == 0:
            # expand to beam size
            probs = probs.expand(bsz, beam_size, -1)

        # gather scores
        scores_buf = torch.gather(probs, dim=2, index=indices_buf.unsqueeze(-1))
        scores_buf = scores_buf.log_().view(bsz, -1)

        # remap indices if using top-k or top-P sampling
        if self.sampling_topk > 0 or self.sampling_topp > 0 or self.sampling_epsilon > 0:
            indices_buf = torch.gather(
                top_indices.expand(bsz, beam_size, -1),
                dim=2,
                index=indices_buf.unsqueeze(-1),
            ).squeeze(2)

        if step == 0:
            beams_buf = indices_buf.new_zeros(bsz, beam_size)
        else:
            beams_buf = torch.arange(0, beam_size).to(indices_buf).repeat(bsz, 1)
            # make scores cumulative
            scores_buf.add_(
                torch.gather(scores[:, :, step - 1], dim=1, index=beams_buf)
            )

        return scores_buf, indices_buf, beams_buf
        

        

        
        
        

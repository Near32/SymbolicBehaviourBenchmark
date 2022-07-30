import numpy as np
from gym.spaces import MultiDiscrete


class CommunicationChannel(MultiDiscrete):
    """
    The communication channel action/observation space is built on top of a multi-discrete action space with a series of 
    `max_sentence_length` discrete action spaces that each allow `vocab_size+1` different actions.
    - The multi-discrete action space consists of a series of discrete action spaces with different number of actions in eachs
    - It is useful to represent game controllers or keyboards where each key can be represented as a discrete action space
    - It is parametrized by passing an array of positive integers specifying number of actions for each discrete action space
    Note: a value of 0 always represents the EoS token. It is a grounded symbol.
        therefore the effective vocabulary size (for grounded and ungrounded symbols) is `1+vocab_size`.
        Any use of the EoS symbol for token i-th will transform the remaining max_sentence_length-i token(s) into EoS symbols.
    """
    def __init__(self, max_sentence_length, vocab_size):
        """
        max_sentence_length: int, maximum number of token/symbol in the sentence.
        vocab_size: int, size of the ungrounded vocabulary.
        """
        assert (max_sentence_length > 0) and (vocab_size > 0)
        self.max_sentence_length = max_sentence_length
        self.vocab_size = vocab_size
        super(CommunicationChannel, self).__init__(nvec=[self.vocab_size+1]*self.max_sentence_length)

    def sample(self):
        output = self.np_random.random_sample(self.max_sentence_length)
        output *= self.nvec
        output = output.astype(self.dtype)
        # Regularise the use of EoS symbol:
        make_eos = False
        for idx, o in enumertate(output):
            if make_eos:    
                output[idx] = 0
                continue
            if o==0:
                make_eos = True
        
        return np.expand_dims(output, axis=0)

    def contains(self, x):
        if isinstance(x, list):
            x = np.array(x)  # Promote list to array for contains check
        # if nvec is uint32 and space dtype is uint32, then 0 <= x < self.nvec guarantees that x
        # is within correct bounds for space dtype (even though x does not have to be unsigned)
        return x.shape == self.shape and (0 <= x).all() and (x < self.nvec).all()

    def to_jsonable(self, sample_n):
        return [sample.tolist() for sample in sample_n]

    def from_jsonable(self, sample_n):
        return np.array(sample_n)

    def __repr__(self):
        return "CommunicationChannel({})".format(self.nvec)

    def __eq__(self, other):
        return isinstance(other, CommunicationChannel) and np.all(self.nvec == other.nvec)

import numpy as np 

def STR2BT(sentences, max_sentence_length=0):
    '''
    :param sentences: List[str] or str to be converted to ByteTensor. 
    :max_sentence_length: int, default max length of sentences, unless a longer sentence is provided.
    '''
    if isinstance(sentences, str):
        sentences = [sentences]
    btss = []
    for s in sentences:
        bts = np.array(list(bytes(s, 'utf-8')))
        if max_sentence_length < bts.shape[0]:  max_sentence_length = bts.shape[0]
        btss.append(bts)
    ret = np.zeros((len(btss), max_sentence_length), dtype=np.uint8)
    for bts_idx, bts, in enumerate(btss):
        ret[bts_idx, :bts.shape[0]] = bts
    return ret

def BT2STR(bt):
    '''
    :param bt: ByteTensor to be converted to List[str]. 
    :return: List[str] 
    '''
    sentences = []
    for idx in range(bt.shape[0]):
        sentence = "".join(map(chr,bt[idx].tolist())).replace('\x00','')
        sentences.append(sentence)
    return sentences


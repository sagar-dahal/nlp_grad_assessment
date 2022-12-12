import conllu
import numpy as np


def get_sentences(file_path):
    with open(file_path, encoding='utf-8') as f:
        data = [sent for sent in conllu.parse_incr(f)]
    return data


def get_leftmost_child(id, sent):
    for i in range(id):
        if sent[i]['head'] == id:
            return sent[i]['id']
    return None


def get_rightmost_child(id, sent):
    for i in range(len(sent)-id):
        if sent[-i-1]['head'] == id:
            return sent[-i-1]['id']
    return None


def has_child(word, buffer):
    for w in buffer:
        if w['head'] == word['id']:
            return True
    return False


def correct_action_unlabelled(stack, buffer):
    left_arc = np.array([[1,0,0]])
    right_arc = np.array([[0,1,0]])
    shift = np.array([[0,0,1]])
    if len(stack) < 3 and len(buffer)>0:
        return shift
    elif len(stack) == 2 and len(buffer) == 0:
        return right_arc
    elif stack[-1]['head'] == stack[-2]['id']:
        if has_child(stack[-1], buffer):
            return shift
        else:
            return right_arc
    elif stack[-2]['head'] == stack[-1]['id']:
        if has_child(stack[-2], buffer):
            return shift
        else:
            return left_arc
    elif len(buffer)>0:
        return shift

def correct_action_labelled(stack, buffer, left_labels, right_labels, shift_label):
    left_arc = np.array([[1,0,0]])
    right_arc = np.array([[0,1,0]])
    shift = np.array([[0,0,1]])
    if len(stack) < 3 and len(buffer)>0:
        return shift, shift_label
    elif len(stack) == 2 and len(buffer) == 0:
        return right_arc, right_labels[stack[1]['deprel']]
    elif stack[-1]['head'] == stack[-2]['id']:
        if has_child(stack[-1], buffer):
            return shift, shift_label
        else:
            return right_arc, right_labels[stack[-1]['deprel']]
    elif stack[-2]['head'] == stack[-1]['id']:
        if has_child(stack[-2], buffer):
            return shift, shift_label
        else:
            return left_arc, left_labels[stack[-2]['deprel']]
    elif len(buffer)>0:
        return shift, shift_label    


def get_input(stack, buffer, sentence, sent_index, children, word_emb, pos_emb):
    if len(stack) == 1:
        s1 = s1_POS = 'NULL'
        s2 = stack[0]
        s2_POS = 'NULL'
        lc_s1 = lc_s1_POS = 'NULL'
        lc_s2 = lc_s2_POS = 'NULL'
        rc_s1 = rc_s1_POS = 'NULL'
        rc_s2 = rc_s2_POS = 'NULL'
    elif len(stack) == 2:
        s1 = stack[1]
        s1_POS = s1['upos']
        s2 = stack[0]
        s2_POS = 'NULL'

        if type(s1['id']) is tuple:
            return 'break'

        lc_s1 = children[sent_index][s1['id']-1][0]
        if lc_s1 == 'NULL':
            lc_s1_POS = 'NULL'
        else:
            lc_s1 = sentence[children[sent_index][s1['id']-1][0]-1]
            lc_s1_POS = lc_s1['upos']

        rc_s1 = children[sent_index][s1['id']-1][1]
        if rc_s1 == 'NULL':
            rc_s1_POS = 'NULL'
        else:
            rc_s1 = sentence[children[sent_index][s1['id']-1][1]-1]
            rc_s1_POS = rc_s1['upos']

        lc_s2 = lc_s2_POS = 'NULL'
        rc_s2 = rc_s2_POS = 'NULL'
    else:
        s1 = stack[-1]
        s1_POS = s1['upos']
        s2 = stack[-2]
        s2_POS = s2['upos']
        if type(s1['id']) is tuple:
            return 'break'
        lc_s1 = children[sent_index][s1['id']-1][0]
        if lc_s1 == 'NULL':
            lc_s1_POS = 'NULL'
        else:
            lc_s1 = sentence[children[sent_index][s1['id']-1][0]-1]
            lc_s1_POS = lc_s1['upos']
        
        rc_s1 = children[sent_index][s1['id']-1][1]
        if rc_s1 == 'NULL':
            rc_s1_POS = 'NULL'
        else:
            rc_s1 = sentence[children[sent_index][s1['id']-1][1]-1]
            rc_s1_POS = rc_s1['upos']
        
        if type(s2['id']) is tuple:
            return 'break'
        lc_s2 = children[sent_index][s2['id']-1][0]
        if lc_s2 == 'NULL':
            lc_s2_POS = 'NULL'
        else:
            lc_s2 = sentence[children[sent_index][s2['id']-1][0]-1]
            lc_s2_POS = lc_s2['upos']

        rc_s2 = children[sent_index][s2['id']-1][1]
        if rc_s2 == 'NULL':
            rc_s2_POS = 'NULL'
        else:
            rc_s2 = sentence[children[sent_index][s2['id']-1][1]-1]
            rc_s2_POS = rc_s2['upos']
    if len(buffer) > 0:
        b1 = buffer[0]
        b1_POS = buffer[0]['upos']
    else:
        b1 = 'NULL'
        b1_POS = 'NULL'

    s1 = word_emb[str(s1)]
    s2 = word_emb[str(s2)]
    lc_s1 = word_emb[str(lc_s1)]
    lc_s2 = word_emb[str(lc_s2)]
    rc_s1 = word_emb[str(rc_s1)]
    rc_s2 = word_emb[str(rc_s2)]
    s1_POS = pos_emb[s1_POS]
    s2_POS = pos_emb[s2_POS]
    lc_s1_POS = pos_emb[lc_s1_POS]
    lc_s2_POS = pos_emb[lc_s2_POS]
    rc_s1_POS = pos_emb[rc_s1_POS]
    rc_s2_POS = pos_emb[rc_s2_POS]
    b1 = word_emb[str(b1)]
    b1_POS = pos_emb[str(b1_POS)]

    return np.concatenate((s1,s2,lc_s1,lc_s2,rc_s1,rc_s2,b1,s1_POS,s2_POS,lc_s1_POS,lc_s2_POS,rc_s1_POS,rc_s2_POS,b1_POS))
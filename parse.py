import numpy as np
from nltk import Tree
import nltk
import os
import re

def build_tree(break_probs, layer, start, end ,threshold=0.8):
    brackets = set()
    layer_probs = break_probs[layer,start:end]
    min_layer = 2
    if end - start > 1:
        point = np.argmin(layer_probs)
        #print(layer, start, end)
        if layer_probs[point] > threshold:
            if layer == min_layer:
                brackets.add((start,end+1))
                return brackets
            return build_tree(break_probs, max(layer-1,min_layer), start, end, threshold)
    
        for span in (layer_probs[:point],layer_probs[point+1:]):
            span_size = span.shape[0]
            if span_size > 0:
                if np.min(span) > 0.7:
                    node_brac = build_tree(break_probs, max(layer-1,min_layer), start, start+span_size, threshold)
                else:
                    node_brac = build_tree(break_probs, layer, start, start+span_size)
                brackets.add((start, start+span_size+1))
                brackets.update(node_brac)
            start += span_size + 1
        return brackets

    else:
        brackets.add((start,start+2))
        return brackets

def word2tree(start, end, text):
    tree = '( '
    for idx in range(start, end):
        s = '( %s) ' % (text[idx])
        tree = tree + s
    tree = tree + ')'
    return Tree.fromstring(tree)


def dump_tree(break_probs, layer, start, end , text, threshold=0.8):
    layer_probs = break_probs[layer,start:end]
    min_layer = 2
    tree = Tree.fromstring('()')
    if end - start > 1:
        point = np.argmin(layer_probs)
        if layer_probs[point] > threshold:
            if layer == min_layer:
                tree = word2tree(start, end+1, text)
                return tree
            return dump_tree(break_probs, max(layer-1,min_layer), start, end, text, threshold)
    
        for span in (layer_probs[:point],layer_probs[point+1:]):
            span_size = span.shape[0]
            if span_size > 0:
                if np.min(span) > 0.7:
                    node_tree = dump_tree(break_probs, max(layer-1,min_layer), start, start+span_size, text, threshold)
                else:
                    node_tree = dump_tree(break_probs, layer, start, start+span_size, text, threshold)
                tree.insert(len(tree)+1,node_tree)
            else:
                tree.insert(len(tree)+1,word2tree(start, start+1, text))
            start += span_size + 1
        return tree
    elif end - start == 1:
        return word2tree(start, start+2, text)
    else:
        return word2tree(start, start+1, text)


def get_break_prob(break_probs, print_prob=False):
    break_probs = break_probs.detach().cpu().numpy()

    all_b = []
    for l in range(break_probs.shape[0]):
        b = []
        for i in range(break_probs.shape[-1]-1):
            b.append(break_probs[l][i][i+1])
        if print_prob:
            print(b)
        all_b.append(b)
    return np.array(all_b)
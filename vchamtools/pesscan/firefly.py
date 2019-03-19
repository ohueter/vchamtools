#!/usr/bin/env python3

__all__ = ["get_vec_block"]

# fetches last $VEC...$END block from punch file
def get_vec_block(filename):
    vec_block = None
    with open(filename, 'r') as f:
        for line in f:
            if line.strip().upper() == '$VEC':
                vec_block = [line]
                for line in f:
                    vec_block.append(line)
                    if line.strip().upper() == '$END':
                        break
    return vec_block

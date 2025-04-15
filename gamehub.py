import numpy as np
import math
import matplotlib.pyplot as plt

import functions as sudoku

def unique_possibility(base, boxco):
    # sudoku.print_sudoku_style(base)
    n = base.shape[0]
    box = sudoku.extract_groups_vectorized(base, boxco, mode="box", indices=True)
    box = box[box[:,2] == 0]
    X,Y = box[:,0], box[:,1]
    poss  = sudoku.get_pos(base, np.array([X, Y]).T)
    
    if(len(poss) == 0):
        print(f"\033[92mBOX FINISHED !\033[0m: @ {boxco}")
        return base
    poss_all = np.unique(np.concatenate(poss))
    #check if box has only one possible number
    for i in range(len(poss)):
        if len(poss[i]) == 1:
            base[X[i], Y[i]] = poss[i][0]
            # print(f"Box unique: {X[i], Y[i]} -> {poss[i][0]}")

    #recalculate possibilities
    box = sudoku.extract_groups_vectorized(base, boxco, mode="box", indices=True)
    box = box[box[:,2] == 0]
    X,Y = box[:,0], box[:,1]
    poss  = sudoku.get_pos(base, np.array([X, Y]).T)
    if(len(poss) == 0):
        print(f"\033[92mBOX FINISHED !\033[0m: @ {boxco}")
        return base
    poss_all = np.unique(np.concatenate(poss))
    #check if number has only one possible box
    bincount = np.bincount(np.concatenate(poss))
    bincount = np.where(bincount[bincount!=0]==1)[0]
    if len(bincount) != 0:
        val_unique_box = poss_all[bincount]
        for i in val_unique_box:
            for j in range(len(poss)):
                if i in poss[j]:
                    base[X[j], Y[j]] = i
                    # print(f"Box unique: {X[j], Y:[j]} -> {i}")
                    break
    return base



def stepper(base):
    n = base.shape[0]
    boxcos = sudoku.get_box_topleft_coords(n)
    num_zeros = np.count_nonzero(base==0)
    improvement = 0
    while improvement != num_zeros:
        num_zeros = improvement
        for i in range(n):
            base = unique_possibility(base, [boxcos[i]])
            improvement = np.count_nonzero(base==0)
            if improvement == 0:
                break
        sudoku.print_sudoku_style(base)
        print("\n\n")
    
    print("\033[92mSUDOKU FINISHED !\033[0m: finished base:")
    sudoku.print_sudoku_style(base)
    return base

stepper(sudoku.initialization(9))
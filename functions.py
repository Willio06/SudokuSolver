import numpy as np
import math
import matplotlib.pyplot as plt
def print_sudoku_progress(base, progress):
    GRAY = '\033[90m'
    YELLOW = '\033[93m'
    RESET = '\033[0m'
    
    n = base.shape[0]
    block_size = int(math.sqrt(n))

    for i in range(n):
        if i % block_size == 0 and i != 0:
            print('-' * (4 * n + block_size - 8))
        
        row = ''
        for j in range(n):
            if j % block_size == 0 and j != 0:
                row += '| '
            
            val = progress[i, j]
            if val == 0:
                row += f'{GRAY} 0{RESET} '
            elif val != base[i, j]:
                row += f'{YELLOW}{val:2}{RESET} '
            else:
                row += f'{val:2} '
        print(row)
def print_sudoku_style(arr):
    GRAY = '\033[90m'
    RESET = '\033[0m'
    n = arr.shape[0]
    block_size = int(math.sqrt(n))

    for i in range(n):
        # Print horizontal block separator
        if i % block_size == 0 and i != 0:
            print('-' * (4 * n + block_size - 8))
        
        row = ''
        for j in range(n):
            # Print vertical block separator
            if j % block_size == 0 and j != 0:
                row += '| '
            
            val = arr[i, j]
            if val == 0:
                row += f'{GRAY} 0{RESET} '
            else:
                row += f'{val:2} '
        print(row)
def initialization(n):
    if(int(np.sqrt(n))**2 != n):
        raise ValueError("n must be a perfect square")
    base = np.zeros((n,n), dtype=int)
    coos = np.array([(0,0,4),(0,2,3),(0,3,5),(0,3,5),(0,4,9),(0,6,7),(0,7,2),(1,1,8),(1,3,6),(1,7,3),(2,0,7),
            (2,2,9),(2,5,4),(2,8,5),(3,0,1),(3,1,5),(3,2,8),(3,4,6),(4,1,7),(4,2,2),(4,6,8),(4,8,9),
            (5,3,2),(5,4,3),(5,5,8),(5,6,5),(5,7,7),(6,1,1),(6,2,5),(6,3,4),(6,8,2),(7,3,8),(7,5,3),(7,6,4),(7,7,5),
            (8,1,3),(8,5,2),(8,7,8),(8,8,7)])
    base[coos[:,0], coos[:,1]] = coos[:,2]
    return base

def extract_groups_vectorized(arr, coords, mode="box", indices=False):
    n = arr.shape[0]
    b = int(math.sqrt(n))
    coords = np.array(coords)

    def get_indices(i, j):
        if mode == "row":
            return np.full(n, i), np.arange(n)
        elif mode == "column":
            return np.arange(n), np.full(n, j)
        elif mode == "box":
            ti, tj = (i // b) * b, (j // b) * b
            di, dj = np.meshgrid(np.arange(b), np.arange(b), indexing='ij')
            return ti + di.ravel(), tj + dj.ravel()
        else:
            raise ValueError("Invalid mode")

    if indices:
        all_entries = [
            np.stack([*(idxs := get_indices(i, j)), arr[idxs]], axis=1)
            for i, j in coords
        ]
        return np.vstack(all_entries)
    else:
        return np.array([arr[get_indices(i, j)] for i, j in coords])

def get_pos(arr, coos):
    """
    arr: the base game array
    coos: the coordinates of the box to be checked in array
    return: the possible numbers that can be put in the box
    """
    n = arr.shape[0]
    Pos= []
    for i in range(coos.shape[0]):
        pos = np.arange(1, n+1)
        if np.all(arr[coos[i,0], coos[i,1]]==0):
            box = extract_groups_vectorized(arr, [coos[i]], mode="box")
            row = extract_groups_vectorized(arr, [coos[i]], mode="row")
            col = extract_groups_vectorized(arr, [coos[i]], mode="column")
            pos = np.setdiff1d(pos, box)
            pos = np.setdiff1d(pos, row)
            pos = np.setdiff1d(pos, col)
            Pos.append(pos)
        else:
            print("\033[91mWARNING\033[0m: played box is called. check implementation")
            print(f"\t Already played box at any position in {coos}")
            return arr[coos[:,0], coos[:,1]]
    return Pos

def get_box_topleft_coords(n):
    if(int(np.sqrt(n))**2 != n):
        raise ValueError("n must be a perfect square")
    b = int(n**0.5)
    starts = np.arange(0, n, b)
    ii, jj = np.meshgrid(starts, starts, indexing='ij')
    return np.stack([ii.ravel(), jj.ravel()], axis=1)



def check_contradiction(base):
    n = base.shape[0]
    all_boxs = get_box_topleft_coords(n)
    for boxco in all_boxs:
        box = extract_groups_vectorized(base, [boxco], mode="box", indices=True)
        vals = box[:,2]!=0
        filled = box[vals,2]
        xi, yi = box[np.logical_not(vals),0], box[np.logical_not(vals),1]
        poss  = get_pos(base, np.array([xi, yi]).T)
        poss_all = np.unique(np.concatenate(poss))
        to_do = np.setdiff1d(np.arange(1,n+1), filled)
        if not np.array_equal(poss_all, to_do):
            print("\033[91mWARNING\033[0m: contradiction found")
            print(f"\t Missing possible numbers {np.setdiff1d(to_do,poss_all)} in box\n{box[:,2]}")
            return True
        if any(len(a) == 0 for a in poss):
            i =[i for i, a in enumerate(poss) if len(a) == 0][0]
            print("\033[91mWARNING\033[0m: contradiction found")
            print(f"\t No possible numbers at ({xi[i]},{yi[i]}) in box\n{box}")
            return True
    return False
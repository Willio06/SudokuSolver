import numpy as np
import math
import matplotlib.pyplot as plt
from collections import defaultdict
class Sudoku(object):
    def __init__(self, base):
        self.base = base
        self.n = base.shape[0]
        self.progress =(np.count_nonzero(self.base)/self.n**2)*100
        self.original = np.copy(base)

        if(int(np.sqrt(self.n))**2 != self.n):
            raise ValueError("n must be a perfect square")
        starts = np.arange(0, self.n, int(self.n**0.5))
        ii, jj = np.meshgrid(starts, starts, indexing='ij')
        self.boxCoos= np.stack([ii.ravel(), jj.ravel()], axis=1)
    def updateProgress(self):
        self.progress =(np.count_nonzero(self.base)/self.n**2)*100
        return self.progress
    def print_sudoku_progress(self):
        GRAY = '\033[90m'
        YELLOW = '\033[93m'
        RESET = '\033[0m'
        base = self.original
        progress = self.base
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
                elif val != self.original[i, j]:
                    row += f'{YELLOW}{val:2}{RESET} '
                else:
                    row += f'{val:2} '
            print(row)
        print("\n")
    def print_sudoku(self):
        GRAY = '\033[90m'
        RESET = '\033[0m'
        arr = self.base
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
        print("\n")
    def extract_groups_vectorized(self, coords, mode="box", indices=False):
        arr = self.base
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
    def get_pos(self, coos, check_line=False):
        """
        arr: the base game array
        coos: the coordinates of the box to be checked in array
        return: the possible numbers that can be put in the box
        """
        arr = self.base
        coos = np.array(coos)
        coos = self.extract_groups_vectorized(coos, mode="box", indices=True)[:,[0,1]]
        coos = coos[arr[coos[:,0], coos[:,1]]==0]
        n = self.n
        Pos= []
        if(coos.size == 0):
            # print("\033[91mWARNING\033[0m: no empty cells in box")
            return []
        for i in range(coos.shape[0]):
            pos = np.arange(1, n+1)
            box = self.extract_groups_vectorized([coos[i]], mode="box")
            row = self.extract_groups_vectorized([coos[i]], mode="row")
            col = self.extract_groups_vectorized([coos[i]], mode="column")
            pos = np.setdiff1d(pos, box)
            pos = np.setdiff1d(pos, row)
            pos = np.setdiff1d(pos, col)
            Pos.append(pos)
        if check_line:
            removers = self.__get_liners_pos()
            keys = coos[:,0]*n+coos[:,1]
            for i in range(len(keys)):
                Pos[i] = np.setdiff1d(Pos[i], removers[keys[i]])
        return Pos, coos
    def check_contradiction(self):
        base = self.base
        n = base.shape[0]
        all_boxs = self.boxCoos
        for boxco in all_boxs:
            box = self.extract_groups_vectorized([boxco], mode="box", indices=True)
            vals = box[:,2]!=0
            filled = box[vals,2]
            xi, yi = box[np.logical_not(vals),0], box[np.logical_not(vals),1]
            out  = self.get_pos([boxco], check_line=False)
            if(len(out) == 0):
                continue
            else:
                poss, _ = out
            poss_all = np.unique(np.concatenate(poss))
            to_do = np.setdiff1d(np.arange(1,n+1), filled)
            if not np.array_equal(poss_all, to_do):
                # print("\033[91mWARNING\033[0m: contradiction found")
                # print(f"\t Missing possible numbers {np.setdiff1d(to_do,poss_all)} in box\n{box[:,2]}")
                return True
            if any(len(a) == 0 for a in poss):
                i =[i for i, a in enumerate(poss) if len(a) == 0][0]
                # print("\033[91mWARNING\033[0m: contradiction found")
                # print(f"\t No possible numbers at ({xi[i]},{yi[i]}) in box\n{box}")
                return True
        return False
    
    def __get_liners_pos(self):
        def on_same_line(points):
            if len(points) <= 1:
                return True
            xs, ys = zip(*points)
            return all(x == xs[0] for x in xs) or all(y == ys[0] for y in ys)
        base = self.base
        n = base.shape[0]
        all_boxs = self.boxCoos
        poss_remove = [[]]*(n**2)
        for boxco in all_boxs:
            out  = self.get_pos([boxco], check_line=False)
            if(len(out) == 0):
                continue
            else:
                poss, coos = out
            coords = coos
            X = coos[:,0]
            Y = coos[:,1]
            num_to_indices = defaultdict(list)
            for idx, values in enumerate(poss):
                for v in values:
                    num_to_indices[v].append(idx)
            result = []
            for number, indices in num_to_indices.items():
                points = [coords[i] for i in indices]
                if on_same_line(points):
                    result.append((number,X[indices], Y[indices]))
            # check if result not trivial and thus has meaningful information
            for num,X,Y in result:
                row = np.all(X == X[0])
                if len(X) == 1 or len(Y) == 1:
                    continue
                if not row:
                    temp = X.copy()
                    X = Y.copy()
                    Y = temp.copy()
                a= np.arange(X[0] - X[0]%int(np.sqrt(n)), X[0] - X[0]%int(np.sqrt(n)) + int(np.sqrt(n)))
                row_indices = np.delete(a,np.where(a==X[0]))
                rows_indices = list(zip(row_indices,[Y[0]]*(int(np.sqrt(n))-1)))
                if(not row):
                    rows_indices = list(zip([X[0]]*(int(np.sqrt(n))-1),row_indices))
                groups = self.extract_groups_vectorized(rows_indices, mode="row"*int(row)+"column"*int(not row), indices=False)
                patience = 0
                for group in groups:
                    if num in group:
                        patience += 1
                    if patience == int(np.sqrt(n))-1:
                        break
                if patience < int(np.sqrt(n))-1:
                    for i in range(n):
                        if(i in Y):
                            continue
                        poss_remove[(X[0]*n+i)*int(row) + (X[0]+n*i)*int(not row)] = np.unique(np.append(poss_remove[(X[0]*n+i)*int(row) + (X[0]+n*i)*int(not row)], int(num)))
        return poss_remove
    
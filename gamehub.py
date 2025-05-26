import numpy as np
import math
import matplotlib.pyplot as plt
import copy
from functions import Sudoku
from sympy.utilities.iterables import multiset_permutations
class SudokuSolver:
    def __init__(self, base):
        self.sudoku = base
        self.GuessMade=[]
        self.Safes=[]
        self.solved = int(self.sudoku.progress) == 100
    def unique_possibility(self, boxco):
        sudoku = self.sudoku
        base = sudoku.base
        # sudoku.print_sudoku_style(base)
        n = base.shape[0]

        out  = sudoku.get_pos(boxco)
        if(len(out) == 0):
            # print(f"\033[92mBOX FINISHED !\033[0m: @ {boxco}")
            return
        else:
            poss, coos = out
        poss_all = np.unique(np.concatenate(poss))
        #check if box has only one possible number
        for i in range(len(poss)):
            if len(poss[i]) == 1:
                base[coos[i][0], coos[i][1]] = poss[i][0]
                # print(f"Box unique: {X[i], Y[i]} -> {poss[i][0]}")

        #recalculate possibilities
       
        out = sudoku.get_pos(boxco)
        if(len(out) == 0):
            # print(f"\033[92mBOX FINISHED !\033[0m: @ {boxco}")
            return
        else:
            poss, coos = out
        if(len(poss) == 0):
            # print(f"\033[92mBOX FINISHED !\033[0m: @ {boxco}")
            return
        poss_all = np.unique(np.concatenate(poss))
        #check if number has only one possible box
        bincount = np.bincount(np.concatenate(poss))
        bincount = np.where(bincount[bincount!=0]==1)[0]
        if len(bincount) != 0:
            val_unique_box = poss_all[bincount]
            for i in val_unique_box:
                for j in range(len(poss)):
                    if i in poss[j]:
                        base[coos[j][0],coos[j][1]] = i
                        # print(f"Box unique: {X[j], Y:[j]} -> {i}")
                        break
    def GuessOption(self, k=2):
        """
        return first occurence of k boxes with same k possibilities
        """
        sudoku = self.sudoku
        found = False
        while not found:
            for boxco in sudoku.boxCoos:
                out  = sudoku.get_pos([boxco])
                if(len(out) == 0):
                    continue
                else:
                    poss, coos = out
                for i in range(len(poss)):
                    if len(poss[i]) == k:
                        for j in range(len(poss)):
                            if i != j and len(poss[j]) == k:
                                if(np.array_equal(np.sort(poss[i]), np.sort(poss[j]))):
                                    return [coos[i], coos[j]], poss[i]
            k = k + 1
            if(k>5):
                print("\033[91mNO GUESS FOUND !\033[0m: something went wrong")
                return [], []
    def exhaust_unique(self):
        n = self.sudoku.n
        boxcos = self.sudoku.boxCoos
        improvement = 0
        while improvement != self.sudoku.updateProgress():
            improvement = self.sudoku.updateProgress()
            if(improvement-100>=0):
                break
            for i in range(n):
                self.unique_possibility([boxcos[i]])
            # sudoku.print_sudoku_progress()
            # print("\n\n")
        
        # sudoku.print_sudoku_progress()
    def BackTrack(self):
        sudoku = self.sudoku

        self.exhaust_unique()

        if sudoku.progress == 100:
            self.solved = True
            return True  # Puzzle is solved

        cooss, guess = self.GuessOption()
        if not cooss:
            return False  # No options left, dead end

        for perm in multiset_permutations(guess):
            # Save full state, not just base
            saved_base = np.copy(sudoku.base)
            saved_progress = sudoku.progress
            self.GuessMade.append((cooss, perm))

            # Apply the guess
            for (x, y), value in zip(cooss, perm):
                sudoku.base[x, y] = value
            # print(f"Guessing: {cooss} -> {perm}")
            self.exhaust_unique()

            # Check for contradiction after propagation
            if sudoku.check_contradiction():
                # print("\033[91mCONTRADICTION!\033[0m Reverting guess.")
                self.GuessMade.pop()
                sudoku.base = saved_base
                sudoku.progress = saved_progress
                continue

            # Go deeper in the search tree
            if self.BackTrack():
                return True  # Solution found in deeper branch

            # Backtrack from failed deeper guess
            self.GuessMade.pop()
            sudoku.base = saved_base
            sudoku.progress = saved_progress

        return False  # All permutations failed — backtrack




        


def solver(sudoku):
    solverObject = SudokuSolver(sudoku)
    solverObject.exhaust_unique()
    if(sudoku.progress == 100):
        print("\033[92mSUDOKU SOLVED !\033[0m")
        return True
    else:
        solverObject.BackTrack()
        if(sudoku.progress == 100):
            print("\033[92mSUDOKU SOLVED !\033[0m")
            return True
        else:
            print("\033[91mSUDOKU NOT SOLVED !\033[0m")
        return False

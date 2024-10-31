from copy import deepcopy
import os
from opfunu.cec_based.cec2020 import *
from scipy.stats import cauchy


PopSize = 100
DimSize = 10
LB = [-100] * DimSize
UB = [100] * DimSize
TrialRuns = 30
MaxFEs = 1000 * DimSize
curFEs = 0
curIter = 0
Pop = np.zeros((PopSize, DimSize))
FitPop = np.zeros(PopSize)

FuncNum = 0
Length = 10
muF = np.random.uniform(0, 1, Length)
muCr = np.random.uniform(0, 1, Length)


def meanL(arr):
    numer = 0
    denom = 0
    for var in arr:
        numer += var ** 2
        denom += var
    return numer / denom


def Check(indi):
    global LB, UB
    for i in range(DimSize):
        range_width = UB[i] - LB[i]
        if indi[i] > UB[i]:
            n = int((indi[i] - UB[i]) / range_width)
            mirrorRange = (indi[i] - UB[i]) - (n * range_width)
            indi[i] = UB[i] - mirrorRange
        elif indi[i] < LB[i]:
            n = int((LB[i] - indi[i]) / range_width)
            mirrorRange = (LB[i] - indi[i]) - (n * range_width)
            indi[i] = LB[i] + mirrorRange
        else:
            pass
    return indi


def Initialization(func):
    global Pop, FitPop, curFEs, DimSize, muF, muCr, Length
    for i in range(PopSize):
        for j in range(DimSize):
            Pop[i][j] = LB[j] + (UB[j] - LB[j]) * np.random.rand()
        FitPop[i] = func(Pop[i])
    muF = np.random.uniform(0, 1, Length)
    muCr = np.random.uniform(0, 1, Length)


def CFDE(func):
    global Pop, FitPop, curFEs, LB, UB, PopSize, DimSize, muF, muCr
    F_List, Cr_List = [], []
    Seq = list(range(PopSize))
    np.random.shuffle(Seq)
    c = 0.1
    sigma = 0.1

    idxF, idxCr = np.random.randint(Length), np.random.randint(Length)
    for i in range(int(PopSize / 2)):
        idx1, idx2 = Seq[2 * i], Seq[2 * i + 1]
        F = cauchy.rvs(muF[idxF], sigma)
        while True:
            if F > 1:
                F = 1
                break
            elif F < 0:
                F = cauchy.rvs(muF[idxF], sigma)
            break
        Cr = np.clip(np.random.normal(muCr[idxCr], sigma), 0, 1)
        jrand = np.random.randint(DimSize)

        if FitPop[idx1] <= FitPop[idx2]:
            Off = Pop[idx2] + F * (Pop[np.argmin(FitPop)] - Pop[idx2]) + F * (Pop[idx1] - Pop[idx2])
            for j in range(DimSize):
                if np.random.rand() < Cr or j == jrand:
                    pass
                else:
                    Off[j] = Pop[idx2][j]
            Off = Check(Off)
            FitOff = func(Off)
            curFEs += 1
            if FitOff < FitPop[idx2]:
                FitPop[idx2] = FitOff
                Pop[idx2] = deepcopy(Off)
                F_List.append(F)
                Cr_List.append(Cr)
        else:
            Off = Pop[idx1] + F * (Pop[np.argmin(FitPop)] - Pop[idx1]) + F * (Pop[idx2] - Pop[idx1])
            for j in range(DimSize):
                if np.random.rand() < Cr or j == jrand:
                    pass
                else:
                    Off[j] = Pop[idx1][j]
            Off = Check(Off)
            FitOff = func(Off)
            curFEs += 1
            if FitOff < FitPop[idx1]:
                FitPop[idx1] = FitOff
                Pop[idx1] = deepcopy(Off)
                F_List.append(F)
                Cr_List.append(Cr)
    if len(F_List) == 0:
        pass
    else:
        muF[idxF] = (1 - c) * muF[idxF] + c * meanL(F_List)
    if len(Cr_List) == 0:
        pass
    else:
        muCr[idxCr] = (1 - c) * muCr[idxCr] + c * np.mean(Cr_List)


def RunCFDE(func):
    global curFEs, curFEs, TrialRuns, Pop, FitPop, DimSize
    All_Trial_Best = []
    for i in range(TrialRuns):
        Best_list = []
        curFEs = 0
        Initialization(func)
        Best_list.append(min(FitPop))
        while curFEs < MaxFEs:
            CFDE(func)
            Best_list.append(min(FitPop))
        All_Trial_Best.append(Best_list)
    np.savetxt("./CFDE_Data/CEC2020/F" + str(FuncNum) + "_" + str(DimSize) + "D.csv", All_Trial_Best, delimiter=",")


def main(Dim):
    global FuncNum, DimSize, Pop, LB, UB, MaxFEs

    DimSize = Dim
    CEC2020 = [F12020(Dim), F22020(Dim), F32020(Dim), F42020(Dim), F52020(Dim),
               F62020(Dim), F72020(Dim), F82020(Dim), F92020(Dim), F102020(Dim)]
    LB, UB = [-100] * Dim, [100] * Dim
    MaxFEs = 1000 * DimSize
    Pop = np.zeros((PopSize, DimSize))
    for i in range(len(CEC2020)):
        FuncNum = i + 1
        RunCFDE(CEC2020[i].evaluate)


if __name__ == "__main__":
    if os.path.exists('CFDE_Data/CEC2020') == False:
        os.makedirs('CFDE_Data/CEC2020')
    for dim in [10, 20]:
        main(dim)

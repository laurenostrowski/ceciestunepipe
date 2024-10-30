import numpy as np

def str_find(ch,string1):
    return [i for i in range(len(string1)) if string1[i]==ch]

def merge_lists(lsts):
    sets = [set(lst) for lst in lsts if lst]
    merged = True
    while merged:
        merged = False
        results = []
        while sets:
            common, rest = sets[0], sets[1:]
            sets = []
            for x in rest:
                if x.isdisjoint(common):
                    sets.append(x)
                else:
                    merged = True
                    common |= x
            results.append(common)
        sets = results
    out = [list(x) for x in sets]
    return out

def print_unit_counts(quality_labels):
    print('sua_1:',len(np.where(quality_labels=='sua_1')[0]))
    print('sua_2:',len(np.where(quality_labels=='sua_2')[0]))
    print('sua_3:',len(np.where(quality_labels=='sua_2')[0]))
    print('mua_4:',len(np.where(quality_labels=='mua_4')[0]))
    print('noise:',len(np.where(quality_labels=='noise')[0]))
    print('total:',len(np.where(quality_labels=='sua_1')[0])+
         len(np.where(quality_labels=='sua_2')[0])+
         len(np.where(quality_labels=='sua_3')[0])+
         len(np.where(quality_labels=='mua_4')[0])+
         len(np.where(quality_labels=='noise')[0]))
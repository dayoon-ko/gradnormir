pth = "/gallery_louvre/dayoon.ko/research/sds/FlagEmbedding/FlagEmbedding/BGE_M3/"

def load_file(pth):
    with open(pth) as f:
        lines = f.readlines() 
    return lines 

def get_score_list(pth):
    lines = load_file(pth)
    gns = []
    for l in lines:
        try:
            sid = l.index("tensor(") + len("tensor(") 
            eid = sid + 5
            gns.append(float(l[sid:eid]))
        except:
            print("err")
            continue
    return gns 
    
def get_recall_and_score_list(pth):
    lines = load_file(pth)
    gns = []
    for l in lines:
        try: 
            sid = l.index("tensor(") + len("tensor(") 
            eid = sid + 5
            gns.append(float(l[sid:eid]))
        except:
            print("err")
            continue
    recalls = []
    for l in lines:
        try: 
            sid = l.index("recall : ") + len("recall : ") 
            eid = sid + 5
            gns.append(float(l[sid:eid].strip()))
        except:
            print("err")
            continue
    return recalls, gns 
    


ood_well_pth = pth + "id"
ood_well = get_score_list(ood_well_pth)
print(sum(ood_well) / len(ood_well))
 

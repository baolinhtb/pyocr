lst = []
in_filepath="data/chn_regions.txt"
out_filepath="data/chn_regions.lst"
split_chr="    "
from idcard import digits,symbols
remove_chrs=digits+symbols
with open(in_filepath,"r") as f:
    for line in f:
        for name in line.strip("\n").split(split_chr):
            orgn = name
            for chr in remove_chrs:
                orgn=orgn.rstrip(chr)
            lst.append(orgn)

for name in lst:
    print name

with open(out_filepath,"w") as f:
    for name in lst:
        f.write("%s\n"%name)
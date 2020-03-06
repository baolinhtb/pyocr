# written by junying, 2020-03-06
# input: x y b a
# output:
#         x
#         y
#         b
#         a
lst = []
in_filepath="data/chn_regions.txt"
out_filepath="data/chn_regions.lst"
split_chr="    "
import string
remove_chrs=[str(digit) for digit in string.digits]
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
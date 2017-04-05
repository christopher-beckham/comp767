import sys

infile, outfile = sys.argv[1], sys.argv[2]

g = open(outfile, "wb")
g.write("episode,loss,reward,eps\n")
with open(infile) as f:
    for line in f:
        splits = line.rstrip().split()
        g.write(",".join([splits[1], splits[8].replace(",",""), splits[11].replace(",",""), splits[14].replace(",","")]) + "\n")

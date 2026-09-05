"""Smoke-batch structural sanity checks for HighFold2 outputs.

Runs on the server. For every manifest row: locates rank_001 PDB, measures
SG-SG distances of annotated DSB pairs (<2.5 A expected) and, for head-to-tail
rows, the terminal N-to-C closure distance. Prints one line per peptide.
"""
import glob
import json
import math

RUN = "/root/autodl-tmp/highfold_cycamp"


def parse(pdb):
    atoms = []
    for line in open(pdb):
        if line.startswith(("ATOM", "HETATM")):
            name = line[12:16].strip()
            res = line[17:20].strip()
            ri = int(line[22:26])
            xyz = (float(line[30:38]), float(line[38:46]), float(line[46:54]))
            atoms.append((name, res, ri, xyz))
    return atoms


def dist(a, b):
    return math.dist(a, b)


lines = open(f"{RUN}/manifest.csv").read().splitlines()
header, rows = lines[0].split(","), [l for l in lines[1:] if l.strip()]
idx = {name: i for i, name in enumerate(header)}

for r in rows:
    f = r.split(",")
    pid, cyc, dsb = f[idx["peptide_id"]], f[idx["flag_cyclic"]], f[idx["disulfide_pairs_flat"]]
    pdbs = sorted(glob.glob(f"{RUN}/results/{pid}/*rank_001*.pdb"))
    if not pdbs:
        print(pid, "NO_PDB")
        continue
    atoms = parse(pdbs[0])
    out = [pid]
    pairs = [int(x) for x in dsb.split()] if dsb else []
    ok_ss = True
    for k in range(0, len(pairs), 2):
        i, j = pairs[k], pairs[k + 1]
        sg1 = [a for a in atoms if a[0] == "SG" and a[2] == i]
        sg2 = [a for a in atoms if a[0] == "SG" and a[2] == j]
        if sg1 and sg2:
            dd = dist(sg1[0][3], sg2[0][3])
            out.append(f"SS{i}-{j}={dd:.2f}")
            if dd > 2.5:
                ok_ss = False
        else:
            out.append(f"SS{i}-{j}=MISSING")
            ok_ss = False
    if cyc == "1":
        ris = sorted(set(a[2] for a in atoms))
        first_n = [a for a in atoms if a[0] == "N" and a[2] == ris[0]]
        last_c = [a for a in atoms if a[0] == "C" and a[2] == ris[-1]]
        if first_n and last_c:
            out.append(f"NC_close={dist(first_n[0][3], last_c[0][3]):.2f}")
        else:
            out.append("NC=MISSING")
    sj = glob.glob(f"{RUN}/results/{pid}/*scores_rank_001*.json")
    if sj:
        s = json.load(open(sj[0]))
        pl = s.get("plddt")
        mean = sum(pl) / len(pl) if isinstance(pl, list) else pl
        out.append(f"pLDDT={mean:.1f}")
    print(" ".join(out), "" if ok_ss else "<< SS_TOO_LONG")

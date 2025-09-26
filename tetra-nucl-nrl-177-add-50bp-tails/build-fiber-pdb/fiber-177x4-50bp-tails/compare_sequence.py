file = "./histone_3.pdb"
res = ""
res_arr = []
res_old = 0

res_map = {"GLY":"G", "ALA":"A", "LEU":"L", "MET":"M", "PHE":"F", "TRP":"W", "LYS":"K", "GLN":"Q", "GLU":"E", "SER":"S",
           "PRO":"P", "VAL":"V", "ILE":"I", "CYS":"C", "TYR":"Y", "HIS":"H", "ARG":"R", "ASN":"N", "ASP":"D", "THR":"T"}

res_ref = ["SGRGKQGGKARAKAKSRSSRAGLQFPVGRVHRLLRKGNYAERVGAGAPVYLAAVLEYLTAEILELAGNAARDNKKTRIIPRHLQLAIRNDEELNKLLGRVTIAQGGVLPNIQAVLLPKCTESHHKAKGK",
           "PEPAKSAPAPKKGSKKAVTKAQKKDGKKRKRSRKESYSVYVYKVLKQVHPDTGISSKAMGIMNSFVNDIFERIAGEASRLAHYNKRSTITSREIQTAVRLLLPGELAKHAVSEGTKAVTKYTSSK",
           "ARTKQTARKSTGGKAPRKQLATKAARKSAPATGGVKKPHRYRPGTVALREIRRYQKSTELLIRKLPFQRLVREIAQDFKTDLRFQSSAVMALQEACEAYLVGLFEDTNLAAIHAKRVTIMPKDIQLARRIRGERA",
           "SGRGKGGKGLGKGGAKRHRKVLRDNIQGITKPAIRRLARRGGVKRISGLIYEETRGVLKVFLENVIRDAVTYTEHAKRKTVTAMDVVYALKRQGRTLYGFGG"]
res_ref = [res_ref[2], res_ref[3], res_ref[0], res_ref[1]]
tot = 0
for _ in range(2):
    for res in res_ref:
        tot += 1
        print(f"\n{tot}")
        for c in res:
            print(c,end="")
            tot += 1
        tot -= 1
        print(f"\n{tot}")
print()
print(tot)
exit()

with open(file, "r") as f:
    for line in f.readlines():
        tokens = line.split()
        if(tokens[0] == "ATOM"):
            if(int(tokens[5]) > res_old):
                res_old = int(tokens[5])
                res += res_map[tokens[3]]
            elif(int(tokens[5]) < res_old):
                res_arr.append(res)
                res_old = int(tokens[5])
                res = res_map[tokens[3]]

res_arr.append(res)
for i in range(len(res_arr)):
    print(i, res_arr[i])

for i in range(len(res_arr)):
    for j in reversed(range(i, len(res_arr))):
        if(res_arr[j] == res_arr[i]):
            res_arr.pop(j)
            break

res_arr[3] = "   "+res_arr[3]
for i in range(len(res_arr)):
    print(res_arr[i], len(res_arr[i]))
    for j in range(len(res_ref[i])):
        if(j >= len(res_arr[i])):
            print(res_ref[i][j], end="")
        elif(res_arr[i][j] != res_ref[i][j]):
            print(res_ref[i][j], end="")
        else:
            print(".", end="")

    print()
    print()


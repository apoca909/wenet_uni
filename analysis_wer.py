wer_file = 'D://abc//wer1.txt'

lines = []

for line in open(wer_file, 'r', encoding='utf-8') :
    if line.find('==') == 0:
        break
    if line.strip():
        lines.append(line.strip())

lines = [lines[i:i+4] for i in range(0, len(lines), 4)]

print(len(lines))
errlines = []
for one in lines:
    lab = one[2][4:]
    rec = one[3][4:]
    if lab == rec:
        continue
    errlines.append(one)
    print(one)

print(len(errlines)/len(lines), len(errlines), len(lines))

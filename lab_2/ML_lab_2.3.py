import matplotlib.pyplot as plt


with open('text.txt', 'r') as file:
    txt = file.read()

x, y, z = 0, 0, 0

let = ['!', '?', '...']

for i in txt:
    if i == let[0]:
        x += 1
    if i == let[1]:
        z += 1

for i in range(0, len(txt)):
    if txt[i: i+3] == let[2]:
        y += 1

val = [x, y, z]
let = ['!', '...', '?']

plt.hist(let, weights=val, edgecolor='k')
plt.show()

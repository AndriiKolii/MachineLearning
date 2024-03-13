import matplotlib.pyplot as plt

with open('text.txt', 'r') as file:
    txt = file.read()

x, y, z = 0, 0, 0

let = ['a', 'b', 'c']

for i in txt.lower():
    if i == let[0]:
        x += 1
    if i == let[1]:
        y += 1
    if i == let[2]:
        z += 1
val = [x, y, z]

plt.hist(let, weights=val, edgecolor='k')
plt.show()

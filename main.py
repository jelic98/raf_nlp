import matplotlib.pyplot as plt

total = 10

file = open("out/weights-ih.txt", "r")

x = []
y = []

for line in file:
    if total < 0:
        break
    data = line.split(" ")
    x.append(data[10])
    y.append(data[11])
    total -= 1

file.close()

plt.scatter(x, y)
plt.show()

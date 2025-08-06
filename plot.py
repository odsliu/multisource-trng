import matplotlib.pyplot as plt

data = eval(input())
num = len(data)

print(sum(data) / num)

# Line Plot
plt.figure(figsize=(10, 6))
plt.plot(data, color='blue', linewidth=1)
plt.title(f'Line Plot of {num} Random Numbers')
plt.xlabel('Index')
plt.ylabel('Value')
plt.grid(True, linestyle='--', alpha=0.7)
plt.savefig('line_plot.png')  # Optional: Save the figure
plt.show()

# Scatter Plot
plt.figure(figsize=(10, 6))
plt.scatter(list(range(num)), data, s=10, color='red', alpha=0.7)
plt.title(f'Scatter Plot of {num} Random Numbers')
plt.xlabel('Index')
plt.ylabel('Value')
plt.grid(True, linestyle='--', alpha=0.5)
plt.savefig('scatter_plot.png')
plt.show()

# Histogram
plt.figure(figsize=(10, 8))
plt.hist(data, bins=20, color='green', edgecolor='black', alpha=0.7)
plt.title(f'Histogram of {num} Random Numbers')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.grid(True, axis='y', linestyle='--', alpha=0.7)
plt.savefig('histogram.png')
plt.show()
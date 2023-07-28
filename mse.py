import numpy as np
import matplotlib.pyplot as plt


def read_data(file_dir):
    values = []
    with open(file_dir, 'r') as file:
        for idx, line in enumerate(file):
            if idx >= 200:
                break
            x = float(line.strip())
            values.append(x)


    return np.array(values)


def mean_squared_error(original_vals, autogen_vals):
    if len(original_vals) != len(autogen_vals):
        return -1
    
    n = len(original_vals)
    squared_error = (original_vals - autogen_vals) ** 2
    mean_squared_error = np.sum(squared_error) / n
    print(mean_squared_error)
    return squared_error

def plot_graph(x, y):

    plt.figure(figsize=(8, 6))
    plt.scatter(x, y, c='b', marker='o')
    plt.xlabel('Points')
    plt.ylabel('Error')
    plt.title('Squared error')
    plt.grid(True)

    # # Calculate the correlation coefficient
    # correlation_coefficient = np.corrcoef(x, y)[0, 1]
    # plt.text(0.3, 0.9, f'Correlation coefficient: {correlation_coefficient:.2f}',
    #          transform=plt.gca().transAxes, fontsize=12, ha='center')

    plt.show()


if __name__ == "__main__":
    original_values = read_data('/home/ckengjwe/dso/r_values_rpg1.txt')
    autogen_values = read_data('/home/ckengjwe/dso/r_values_rpg2.txt')

    se = mean_squared_error(original_values, autogen_values)
    x = np.linspace(1, 200, 200, dtype = int)
    plot_graph(x, se)     
    



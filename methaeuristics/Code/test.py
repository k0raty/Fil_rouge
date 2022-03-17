from seaborn import color_palette
import matplotlib.pyplot as plt

nbr_of_colors = 5
colors = color_palette(n_colors=nbr_of_colors)
print(colors)

X = [0, 1, 2]
Y = [2, 3, 0]


def test_palette():
    plt.close('all')

    for i in range(nbr_of_colors):
        color = colors[i]

        print(color)

        plt.plot(X, Y,
                 marker='o',
                 markerfacecolor='blue',
                 markeredgecolor='blue',
                 linestyle='solid',
                 linewidth=0.5,
                 color=color,
                 )
        plt.show()

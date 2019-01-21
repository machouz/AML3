import matplotlib.pyplot as plt


def create_graph(name, array_datas=[], array_legends=["Validation"],
                 xlabel="Epoch", ylabel="Loss",
                 make_new=True):
    if make_new:
        plt.figure()
    lines = []
    for data in array_datas:
        line, = plt.plot(data)
        lines.append(line)
        plt.title(name)
    plt.legend(lines, array_legends, loc=4)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.show()
    plt.savefig(name)

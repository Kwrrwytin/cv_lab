import os
import argparse
import matplotlib.pyplot as plt
import numpy as np

def read_data(dir_path):
    with open(dir_path, "r") as f:
        raw_data = f.read()
        data = raw_data[1:-1].split(", ")
    return np.asfarray(data, float)

def draw_curve(var, type, folder_path, f_names):
    plt.figure()

    ax = plt.axes()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.xlabel('epochs')    # x轴标签
    plt.ylabel('loss')     # y轴标签

    for filename in f_names:
        pos1 = filename.find('_')
        pos2 = filename[pos1+1:].find('_')
        pos3 = filename.rfind('_')

        file_path = os.path.join(folder_path, filename)
        y_loss = read_data(file_path)
        x_val = range(1, len(y_loss)+1)
        if var == 'activation':
            l = filename[pos1+1 : pos1+1+pos2]
        elif var == 'neurons':
            l = filename[: pos1]
        elif var == 'layers':
            l = filename[pos1+1+pos2+1:pos3]

        plt.plot(x_val, y_loss, linewidth=2, linestyle="solid", label=l)
        plt.legend(loc='center', bbox_to_anchor=(0.7, 0.85))
    
    plt.title('{} Loss Curve - {}'.format(type, var))
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='loss_curve')
    parser.add_argument("--variable", type=str, default='activation',
                        help="controlled variable. such as activation, neurons, layers...")
    parser.add_argument("--type", type=str, default='test',
                        help="draw which curve? such as train, test")

    args = parser.parse_args()
    folder_path = "./loss_data/{}/{}".format(args.variable, args.type)
    file_names = os.listdir(folder_path)

    draw_curve(args.variable, args.type, folder_path, file_names)
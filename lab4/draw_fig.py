import os
import argparse
import matplotlib.pyplot as plt
import numpy as np

def read_data(dir_path):
    with open(dir_path, "r") as f:
        raw_data = f.read()
        data = raw_data[1:-1].split(", ")
    return np.asfarray(data, float)

def draw_acc_curve(var, type, folder_path, f_names):
    plt.figure()

    ax = plt.axes()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.xlabel('R')    # x轴标签
    plt.ylabel('ASR')     # y轴标签

    plt.ylim(0,100)

    for filename in f_names:
        

        file_path = os.path.join(folder_path, filename)
        y_loss = read_data(file_path)
        x_val = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

        plt.plot(x_val, y_loss, linewidth=2, linestyle="solid", label=filename)
        plt.legend(loc="lower right")
    
    plt.title('ASR - R Curve')
    plt.show()

def draw_class_bar(var, type, folder_path, f_names):
    plt.figure()

    ax = plt.axes()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.xlabel('class label')    # x轴标签
    plt.ylabel('class ACC')     # y轴标签

    plt.ylim(0, 1)

    x_val = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    x = np.arange(10)


    for i, filename in enumerate(f_names):
        # pos_lr = filename.find('_')
        pos_nw = filename.rfind('_')
        pos_txt = filename.rfind('.')
        # if filename[pos_lr+6] == '0':
        #     continue
        if var == 'network':
            l = filename[pos_nw+1:pos_txt]

        file_path = os.path.join(folder_path, filename)
        y_acc = read_data(file_path)

        plt.bar(x+i*0.4, y_acc, width=0.8, label=l)
        plt.legend(loc="lower right")
        
    plt.xticks(x, x_val)
    plt.title('{} - k Curve'.format(type))
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='acc_curve')
    parser.add_argument("--variable", type=str, default='network',
                        help="controlled variable. such as activation, neurons, layers...")
    parser.add_argument("--type", type=str, default='time',
                        help="draw which curve? such as acc, class_acc")

    args = parser.parse_args()
    folder_path = "./results/"
    file_names = os.listdir(folder_path)
    draw_acc_curve(args.variable, args.type, folder_path, file_names)

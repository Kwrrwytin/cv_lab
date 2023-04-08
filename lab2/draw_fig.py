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

    plt.xlabel('epochs')    # x轴标签
    plt.ylabel('acc')     # y轴标签

    for filename in f_names:
        pos1 = filename.find('_')
        pos = filename[pos1+1:].find('_')
        pos2 = filename.rfind('_')
        lr_name = filename[pos1+1:pos1+1+pos2]
        if var == 'network':
            l = lr_name+filename[pos2+1:]

        file_path = os.path.join(folder_path, filename)
        y_loss = read_data(file_path)
        x_val = range(1, len(y_loss)+1)

        plt.plot(x_val, y_loss, linewidth=2, linestyle="solid", label=l)
        plt.legend(loc="lower right")
    
    plt.title('{} ACC Curve'.format(type))
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
    plt.title('{} ACC Curve'.format(type))
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='acc_curve')
    parser.add_argument("--variable", type=str, default='network',
                        help="controlled variable. such as activation, neurons, layers...")
    parser.add_argument("--type", type=str, default='acc',
                        help="draw which curve? such as acc, class_acc")

    args = parser.parse_args()
    folder_path = "./results/{}".format(args.type)
    file_names = os.listdir(folder_path)
    if args.type == 'acc':
        draw_acc_curve(args.variable, args.type, folder_path, file_names)
    else:
        draw_class_bar(args.variable, args.type, folder_path, file_names)

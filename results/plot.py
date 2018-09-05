#!/usr/bin/env python
import matplotlib as mpl
mpl.use('Agg')  # NOQA
import matplotlib.pyplot as plt

import json


def plot_loss(ax, logs, train):
    if train:
        title = 'Train loss'
        key = 'main/loss'
    else:
        title = 'Validation loss'
        key = 'validation/main/loss'

    ax.set_title(title)
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss')
    ax.set_ylim(0, 20)

    for label, log in logs:
        xs = [*map(lambda e: e['epoch'], log)]
        ys = [*map(lambda e: e[key], log)]
        ax.plot(xs, ys, label=label)

    # ax.legend(loc='upper right')


def plot_accuracy(ax, logs, train):
    if train:
        title = 'Train accuracy'
        key = 'main/accuracy'
    else:
        title = 'Validation accuracy'
        key = 'validation/main/accuracy'

    ax.set_title(title)
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Accuracy')
    ax.set_ylim(0, 1)

    for label, log in logs:
        xs = [*map(lambda e: e['epoch'], log)]
        ys = [*map(lambda e: e[key], log)]
        ax.plot(xs, ys, label=label)

    # ax.legend(loc='upper left')

    if train:
        ax.legend(bbox_to_anchor=(1, 1.15), borderaxespad=0, loc='lower right')


def plot_elapsed_time(ax, logs):
    ax.set_title('Elapsed time')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Seconds')

    for label, log in logs:
        xs = [*map(lambda e: e['epoch'], log)]
        ys = [*map(lambda e: e['elapsed_time'], log)]
        ax.plot(xs, ys, label=label)

    ax.legend(loc='upper left')


def main():
    log_labels = [
        ('bn', 'Batch Norm'),
        ('gnchainer', 'Group Norm (Chainer)'),
        ('gnalb1', 'Group Norm (ALB1)'),
        ('gnalb2', 'Group Norm (ALB2)'),
    ]

    def load_log(shortname, label):
        filename = 'log_{}.json'.format(shortname)
        with open(filename) as f:
            log = json.load(f)
        return label, log

    logs = [load_log(shortname, label) for shortname, label in log_labels]

    mpl.rcParams['font.family'] = 'Avenir Next LT Pro'
    plt.style.use('tableau-colorblind10')

    fig, axes = plt.subplots(2, 2, figsize=(640/72, 608/72))
    plot_loss(axes[0, 0], logs, True)
    plot_accuracy(axes[0, 1], logs, True)
    plot_loss(axes[1, 0], logs, False)
    plot_accuracy(axes[1, 1], logs, False)
    fig.tight_layout()

    filename = 'precision.svg'
    fig.savefig(filename, transparent=True)
    plt.close(fig)
    with open(filename, 'r+') as f:
        svg = f.read()
        svg = svg.replace('pt', 'px')
        f.seek(0)
        f.write(svg)

    fig, ax = plt.subplots(figsize=(480/72, 384/72))
    plot_elapsed_time(ax, logs)
    fig.tight_layout()

    filename = 'elapsed_time.svg'
    fig.savefig(filename, transparent=True)
    plt.close(fig)
    with open(filename, 'r+') as f:
        svg = f.read()
        svg = svg.replace('pt', 'px')
        f.seek(0)
        f.write(svg)


if __name__ == '__main__':
    main()

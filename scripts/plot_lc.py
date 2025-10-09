import argparse, math

from matplotlib import pyplot as plt

def main(args):
    train_x, train_y = [], []
    val_x, val_y = [], []
    lr_x, lr_y = [], []
    with open(args.loss_file, "r") as f:
        for line in f:
            line = line.rstrip()
            line_type, x, y = line.split(" ")
            if line_type == "TRAIN":
                train_x.append(float(x))
                train_y.append(float(y))
            elif line_type == "VALID":
                val_x.append(float(x))
                val_y.append(float(y))
            elif line_type == "LR":
                lr_x.append(float(x))
                lr_y.append(float(y))
            else:
                raise ValueError(f"Line type {line_type} not recognised")

    if any(math.isnan(y_1) or math.isnan(y_2) for y_1, y_2 in zip(train_y, val_y)):
        print("Found nans, converting nan -> -0.1")
        train_y = [ y if not math.isnan(y) else -0.1 for y in train_y ]
        val_y = [ y if not math.isnan(y) else -0.1 for y in val_y ]

    _, ax = plt.subplots(1, 1, figsize=(8,6))
    ax.plot(train_x, train_y, c="k", alpha=0.6, label="Train")
    ax.plot(val_x, val_y, c="r", label="Val")
    ax.set_ylim(top=-2, bottom=min(train_y) * 1.1)

    ax2 = ax.twinx()
    ax2.plot(lr_x, lr_y, c="b", alpha=0.5, label="LR")
    ax2.set_ylabel("Learning rate", color="b", fontsize=14)
    ax2.tick_params(axis='y', labelcolor="b")

    # main axis labels
    ax.set_xlabel("Iter", loc="right", fontsize=16)
    ax.set_ylabel("Loss", loc="top", fontsize=16)
    ax.grid()

    # combine legends from both axes
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, fontsize=14, loc="best")

    plt.tight_layout()
    plt.show()

def parse_cli():
    parser = argparse.ArgumentParser()

    parser.add_argument("loss_file", type=str)

    args = parser.parse_args()

    return args

if __name__ == "__main__":
    main(parse_cli())

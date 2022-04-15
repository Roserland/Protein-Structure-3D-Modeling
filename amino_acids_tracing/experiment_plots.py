import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import os

train_words_total = 114994
valid_words_total = 68746


def tracing_loss_plot(output_dir='./output_dir/imgs'):
    train_log_file = './output_dir/train.log'
    valid_log_file = './output_dir/valid.log'

    t_df = pd.read_csv(train_log_file)[:150]
    # t_df["pos_loss"] = t_df["loss"] * 2 / train_words_total - t_df["ppl"]
    v_df = pd.read_csv(valid_log_file)[:150]
    print(t_df.head())

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.plot(t_df["epoch"], t_df['seq_loss'], label="train_seq_loss", linewidth=1)
    ax1.plot(v_df["epoch"], v_df['seq_loss'], label="valid_seq_loss", linewidth=1)
    ax1.plot(t_df["epoch"], t_df['pos_loss'], label="train_pos_loss", linewidth=1)
    ax1.plot(v_df["epoch"], v_df['pos_loss'], label="valid_pos_loss", linewidth=1)
    ax1.set_ylim([0, 4.5])

    ax1.set_ylabel("Sequence loss")
    ax1.set_title("TraceFormer Training Process")
    ax1.set_xlabel("epochs")
    ax1.legend()
    plt.savefig(os.path.join(output_dir, 'loss.png'))
    plt.close()

    

    plt.plot(t_df["epoch"], t_df['accuracy'] / 100, label="train_seq_acc", linewidth=1)
    plt.plot(v_df["epoch"], v_df['accuracy'] / 100, label="valid_seq_acc", linewidth=1)
    plt.title("TraceFormer Training Process")
    plt.xlabel("epochs")
    plt.legend(loc="center right")
    plt.savefig(os.path.join(output_dir, 'acc.png'))
    plt.close()



if __name__ == "__main__":
    tracing_loss_plot()

    a = torch.tensor([[0.1, 0.2, 0.9], [0.4, 0.1, 0.2]])
    b = torch.tensor([1, 2], dtype=torch.long)

    ce_loss_fn = torch.nn.CrossEntropyLoss(ignore_index=0, reduction="sum")
    ce_loss = ce_loss_fn(a, b)
    print("ce loss: ", ce_loss)
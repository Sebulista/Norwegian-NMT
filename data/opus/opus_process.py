""" Script for pretty printing csv data from the OPUS API about available data """
import pandas
import glob
import os
from tabulate import tabulate
import matplotlib.pyplot as plt
from pandas.plotting import table
import re

results = glob.glob("*.csv")

def process_csv(filename):
    df = pandas.read_csv(filename, dtype={"corpus":str, "xml":"Int32", "moses":"Int32", "tmx":"Int32"})
    if re.search("unique", filename):
        df.sort_values(by="moses", ascending = False, inplace=True)
    else:
        df.sort_values(by="corpus", inplace=True)
    df.loc["Total"] = pandas.Series(df[["xml", "moses", "tmx"]].sum(), index=["xml", "moses", "tmx"])
    df = df.fillna(0)
    
    processed_dir = "processed/"

    if not os.path.exists(processed_dir):
        os.mkdir(processed_dir)

    df.to_csv(os.path.join(processed_dir, filename), sep=",", index=False, encoding="utf-8")

    print(f"\n{filename}")
    print(df)

    make_png(df, filename)

def make_png(df, filename):
    fig, ax = plt.subplots(figsize=(len(df.columns)+2, len(df)/7)) # set size frame
    ax.xaxis.set_visible(False)  # hide the x axis
    ax.yaxis.set_visible(False)  # hide the y axis 
    ax.set_frame_on(False)  # no visible frame
    tab = table(ax, df, loc='center', cellLoc = 'center')

    tab.auto_set_font_size(False) # Activate set fontsize manually
    tab.set_fontsize(10) # if ++fontsize is necessary ++colWidths
    tab.scale(1.2, 1.2) # Table size

    plot_dir = "pngs/"

    if not os.path.exists(plot_dir):
        os.mkdir(plot_dir)

    plt.savefig(os.path.join(plot_dir, f'{filename[:-4]}.png'))

def pretty(filename):
    df = pandas.read_csv(filename, dtype={"corpus":str, "xml":"Int32", "moses":"Int32", "tmx":"Int32"})
    if re.search("unique", filename):
        df.sort_values(by="moses", ascending = False, inplace=True)
    else:
        df.sort_values(by="corpus", inplace=True)
    df.loc["Total"] = pandas.Series(df[["xml", "moses", "tmx"]].sum(), index=["xml", "moses", "tmx"])
    df = df.fillna(0)

    print(f"\n{filename}")
    print(tabulate(df, headers = "keys"))


for file in results:
    process_csv(file)
    pretty(file)

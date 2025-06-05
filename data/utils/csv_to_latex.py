""" A simple conversion of a csv files with file line counts into a latex formatted table"""
from pathlib import Path

def read_table(path):
    with open(path, "r", encoding = "utf-8") as f:
        data = f.read().strip().split("\n")

    latex = ""

    pretty = r"\textbf{Corpus} & \textbf{Raw count} & \textbf{Deduped count} & $\Delta$"
    latex += f"{pretty} \\\\ \n \\hline\n"
    
    for row in data[1:]:
        pretty = f"\\textbf{{{row.split(',')[0]}}}"
        entries = []
        for entry in row.split(',')[1:]:
            pretty += f" & {int(entry):,}"
            entries.append(int(entry))
        delta = (((entries[1] / entries[0])*100)-100)
        pretty += f" & {delta:.2f}\%"
        pretty += "\\\\\n"
        latex += pretty
            

    latex = latex.replace("_", r"\_")
    return latex

def main():
    data = read_table("path/to/table")
    print(data)
    

if __name__ == "__main__":
    main()

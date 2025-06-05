""" CLI script to get the number of lines in each file within a directory
and compare it against up to two other directories (cmp_dir). Output the data as a
csv file"""

from pathlib import Path
import gzip
import pandas as pd
import argparse

def get_line_count(filename):
    """ Gets the line count of a text file 

    Arguments:
    filename (str | Path): path to text file
    """
    with open(filename, 'r', encoding='utf-8') as file:
        return sum(1 for _ in file)
    
def get_line_count_zipped(filename):
    """ Gets the line count of a zipped text file 

    Arguments:
    filename (str | Path): path to zipped text file (file.txt.gzip)
    """
    with gzip.open(filename, "rt", encoding = "utf-8") as file:
        return sum(1 for _ in file)
    
def parse_arguments():
    parser = argparse.ArgumentParser()
    #parser.add_argument("--direction", type=str, required=True, help="Language translation direction (f.ex. nb-en)")

    parser.add_argument("--org_dir", "-o", type = str, required = True)
    parser.add_argument("--cmp_dir", "-c", type = str, required = True)
    parser.add_argument("--cmp_dir2", "-c2", type = str, default = None)
    parser.add_argument("--stats_dir", "-s", type = str, required = True)
    parser.add_argument("--direction", "-d", type = str, required = True, choices = ["nb_en", "nn_nb", "nn_en"])
    parser.add_argument("--title", type = str, default = "deduped_count")
    parser.add_argument("--title2", type = str, default = None)
    #parser.add_argument("--reversed", action = store_true)

    return parser.parse_args()

"""def get_counts(raw_file, deduped_file):
    raw_count = get_line_count(raw_file)
    deduped_count = get_line_count(deduped_file)

    return raw_count, deduped_count"""

def pretty(filename):
    """ Reads a csv file and sorts by the line counts of the original file

    Arguments:
    filename (str | Path): path to csv file
    """
    df = pd.read_csv(filename, dtype={"corpus":str, "raw_count":"Int32", "deduped_count":"Int32"})
    df.sort_values(by="raw_count", inplace=True)
    df.loc["Total"] = pd.Series(df[["raw_count", "deduped_count"]].sum(), index=["raw_count", "deduped_count"])
    df = df.fillna(0)

    df.to_csv(Path(f"path/to/output_folder/{filename}"), sep=",", index=False, encoding="utf-8")


def get_stats(raw_dir, deduped_dir, stats_dir, title):
    #files should be the same and have the same name

    rows = []

    for child in raw_dir.iterdir():
        if child.is_file():
            name = child.name
    
            print(f"Processing: {name}")
    
            raw_count = get_line_count(child)
            deduped = deduped_dir / name
            deduped_count = get_line_count(deduped)
    
            rows.append([name, raw_count, deduped_count])

    df = pd.DataFrame(rows, columns = ["corpus", "raw_count", title])#, dtype={"corpus":str, "raw_count":"Int32", "deduped_count":"Int32"})
    df.sort_values(by="raw_count", ascending = False, inplace=True)
    df.loc["Total"] = pd.Series(df[["raw_count", title]].sum(), index=["raw_count", title])
    df = df.fillna(0)

    #store as ints
    for column in df.select_dtypes(include='float').columns:
        if not df[column].isnull().any():  # Check if column does not contain NaN values
            df[column] = df[column].astype(int)
    df.to_csv(stats_dir / f"{deduped_dir.name}.csv", sep=",", index=False, encoding="utf-8")    

def get_stats2(raw_dir, cmp_dir_1, cmp_dir_2, stats_dir, args):
    rows = []

    for child in raw_dir.iterdir():
        if child.is_file():
            name = child.name
    
            print(f"Processing: {name}")
    
            raw_count = get_line_count(child)
            cmp_1 = cmp_dir_1 / name
            processed_count_1 = get_line_count(cmp_1)

            cmp_2 = cmp_dir_2 / name
            processed_count_2 = get_line_count(cmp_2)

            sum_cmp = processed_count_1 + processed_count_2

            delta = sum_cmp - raw_count
    
            rows.append([name, raw_count, processed_count_1, processed_count_2, sum_cmp, delta])

    df = pd.DataFrame(rows, columns = ["corpus", "raw_count", args.title, args.title2, "sum_processed", "delta"])#, dtype={"corpus":str, "raw_count":"Int32", "deduped_count":"Int32"})
    df.sort_values(by="raw_count", ascending = False, inplace=True)
    df.loc["Total"] = pd.Series(df[["raw_count", args.title, args.title2, "sum_processed", "delta"]].sum(), index=["raw_count", args.title, args.title2, "sum_processed", "delta"])
    df = df.fillna(0)

    #store as ints
    for column in df.select_dtypes(include='float').columns:
        if not df[column].isnull().any():  # Check if column does not contain NaN values
            df[column] = df[column].astype(int)
    df.to_csv(stats_dir / f"{args.title[:3]}_{raw_dir.name}.csv", sep=",", index=False, encoding="utf-8")    

if __name__ == "__main__":
    args = parse_arguments()
    #direction = args.direction
    org_dir = Path(args.org_dir) / args.direction
    cmp_dir = Path(args.cmp_dir) / args.direction
    stats_dir = Path(args.stats_dir)

    assert org_dir.is_dir(), f"{org_dir} is not a dir"
    assert cmp_dir.is_dir(), f"{cmp_dir} is not a dir"
    assert stats_dir.is_dir(), f"{stats_dir} is not a dir"

    if args.cmp_dir2:
        cmp_dir2 = Path (args.cmp_dir2) / args.direction
        assert cmp_dir2.is_dir(), f"{cmp_dir2} is not a dir"

        get_stats2(org_dir, cmp_dir, cmp_dir2, stats_dir, args)
    else:
        get_stats(org_dir, cmp_dir, stats_dir, args.title)

        

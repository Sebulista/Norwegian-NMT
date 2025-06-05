""" CLI program to apply bicleaner-hardrules to a folder of files containing bitext """
import subprocess
import shlex
from pathlib import Path
from argparse import ArgumentParser
import time
import os


def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument("--source_dir", "-s", type = str, help = "Folder in which to bifix the files")
    parser.add_argument("--outdir", "-o", type = str, help = "Output folder")

    parser.add_argument("--direction", "-d", type = str, help = "Language direction for which to filter", choices = ["nb_en", "nn_nb", "nn_en"])
    parser.add_argument("--file", "-f", type = str, help = "Single file to process")

    args = parser.parse_args()

    return args

def biclean_file(file, slang = "en", tlang = "nb"):
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
    os.environ['XLA_FLAGS'] = '--xla_gpu_cuda_data_dir=/cluster/software/EL9/easybuild/software/CUDA/12.2.0'
    subprocess.call(shlex.split(f"bicleaner-ai-classify {file} - path/to/bicleaner-models/{slang}-{tlang}/ --scol 1 --tcol 2 --disable_hardrules --disable_lm_filter --disable_porn_removal --disable_minimal_length --batch_size 64"), env=os.environ)

def biclean(path, output_path, args):
    (output_path / args.direction).mkdir(exist_ok = True)
    for file in sorted(path.iterdir(), key = os.path.getsize):
        input = path / file.name
        output = output_path / args.direction / file.name
        slang = args.direction.split("_")[1]
        tlang = args.direction.split("_")[0]

        if not output.exists():
            t = time.localtime()
            time_string = time.strftime('%H:%M:%S', t)
            
            print(f"{time_string}: PROCESSING: {input.name}", flush = True)

            os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
            os.environ['XLA_FLAGS'] = '--xla_gpu_cuda_data_dir=/cluster/software/EL9/easybuild/software/CUDA/12.2.0'
            subprocess.call(shlex.split(f"bicleaner-ai-classify {input} {output} path/to/bicleaner-models/{slang}-{tlang}/ --scol 1 --tcol 2 --disable_hardrules --disable_lm_filter --disable_porn_removal --disable_minimal_length --batch_size 64"), env=os.environ)
        else:
            print(f"Skipping {output} it has been processed already.")

def main():
    args = parse_arguments()
    if args.direction is None and args.file is not None:
        biclean_file(args.file)
    else:
        biclean(Path(args.src_dir) / args.direction, Path(args.outdir), args)

if __name__== "__main__":
    main()
    

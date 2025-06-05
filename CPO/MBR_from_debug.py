""" Create a preference dataset by MBR ranking based on a debug file containing
all hypotheses """

from pathlib import Path
from argparse import ArgumentParser
import json
import re
from tqdm import tqdm

from mbrs.metrics import MetricBLEU, MetricChrF, MetricCOMET, MetricCOMETkiwi, MetricMetricX, MetricTER, MetricXCOMET
from mbrs.decoders import DecoderMBR, DecoderEnsembleMBR

MBR_MAP = {
    "bleu": MetricBLEU,
    "chrf": MetricChrF,
    "chrf_pp": MetricChrF,
    "ter": MetricTER, 
    "metricx": MetricMetricX,
    "comet": MetricCOMET,
    "kiwi": MetricCOMETkiwi,
    "xcomet": MetricXCOMET,
}

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("file", type = str, help = "Which file to process")
    parser.add_argument("--metric", type = str, default = "chrf_pp", choices = MBR_MAP.keys(), help = "Which MBR metric to use")
    args = parser.parse_args()

    return args

def init_mbr_metrics(metrics: list[str]) -> list["mbrs.metrics.Metric"]:
    return [init_mbr_metric(metric) for metric in metrics]

def init_mbr_metric(metric: str) -> "mbrs.metrics.Metric":
    match metric:
        case "bleu":
            # Removed multi-threading
            metric_cfg = MBR_MAP[metric].Config()
        case "chrf_pp":
            # Removed multi-threading
            metric_cfg = MBR_MAP[metric].Config(
                word_order=2,
            )
        case "chrf":
            metric_cfg = MBR_MAP[metric].Config(
                fastchrf=True,
            )
        case "ter":
            metric_cfg = MBR_MAP[metric].Config()
        case "comet":
            metric_cfg = MBR_MAP[metric].Config(
                model="Unbabel/wmt22-comet-da",
                batch_size=64,
                fp16=True,
            )
        case "kiwi":
            metric_cfg = MBR_MAP[metric].Config(
                model="Unbabel/wmt22-cometkiwi-da",
                batch_size=64,
                fp16=True,
            )
        case "metricx":
            metric_cfg = MBR_MAP[metric].Config(
                model="google/metricx-24-hybrid-large-v2p6-bfloat16",
                batch_size=64,
                bf16=True,
            )
        case "xcomet":
            raise NotImplementedError
        case _:
            raise NotImplementedError 

    metric = MBR_MAP[metric](metric_cfg)
    return metric

def parse_entry(entry: str):
    #Split and remove empty lines
    lines = [l for l in entry.split("\n") if l]
    SOURCE = lines[0].split(" ", 1)[-1].strip()
    # Is not really needed
    ref = lines[1].split(" ", 1)[-1].strip()

    hyps =  [l for l in lines if l.startswith(">>HYP")]
    HYPOTHESES = [h.split("\t")[-1].strip() for h in hyps]

    return SOURCE, ref, HYPOTHESES

def format_json(slang: str, tlang: str, metric: str, source_sent: str, reference: str, sorted_hypotheses: list[tuple[str, int]], ref_score: tuple[str, int]) -> str:
    """Fill out a dictionary and return a json compatible dict string"""
    metric = f"MBR_{metric}"

    dict = {
        slang: source_sent,
        tlang: reference,
        # score
        f"high_{tlang}_{metric}": sorted_hypotheses[0][1],
        f"low_{tlang}_{metric}": sorted_hypotheses[-1][1],
        # sent
        f"high_{tlang}": sorted_hypotheses[0][0],
        f"low_{tlang}": sorted_hypotheses[-1][0],
        "language_pair": f"{slang}-{tlang}",
        "required_directions": f"{slang}-{tlang}",
        f"ref_{tlang}_{metric}": ref_score[1] if ref_score else -1.0,
    }

    t_dict = {"translation": dict}
    
    json_string = json.dumps(t_dict, ensure_ascii = False)

    return json_string

if __name__ == "__main__":
    args = parse_args()
    file = Path(args.file)
    assert file.is_file(), f"File does not exist"
    assert "DEBUG" in file.name, f"File is not in the expected format"

    print(f"\nProcessing: {file}")
    
    # Add MBR spec
    output_file_name = re.sub("QE", f"MBR_{args.metric}", file.name)
    # Change extension
    output_file_name = re.sub(r"_DEBUG\.txt", ".jsonl", output_file_name)
    output_file = file.parent / output_file_name
    print(f"OUTPUT FILE: {output_file}")
    assert not output_file.is_file(), f"Filename {output_file.name} already exists"

    slang,tlang = re.search(r"(en|nn|nb)_(en|nn|nb)", file.name).group(0).split("_")
    
    with open(file, "r", encoding = "utf-8") as in_file:
        data = in_file.read().strip().split("\n\n\n")

    metric = init_mbr_metric(args.metric)
    decoder_cfg = DecoderMBR.Config()
    decoder = DecoderMBR(decoder_cfg, metric)

    # GET N_best from first entry, assume all have the same N
    N_BEST=len(parse_entry(data[0])[1])

    QE_last = 0
    QE_not = 0
    
    with open(output_file, "w", encoding = "utf-8") as out_file:
        for entry in tqdm(data, miniters = len(data)//100):
            SOURCE, ref, HYPOTHESES = parse_entry(entry)
            output = decoder.decode(HYPOTHESES, HYPOTHESES, source=SOURCE, nbest=N_BEST)
    
            sorted_hypotheses = list(zip(output.sentence, output.score))
            if output.idx[0] != 0:
                QE_not += 1
            if output.idx[-1] == 0:
                QE_last += 1
            
            j = format_json(slang, tlang, args.metric, SOURCE, ref, sorted_hypotheses, None)
            
            out_file.write(f"{j}\n")

    print(f"Previous preferred sentence not preferred: {QE_not/len(data):.2%} ({QE_not})")
    print(f"Previous preferred now dispreffered: {QE_last/len(data):.2%} ({QE_last})")

    

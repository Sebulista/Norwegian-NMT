import os
# Tensorflow
if 'TF_CPP_MIN_LOG_LEVEL' not in os.environ:
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import datasets
from datasets.utils.logging import disable_progress_bar
disable_progress_bar()

import torch
import transformers

try:
    from . import models
except (ImportError, SystemError):
    import models

# Disable progress bar
transformers.logging.set_verbosity_warning() 

def get_dataset(
    input_data: list[dict[str, str]], tokenizer, max_input_length, device, is_qe: bool
):
        """Gets the test dataset for prediction.
    
        If `is_qe` is true, the input data must have "hypothesis" and "source" fields.
        If it is false, there must be "hypothesis" and "reference" fields.
    
        Args:
          input_file: The path to the jsonl input file.
          tokenizer: The tokenizer to use.
          max_input_length: The maximum input sequence length.
          device: The ID of the device to put the PyTorch tensors on.
          is_qe: Indicates whether the metric is a QE metric or not.
    
        Returns:
          The dataset.
        """
    
        def _make_input(example):
            if is_qe:
                example["input"] = (
                    "source: "
                    + example["source"]
                    + " candidate: "
                    + example["hypothesis"]
                )
            else:
                example["input"] = (
                    "source: "
                    + example["source"]
                    + " candidate: "
                    + example["hypothesis"]
                    + " reference: "
                    + example["reference"]
                )
            return example
    
        def _tokenize(example):
            return tokenizer(
                example["input"],
                max_length=max_input_length,
                truncation=True,
                padding=False,
                #padding="max_length",
            )
    
        def _remove_eos(example):
            example["input_ids"] = example["input_ids"][:-1]
            example["attention_mask"] = example["attention_mask"][:-1]
            return example

    
        #ds = datasets.load_dataset("json", data_files={"test": input_file})
        ds = datasets.Dataset.from_list(input_data)
        ds = ds.map(_make_input)
        ds = ds.map(_tokenize)
        ds = ds.map(_remove_eos)
        ds.set_format(
            type="torch",
            columns=["input_ids", "attention_mask"],
            device=device,
            output_all_columns=True,
        )
        return ds

class MetricX():
    def __init__(self, batch_size: int = 64):
        self.max_input_length = 1536
        
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.bsz = batch_size // torch.cuda.device_count()
        else:
            self.device = torch.device("cpu")
            self.bsz = batch_size

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            "google/mt5-xl",
            legacy=False,
            use_fast=False,
        )
        
        self.model = models.MT5ForRegression.from_pretrained(
            #"google/metricx-24-hybrid-large-v2p6",
            "google/metricx-24-hybrid-large-v2p6-bfloat16",
            torch_dtype=torch.bfloat16,
        )
        
        self.model.to(self.device)
        self.model.eval()

        # Data collator for batching
        data_collator = transformers.DataCollatorWithPadding(tokenizer = self.tokenizer, padding = True)
        
        self.scorer = transformers.Trainer(
            model=self.model,
            args=transformers.TrainingArguments(
                output_dir="output/path",
                dataloader_pin_memory=False,
                per_device_eval_batch_size = self.bsz,
                disable_tqdm=True,
            ),
            tokenizer=self.tokenizer,
            data_collator=data_collator,
        )


    
    def score(self, score_data: list[dict[str, str]], is_qe: bool = False) -> float:
        """ Given formatted data, returns the metricX system score

        If `is_qe` is true, the input data must have "hypothesis" and "source" fields.
        If it is false, there must be "hypothesis", "source" and "reference" fields.

        Args:
          score_data: A list of formatted dicts see above.
          is_qe: Indicates whether the metric is a QE metric or not.
    
        Returns:
          The system score.

        """
        '''ds = self.get_dataset(
            score_data,
            self.device,
            is_qe,
        )'''

        ds = get_dataset(
            score_data,
            self.tokenizer,
            self.max_input_length,
            self.device,
            is_qe,
        )

        predictions, _, _ = self.scorer.predict(test_dataset=ds)

        # Why float(pred) isn't it already a float?
        preds = [float(pred) for pred in predictions]
        system_score = sum(preds) / len(preds)

        return system_score



if __name__ == "__main__":
    # List of random sentences
    sentences = [
        "The sun sets beautifully beyond the mountains.",
        "Cats are known for their curious nature.",
        "Technology evolves at a rapid pace each year.",
        "Gardening can be a relaxing and rewarding hobby.",
        "Traveling opens the mind to new experiences.",
        "Books can transport you to distant worlds and times.",
        "Exercise is essential for maintaining good health.",
        "Cooking is both an art and a science.",
        "Music has the power to move and inspire people.",
        "The ocean's waves can be both calming and powerful."
    ]
    
    
    # Creating three lists of random sentences
    src = sentences[:5]
    ref = sentences[5:10]
    hyp = sentences[0:5]

    inp = format_data(src, hyp, ref)

    metricx = MetricX()
    ds = get_dataset(inp, metricx.tokenizer, 100, metricx.device, False)

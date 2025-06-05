""" Convert a metrics table to a latex table """
import pandas as pd

import regex as re

from argparse import ArgumentParser


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("file", type = str, help = "Which file to read to create a table")
    parser.add_argument("--truncate", action = "store_true", help = "Truncate the model name to 35 characters")

    args = parser.parse_args()
    return args


class MetricTable():
    def __init__(self, file: str):
        self.file = file
        self.dataframe = self.parse_multi_index()

        self.metrics = self.dataframe.index.get_level_values("METRIC").unique().tolist()
        self.datasets = self.dataframe.columns.get_level_values("DATASET").unique().tolist()

        self.best_metrics = self._get_best_metrics()

    @staticmethod
    def metric_to_float(metric_value: str) -> float:
        """ Convert string values from the table file into float values and
        convert percentages into floats between 0 and 1 """
        if metric_value.endswith("%"):
            float_value = round( float(metric_value.strip("% ")) / 100, 4)
        else:
            float_value = (float(metric_value))

        return float_value

    @staticmethod
    def format_score(val: float) -> str:
        return f"{val:.2f}"
    
    @staticmethod
    def format_percentage(val: float) -> str:
        #return f"{val:.2%}"
        return f"{val*100:.2f}\%"
    
    @staticmethod
    def truncate_index(val: str) -> str:
        return val.map(lambda x: x[:10]) 
    
    def get_percentage_metrics(self) -> list[str]:
        """ Return a list of the metrics that should be converted to percentages """
        return [metric for metric in self.metrics if metric in ["Comet", "CometKiwi", "BERTSCORE"]]
    
    def get_raw_metrics(self) -> list[str]:
        return [metric for metric in self.metrics if metric not in self.get_percentage_metrics()]
    
            
    #@staticmethod
    def highlight_max(self, s: pd.Series) -> list[str]:
        """ Find the highest metric score per dataset and style it in boldface and underscored """
        metric = s.name[1]
        styling = []
        for d in self.datasets:
            style = (s.loc["SCORE", d] == self.best_metrics[d][metric])
            styling.append('text-decoration: underline; font-weight: bold' if style else '')


        return styling
    
    def style_dataframe(self, dataframe: pd.DataFrame) -> "pd.io.formats.style.Styler":
        styled_df = dataframe.style.apply(self.highlight_max, axis = 0)\
        .format(self.format_score, subset = pd.IndexSlice[pd.IndexSlice[:, self.get_raw_metrics()], :])\
        .format(self.format_percentage, subset = pd.IndexSlice[pd.IndexSlice[:, self.get_percentage_metrics()], :])\
        .format_index(escape="latex", axis=1)  \
        .format_index(escape="latex", axis=0)
        

        return styled_df
    
    def highlight_max_latex(self, s: pd.Series) -> list[str]:
        """ Find the highest metric score per dataset and style it in boldface and underscored """
        metric = s.name[1]
        styling = []
        for score in s:
            style = (score == self.best_metrics["MACRO"][metric])
            styling.append('text-decoration: underline; font-weight: bold' if style else '')


        return styling


    def style_dataframe_latex(self, dataframe: pd.DataFrame) -> "pd.io.formats.style.Styler":
        styled_df = dataframe.style.apply(self.highlight_max_latex, axis = 0)\
        .format(self.format_score, subset = pd.IndexSlice[:,("SCORE", table.get_raw_metrics())])\
        .format(self.format_percentage, subset = pd.IndexSlice[:,("SCORE", table.get_percentage_metrics())])\
        .format_index(escape="latex", axis=1)  \
        .format_index(escape="latex", axis=0)
        

        return styled_df


    def _get_best_metrics(self) -> dict[str, dict[str, float]]:
        """ Compile a dict of the highest score of each metric for each dataset 
        
        Returns a nested dict on the form: {DATSET: {METRIC: BEST_SCORE}}"""
        best = {}
        for dataset in self.datasets:
            best[dataset] = {}
            for metric in self.metrics:
                s = self.dataframe.loc[pd.IndexSlice[:, metric], pd.IndexSlice[:, dataset]]
                # MetricX should be minimized, the rest maximized
                best[dataset][metric] = s.min().item() if metric.startswith("MetricX") else s.max().item()

        return best

    def _parse_entry(self, entry: str, datasets: list[str]) -> tuple[list[tuple[str, str, str]], list[float]]:
        """Parse the evaluation of a single model

        Inputs:
            entry (str): A string representation of the single model evaluation, consisting of multiple rows 
                        (corresponding to the number of metrics) separated by \n
                        
            datasets (list[str]): A list of dataset names (shared by each row)
        
        Returns a list of tuples on the general form (MODEL_NAME, DATASET, METRIC) corresponding to the multiindex,
        and a list of float scores with length equal to the index tuples
        """
        rows = entry.strip().split("\n")
        model = rows[0].split(",")[0].strip()

        indices = []
        scores = []
        
        for i in range(len(rows)):
            values = rows[i].split(",")
            #First column is empty, second is the metric
            metric = values[1].strip()

            # Temporarily remove CHRF++
            if metric == "CHRF++":
                continue

            for j in range(2, len(values)):
                index = (model, datasets[j-2], metric)
                score = self.metric_to_float(values[j])
                
                indices.append(index)
                scores.append(score)
            
        
        return indices, scores


    def parse_multi_index(self) -> pd.DataFrame:
        """ Read an evaluation file on a specified format and parse it into a DataFrame """

        with open(self.file, "r", encoding = "utf-8") as in_file:
            data = in_file.read().strip().split("\n\n")

        header = data[0].split(",")

        direction = header[0].strip()
        datasets = [header[i].strip() for i in range(2, len(header))]

        indices = []
        scores = []
        
        for entry in data[1:]:
            i, s = self._parse_entry(entry, datasets)
            indices.extend(i)
            scores.extend(s)

        index = pd.MultiIndex.from_tuples(indices, names = ["MODEL", "DATASET", "METRIC"])
        df = pd.DataFrame(scores, index=index, columns=['SCORE'])
        df = df.unstack(level = "DATASET")

        return df


    def _reorder_metrics(self, df: pd.DataFrame, target_metric: str) -> pd.DataFrame:
        """ Sort the index so that the target metric appears first, the others alphabetically """

        # Regular way, metrics are the (second) rows
        if "METRIC" in df.index.names:
            ordered_indices = []

            for model in df.index.get_level_values("MODEL").unique().tolist():
                model_data = df.loc[model]

                sorted_metrics = sorted(model_data.index, key = lambda x: (x != target_metric, x))

                ordered_indices.extend([(model, metric) for metric in sorted_metrics])

            return df.reindex(ordered_indices)
        
        # Transposed way, metrics are the columns
        else:
            assert "METRIC" in df.columns.names, f"METRIC is not in the dataframe"
            ordered_columns = sorted(df.columns.tolist(), key = lambda x: (x[1] != target_metric, x[1]))
            # Reorder DataFrame columns
            return df[ordered_columns]


    def sort_df(self, df: pd.DataFrame, metric: str, dataset: str = "MACRO", style: bool = True) -> "pd.io.formats.style.Styler":
        """ Sort the dataframe based on metric and dataset and style it """
        
        ascending = metric.startswith("MetricX")
        # Reorder metric ordering
        sorted_index = self._reorder_metrics(df, metric)


        # Regular way, metrics are the (second) rows
        if "METRIC" in df.index.names:
            # Filter the DataFrame to only include rows of a specific metric and sort based on column value
            filtered_df = sorted_index.xs(metric, level='METRIC')
            
            # Sort the models based on the specific dataset in descending order
            sorted_models = filtered_df.sort_values(by=('SCORE', dataset), ascending=ascending).index.tolist()

        # Transposed way, metrics are the columns
        else:
            assert "METRIC" in df.columns.names, f"METRIC is not in the dataframe"
             # Filter the DataFrame to only include rows of a specific dataset and sort based on column value
            filtered_df = sorted_index.xs(dataset, level = "DATASET")

            # Sort the models based on the specific dataset in descending order
            sorted_models = filtered_df.sort_values(by=('SCORE', metric), ascending=ascending).index.tolist()

        # Reorder the original dataframe based on sorted models
        sorted_df = sorted_index.loc[sorted_models]
        
        if style:
            return self.style_dataframe(sorted_df)
        else:
            return sorted_df
    

    def transpose(self, df: pd.DataFrame):
        """Swap rows and columns, i.e. set METRIC to the top row and SCORE to the column (or vice versa)"""
        if "METRIC" in df.index.names:
            return df.stack(level = "DATASET", future_stack=True).unstack(level="METRIC")
        elif "DATASET" in df.index.names:
            return df.stack(level = "METRIC", future_stack=True).unstack(level="DATASET")
    
    def limit_dataset(self, df: pd.DataFrame, dataset: str):
        """Limit the rows to one dataset"""
        assert "DATASET" in df.index.names, f"DATASET is not in index, please transpose first"
        return df.xs(dataset, level = "DATASET")
    
    def limit_metric(self, df: pd.DataFrame, metric: str):
        """Limit the rows to one metric"""
        assert "METRIC" in df.index.names, f"METRIC is not in index, please transpose first"
        return df.xs(metric, level = "METRIC")
    

    def print_latex(self, dataset: str = "MACRO", sort_metric: str = "BLEU", truncate: bool = False, direction: str = None):
        df = self.transpose(self.dataframe)
        df = self.sort_df(df, metric = sort_metric, dataset = dataset, style = False)
        df = self.limit_dataset(df, dataset)
        df = self.style_dataframe_latex(df)
        latex_table = df.to_latex(convert_css = True).split("\n")
        if truncate:
            #Truncate names
            for i in range(4, len(latex_table)-2):
                line = latex_table[i]
                #Remove LORA
                line = re.sub(r'LORA\\_\d+\\_', 'LORA\\_', line)
                name, rest = line.split("&", 1)
                name = name[:36]
                if name[-1] == "\\":
                    name = name[:-1]
                new_line = f"{name} &{rest}"
                latex_table[i] = new_line


        # Add indentation
        latex_table = ["    "+l for l in latex_table]
        latex_table = ["    "+l if (i not in [0,1, len(latex_table)-2]) else l for i,l in enumerate(latex_table)]

        # Delete multicolumn line
        del latex_table[1]

        # Add 'table'
        latex_table = [
            r"\begin{table}[h]",
            r"   \hspace*{-2cm}",
            r"   \centering",
        ] + latex_table
        latex_table[-1] = r"   \caption{Evaluation results for the " + r"\_".join(direction.split("_")) + r" direction. The highest score for each metric is reported in boldface.}"
        latex_table.append(r"  \label{tab:results_" + direction + r"}")
        latex_table.append(r"\end{table}")

        latex_table = "\n".join(latex_table)
        print(latex_table)



if __name__ == "__main__":
    args = parse_args()
    #file = "data/nn_nb.csv"

    table = MetricTable(args.file)

    df = table.transpose(table.dataframe)
    df = table.sort_df(df, metric = "Comet", dataset = "MACRO", style = False)
    df = table.limit_dataset(df, "MACRO")
    table.print_latex(truncate = args.truncate, direction = args.file.split("/")[-1].split(".")[0])
    #df = table.sort_df("BLEU")
    #print(table.style_dataframe(df).to_latex(convert_css = True))
    

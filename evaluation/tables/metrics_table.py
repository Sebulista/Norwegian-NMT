""" Load metric data and read into pandas dataframes """
import pandas as pd

from argparse import ArgumentParser


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("file", type = str, help = "Which file to read to create a table")

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
        return f"{val:.2%}"
    
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
        styled_df = dataframe.style.apply(self.highlight_max, axis = 1)\
        .format(self.format_score, subset = pd.IndexSlice[pd.IndexSlice[:, self.get_raw_metrics()], :])\
        .format(self.format_percentage, subset = pd.IndexSlice[pd.IndexSlice[:, self.get_percentage_metrics()], :])

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
                if "@" in values[j]:
                    score = self.metric_to_float(values[j].split("@")[0])
                else:
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


    def _reorder_metrics(self, target_metric: str) -> pd.DataFrame:
        """ Sort the index so that the target metric appears first, the others alphabetically """

        ordered_indices = []

        for model in self.dataframe.index.get_level_values("MODEL").unique().tolist():
            model_data = self.dataframe.loc[model]

            sorted_metrics = sorted(model_data.index, key = lambda x: (x != target_metric, x))

            ordered_indices.extend([(model, metric) for metric in sorted_metrics])

        return self.dataframe.reindex(ordered_indices)


    def sort_df(self, metric: str, dataset: str = "MACRO") -> "pd.io.formats.style.Styler":
        """ Sort the dataframe based on metric and dataset and style it """
        
        ascending = metric.startswith("MetricX")
        # Reorder metric ordering
        sorted_index = self._reorder_metrics(metric)
        
        # Filter the DataFrame to only include rows of a specific metric and sort based on column value
        filtered_df = sorted_index.xs(metric, level='METRIC')
        
        # Sort the models based on the specific dataset in descending order
        sorted_models = filtered_df.sort_values(by=('SCORE', dataset), ascending=ascending).index.tolist()

        # Reorder the original dataframe based on sorted models
        sorted_df = sorted_index.loc[sorted_models]
        
        return self.style_dataframe(sorted_df)
    


if __name__ == "__main__":
    file = "data/nb_en.csv"

    table = MetricTable(file)

    df = table.dataframe
    table.style_dataframe(df).to_html()
    

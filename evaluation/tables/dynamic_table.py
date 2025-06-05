""" Streamlit application for interacting with the evaluation table """
import streamlit as st
st.set_page_config(layout="wide")

from pathlib import Path

from metrics_table import MetricTable

@st.cache_data
def get_file_options(folder: Path) -> list[str]:
    """ Get all files within a directory and cache the result"""
    options = set()
    for child in sorted(folder.iterdir()):
        if child.is_file():
            options.add(child.name)

    return sorted(list(options))


def file_selector(folder_path: str) -> Path:
    """ Select file with evaluation metrics per model

    The selectbox only show the file names to avoid path cluttering
     
    Returns the full path"""
    path = Path(folder_path)
    options = get_file_options(path)

    selected_file = st.selectbox("Select a directory", options)

    return path/selected_file


@st.cache_data
def get_dataframe(_table: MetricTable, metric: str, dataset: str, direction: str) -> "pd.DataFrame":
    """ Retrieve the sorted dataframe and cache the results.
    
    direction (str): is a dummy parameter to ensure the arguments create a unique state (for caching)
    _table (MetricTable): is underscored to not hash it for streamlit caching
    """
    return _table.sort_df(metric, dataset).to_html()

if __name__ == "__main__":
    file = file_selector("data/")

    table = MetricTable(file)

    metric = st.selectbox("Which metric to sort by:", table.metrics)
    dataset = st.selectbox("Which dataset to sort by:", table.datasets)

    st.write(get_dataframe(table, metric, dataset, file.name), unsafe_allow_html=True)

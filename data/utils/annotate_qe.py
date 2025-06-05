""" Streamlit application for annotating bitext """
import streamlit as st
import pandas as pd
from pathlib import Path

qe_folder = Path("/mnt/d/Master/data/test/QUALITY_ESTIMATION")
#input_file = qe_folder / "maalfrid.sample.en-nb"
input_file = qe_folder / "MultiParaCrawl-v9b.sample.nb-nn"
output_file = qe_folder / f"ANNOTATED_{input_file.name}"

assert(qe_folder.exists())

LANGMAP = {
    "en": "Engelsk",
    "nb": "Bokmål",
    "nn": "Nynorsk"
}

#st.set_page_config(layout="wide")
st.title("Annotate bitexts for quality estimation")

if 'initialized' not in st.session_state or not st.session_state.initialized:
    data = []
    if output_file.is_file():
        with open(output_file, "r", encoding = "utf-8") as in_file:
            data = in_file.read().strip().split("\n")
    st.session_state.current_index = len(data)
    st.session_state.initialized = True

def increment_append(row, max: int, keep: bool):
    increment_index(max)
    save_new_row(row, keep)

def increment_index(max: int):
    if st.session_state.current_index < max-1:
        st.session_state.current_index += 1


@st.cache_data
def load_csv(uploaded_file):
    column_names = [LANGMAP[lang] for lang in uploaded_file.name.split(".")[-1].split("-")]
    dtypes = {0: "object", 1: "object", 2: "float64"}
    df = pd.read_csv(uploaded_file, sep = "\t", names = column_names+["score"], dtype = dtypes)

    return  df.sort_values("score")

def save_new_row(row, keep: bool):
    #newfile = output_path / f"ANNOTATED_{st.session_state.file_name}"
    with open(output_file, "a", encoding = "utf-8") as out:
        out.write(f"{row.iloc[0]}\t{row.iloc[1]}\t{int(keep)}\n")

##Upload file
#uploaded_file = st.file_uploader("Choose bitext to annotate")
#if "file_name" not in st.session_state and uploaded_file is not None:
#    st.session_state.file_name = uploaded_file.name



df = load_csv(input_file)


st.subheader("Bitext")

st.table(df.iloc[[st.session_state.current_index]])

discard, add, _ = st.columns([1,1,4])

with discard:
    st.button("Discard", icon = "❌", on_click=increment_append, args = [df.iloc[st.session_state.current_index], len(df), False])

with add:
    st.button("Keep", icon = "✔️", on_click=increment_append, args = [df.iloc[st.session_state.current_index], len(df), True])

st.button("Next", on_click=increment_index, args = [len(df)])

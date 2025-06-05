""" A simple streamlit application to inspect a tsv file, intended for 
viewing a file of parallel sentences with clear distinction of the source and target """

import streamlit as st
import pandas as pd

st.set_page_config(layout="wide")

st.title("Visualize a tab separated file for better legibility")

uploaded_file = st.file_uploader("Choose the first bitext")
uploaded_file2 = st.file_uploader("Choose the second bitext")

if uploaded_file is not None and uploaded_file2 is not None:
    df = pd.read_csv(uploaded_file, sep="\t", names = ["Norsk", "Engelsk"])
    df2 = pd.read_csv(uploaded_file2, sep="\t", names = ["Norsk", "Engelsk"])

    data_container = st.container()
    with data_container:
        table1, table2 = st.columns(2)

        with table1:
            st.table(df)

        with table2:
            st.table(df2)

""" Basic streamlit webpage to inspect the cleaned texts and compare with xml"""

import streamlit as st
import pandas as pd

from pathlib import Path
from collections import defaultdict
import requests
from bs4 import BeautifulSoup

from pygments import highlight
from pygments.lexers import XmlLexer
from pygments.formatters import HtmlFormatter
from pygments.styles import get_all_styles
from xml.dom.minidom import parseString

st.set_page_config(layout="wide")

st.title("Compare bitexts")


styles = ["abap", "abnf", "actionscript", "ada", "agda", "al", "antlr4", "apacheconf", "apex", "apl", "applescript", "aql", "arduino", "arff", "asciidoc", "asm6502", "asmatmel", "aspnet", "autohotkey", "autoit", "avisynth", "avroIdl (avro-idl)", "bash", "basic", "batch", "bbcode", "bicep", "birb", "bison", "bnf", "brainfuck", "brightscript", "bro", "bsl", "c", "cfscript", "chaiscript", "cil", "clike", "clojure", "cmake", "cobol", "coffeescript", "concurnas", "coq", "cpp", "crystal", "csharp", "cshtml", "csp", "cssExtras (css-extras)", "css", "csv", "cypher", "d", "dart", "dataweave", "dax", "dhall", "diff", "django", "dnsZoneFile (dns-zone-file)", "docker", "dot", "ebnf", "editorconfig", "eiffel", "ejs", "elixir", "elm", "erb", "erlang", "etlua", "excelFormula (excel-formula)", "factor", "falselang (false)", "firestoreSecurityRules (firestore-security-rules)", "flow", "fortran", "fsharp", "ftl", "gap", "gcode", "gdscript", "gedcom", "gherkin", "git", "glsl", "gml", "gn", "goModule (go-module)", "go", "graphql", "groovy", "haml", "handlebars", "haskell", "haxe", "hcl", "hlsl", "hoon", "hpkp", "hsts", "http", "ichigojam", "icon", "icuMessageFormat (icu-message-format)", "idris", "iecst", "ignore", "inform7", "ini", "io", "j", "java", "javadoc", "javadoclike", "javascript", "javastacktrace", "jexl", "jolie", "jq", "jsExtras (js-extras)", "jsTemplates (js-templates)", "jsdoc", "json", "json5", "jsonp", "jsstacktrace", "jsx", "julia", "keepalived", "keyman", "kotlin", "kumir", "kusto", "latex", "latte", "less", "lilypond", "liquid", "lisp", "livescript", "llvm", "log", "lolcode", "lua", "magma", "makefile", "markdown", "markupTemplating (markup-templating)", "markup", "matlab", "maxscript", "mel", "mermaid", "mizar", "mongodb", "monkey", "moonscript", "n1ql", "n4js", "nand2tetrisHdl (nand2tetris-hdl)", "naniscript", "nasm", "neon", "nevod", "nginx", "nim", "nix", "nsis", "objectivec", "ocaml", "opencl", "openqasm", "oz", "parigp", "parser", "pascal", "pascaligo", "pcaxis", "peoplecode", "perl", "phpExtras (php-extras)", "php", "phpdoc", "plsql", "powerquery", "powershell", "processing", "prolog", "promql", "properties", "protobuf", "psl", "pug", "puppet", "pure", "purebasic", "purescript", "python", "q", "qml", "qore", "qsharp", "r", "racket", "reason", "regex", "rego", "renpy", "rest", "rip", "roboconf", "robotframework", "ruby", "rust", "sas", "sass", "scala", "scheme", "scss", "shellSession (shell-session)", "smali", "smalltalk", "smarty", "sml", "solidity", "solutionFile (solution-file)", "soy", "sparql", "splunkSpl (splunk-spl)", "sqf", "sql", "squirrel", "stan", "stylus", "swift", "systemd", "t4Cs (t4-cs)", "t4Templating (t4-templating)", "t4Vb (t4-vb)", "tap", "tcl", "textile", "toml", "tremor", "tsx", "tt2", "turtle", "twig", "typescript", "typoscript", "unrealscript", "uorazor", "uri", "v", "vala", "vbnet", "velocity", "verilog", "vhdl", "vim", "visualBasic (visual-basic)", "warpscript", "wasm", "webIdl (web-idl)", "wiki", "wolfram", "wren", "xeora", "xmlDoc (xml-doc)", "xojo", "xquery", "yaml", "yang", "zig"]

def get_file_name(path):
    name = path.name

    name = ".".join(name.split(".")[:-1])

    return name

#get document name
def get_doc_name(url):
    name = url.split("/")[-1].split(".")
    if name[-3] in ["eng", "nob", "no"]:
        name = ".".join(name[:-3])
    else:
        name = ".".join(name[:-2])

    return name

#get language
def get_doc_lang(url):
    url_split = url.split("//")[-1].split("/")
    lang = url_split[3]

    return lang

@st.cache_data
def get_options(folder: Path):
    options = set()
    for child in sorted(folder.iterdir()):
        options.add(child.name)

    return sorted(list(options))

def folder_selector(folder_path, selected):
    path = Path(folder_path)
    options = get_options(path)

    index = 0
    if selected != None:
        index = options.index(selected.name)
        
    selected_folder = st.selectbox("Select a directory", options, on_change = reset_index, args = [True], index=index)

    return path / selected_folder


def grandchild_selector(parent, selected):
    options = get_options(parent)

    index = 0
    if selected != None:
        index = options.index(selected.name)

    selected_folder = st.selectbox("Select a subdirectory", options, on_change = reset_index, index=index)
    return parent / selected_folder


def reset_index(reset_grandchild=False):
    st.session_state.current_index = 0
    if reset_grandchild:
        st.session_state.selected_grandchild = None

def increment_index(max: int):
    if st.session_state.current_index < max-1:
        st.session_state.current_index += 1

def decrement_index():
    if st.session_state.current_index > 0:
        st.session_state.current_index -= 1


@st.cache_data
def get_files(folder: Path):
    files = sorted([file for file in folder.iterdir() if file.is_file()])
    uniq_files = {get_file_name(file) for file in files}
    name_to_path = defaultdict(list)

    for file in files:
        file_name = get_file_name(file)
        if file_name in uniq_files:
            name_to_path[file_name].append(file)

    int_to_name = {i: name for i, name in enumerate(name_to_path.keys())}

    return name_to_path, int_to_name

@st.cache_data
def get_urls(subfolder: Path, file_name: str):
    lang_map = {"nob": "nb", "eng" : "en", "nno" : "nn"}
    # Path to folder that holds aligned urls
    url_folder = Path("path/to/urls")
    file = (url_folder / subfolder / "s-t.txt")

    with open(file, "r", encoding = "utf-8") as f:
        lines = f.read().strip().split("\n\n")

    urls = {}

    for line in lines:
        segments = line.split("\n")
        url = segments[0].split()[-1]

        if get_doc_name(url) == file_name:
            for segment in segments:
                url = segment.split()[-1]
                lang = lang_map[get_doc_lang(url)]

                urls[lang] = url

            return urls

    
def remove_extra_newlines(xml_string):
    """Remove extra new lines from the pretty-printed XML."""
    lines = xml_string.split('\n')
    cleaned_lines = [line for line in lines if line.strip() != '']
    return '\n'.join(cleaned_lines)

@st.cache_data
def get_raw_xml(url):
    try:
        response = requests.get(url, timeout=2)
    except requests.exceptions.ConnectTimeout:
        return None
    response.encoding = 'utf-8'
    xml_content = response.text

    soup = BeautifulSoup(xml_content, "lxml-xml")
    #body = soup.find("body")

    #return str(soup)
    #return soup.prettify()

    dom = parseString(xml_content)
    pretty_xml = dom.toprettyxml(indent="    ")  # Using 4 spaces for indentation
    cleaned_pretty_xml = remove_extra_newlines(pretty_xml)
    return cleaned_pretty_xml

def render_xml(xml_code, style='friendly'):
    # Generate the highlighted code with the selected style
    formatter = HtmlFormatter(style=style, lineseparator="<br>")
    highlighted_code = highlight(xml_code, XmlLexer(), formatter)
    # Use custom CSS to improve the highlighting style
    css = f"<style>{formatter.get_style_defs()}</style>\n"
    # Display HTML content in Streamlit
    st.markdown(css + highlighted_code, unsafe_allow_html=True)

def set_main():
    st.session_state.view = "main"

def detailed(text: pd.DataFrame, xml: str):
    st.session_state.view = "detailed"
    st.session_state.text = text
    st.session_state.xml = xml


def display_compare_xml():
    st.button("Back", on_click=set_main)
    c1,c2 = st.columns(2)
    with c1:
        st.table(st.session_state.text)

    with c2:
        #selected_style = st.selectbox('Select Pygments Style', styles)
        selected_style = "hcl"
        st.code(st.session_state.xml, language=selected_style, wrap_lines = True)
        # List available Pygments styles
        #styles = list(get_all_styles())

        # Select a style using Streamlit's selectbox
        #selected_style = st.selectbox('Select Pygments Style', styles)
        #render_xml(st.session_state.xml, selected_style)

def read_file(file_path: Path):
    with open(file_path, "r", encoding = "utf-8") as f:
        content = f.read().strip()
    lines = content.split("\n")

    df = pd.DataFrame(lines, columns=[file_path.name])

    return df

def main():    
    folder = folder_selector("path/to/cleaned_texts", st.session_state.selected_folder)
    st.session_state.selected_folder = folder
    grandchild = grandchild_selector(folder, st.session_state.selected_grandchild)
    st.session_state.selected_grandchild = grandchild


    name_to_path, int_to_name = get_files(grandchild)

    data_container = st.container()
    with data_container:
        files = name_to_path[int_to_name[st.session_state.current_index]]

        urls = get_urls(Path(f"{folder.name}/{grandchild.name}"), int_to_name[st.session_state.current_index])
        df = pd.DataFrame(sorted(urls.items()))
        st.markdown(df.to_html(render_links=True, index=False, header = False, border = 0),unsafe_allow_html=True)

        st.write(f"{st.session_state.current_index}/{len(name_to_path)-1}")
        st.write(int_to_name[st.session_state.current_index])

        top_cols = st.columns([1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1])
        with top_cols[1]:
            st.button("next", on_click=increment_index, args = [len(name_to_path)],key = "top_next")
        with top_cols[0]:
            st.button("previous", on_click=decrement_index, key = "top_previous")

        N_columns = len(files)

        cols = st.columns(N_columns)

        with cols[0]:
            text_1 = read_file(files[0])
            placeholder_button_one = st.empty()
            with placeholder_button_one.container():
                st.write(":green[LOADING XML]")
            st.table(text_1)

        with cols[1]:
            text_2 = read_file(files[1])
            placeholder_button_two = st.empty()
            with placeholder_button_two.container():
                st.write(":green[LOADING XML]")
            st.table(text_2)

        if N_columns == 3:
            with cols[2]:
                text_3 = read_file(files[2])
                placeholder_button_three = st.empty()
                with placeholder_button_three.container():
                    st.write(":green[LOADING XML]")
                st.table(text_3)

        bottom_cols = st.columns([1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1])
        with bottom_cols[1]:
            st.button("next", on_click=increment_index, args = [len(name_to_path)], key = "bottom_next")
        with bottom_cols[0]:
            st.button("previous", on_click=decrement_index, key = "bottom_previous")

        #LOAD XML
        #1
        with placeholder_button_one.container():
            xml = get_raw_xml(urls[files[0].name.split(".")[-1]])
            button = st.button("Compare XML", on_click=detailed, args = [text_1, xml], key = "one")
        #2
        with placeholder_button_two.container():
            xml = get_raw_xml(urls[files[1].name.split(".")[-1]])
            st.button("Compare XML", on_click=detailed, args = [text_2, xml], key = "two")
        #3
        if N_columns == 3:
            with placeholder_button_three.container():
                xml = get_raw_xml(urls[files[2].name.split(".")[-1]])
                st.button("Compare XML", on_click=detailed, args = [text_3, xml], key = "three")

if __name__ == "__main__":
    if 'initialized' not in st.session_state or not st.session_state.initialized:
        st.session_state.current_index = 0
        st.session_state.selected_folder = None
        st.session_state.selected_grandchild = None
        st.session_state.view = "main"

        st.session_state.initialized = True

    if st.session_state.view == "main":
        main()
    elif st.session_state.view == "detailed":
        display_compare_xml()

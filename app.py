"""Main Module"""
import streamlit as st
import src.pages.Home
import src.pages.AttrGrad
import src.pages.AttrAttention
import src.pages.AttrAdditionalExps

PAGES = {
    "Home": src.pages.Home,
    "[Attribution] Gradient": src.pages.AttrGrad,
    "[Attribution] Attention": src.pages.AttrAttention,
    "[Attribution] Additional Experiments": src.pages.AttrAdditionalExps
}

def main():
    """Main function of the App"""
    st.set_page_config(
        page_title="Demo for XAI", 
        page_icon=":face_with_monocle:",
        layout="centered", # "wide",
        # initial_sidebar_state="collapsed"
    )
    st.sidebar.title("Navigation")
    selection = st.sidebar.radio("Go to", list(PAGES.keys()))
    page = PAGES[selection]
    with st.spinner(f"Loading {selection} ..."):
        page.write()

if __name__ == "__main__":
    main()
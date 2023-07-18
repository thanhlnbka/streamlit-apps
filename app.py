import streamlit as st 
import pandas as pd 



st.write("""
# Simple app predict type flower


""")
st.sidebar.header("User input parameter")


def test(key="1"): 
    s_length = st.sidebar.slider("S lenght",0.0,1.0,0.1, key=f"{key}_length")
    s_width = st.sidebar.slider("S width", 2,8,4, key=f"{key}_width")

    data = {
        "length": s_length, 
        "width": s_width
    }
    return pd.DataFrame(data, index=[0])


df = test(key="1")
df1 = test(key="2")
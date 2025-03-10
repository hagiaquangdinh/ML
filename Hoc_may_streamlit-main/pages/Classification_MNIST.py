import streamlit as st
from buoi4.Classification import Classification

if "last_page" in st.session_state and st.session_state.last_page != "Classification":
    st.session_state.clear()  # Xóa toàn bộ session

st.session_state.last_page = "Classification" 

st.title("🔢 Classification MNIST")


# Gọi hàm Classification từ module
Classification()

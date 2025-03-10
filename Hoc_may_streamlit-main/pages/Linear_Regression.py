import streamlit as st
from buoi2.tien_su_ly_du_lieu import main

if "last_page" in st.session_state and st.session_state.last_page != "linear":
    st.session_state.clear()  # Xóa toàn bộ session

st.session_state.last_page = "linear" 


st.title("📈 Linear Regression")


# Gọi hàm main từ module
main()

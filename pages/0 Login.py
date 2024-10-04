import streamlit as st
from dotenv import load_dotenv


def authenticate(username: str, password: str):
    
    if username == "HR" and password == "1234":
        return ["HR", "Legal", "Commercial", "Domain 5", "Domain 6", "Domain 7"]
    
    elif username == "IT" and password == "1234":
        return ["IT", "Domain 8", "Domain 9", "Domain 10"]
    
    elif username == "admin" and password == "1234":
        return ["HR", "IT", "Legal", "Commercial", "Domain 5", "Domain 6", "Domain 7", "Domain 8", "Domain 9", "Domain 10"]
    
    else:
        return None
    
    
    
def main():
    
    #setting the page configuration such as page title and other things
    st.set_page_config(page_title='chat with your Documents', page_icon=':books:')
    st.image("LOGO.png", width=455)

    st.title("Log In")
    
    if "domains" not in st.session_state:
        st.session_state.domains = None
        
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False
    
    username = st.text_input("Please enter your username")
    password = st.text_input("Please enetr your password", type = "password")
        
    if st.button("Login"):
        
        domains = authenticate(username, password)
        
        if domains:
            st.success("Login Successful")
            st.session_state.domains = domains
            st.session_state.logged_in = True
        else:
            st.warning("Please enter the correct credentials")
            
            
            
if __name__ == "__main__":
    main()
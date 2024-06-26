import streamlit as st,     datetime


def st_core():
    if "messages" not in st.session_state:      st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):  
            st.markdown(message["content"])

    if prompt := st.chat_input("What is up?"):
        with st.chat_message("user"):           
            st.markdown(prompt)
        st.session_state.messages.append({"role": "user",    "content": prompt})

    response = f"i Echo: {prompt}"
    with st.chat_message("assistant"):          
        st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})   

def st_allin1():
    st.code("""
            connection.close()    
    """)
    

    st.title("Form for the Users")
    st.write("Here, you can answer to some questions in this form.")

    user_id = st.text_input("ID", value="Your ID", max_chars=7)
    age     = st.number_input("Age", min_value=18, max_value=100, step=1)
    b_date  = st.date_input("Date of Birth", min_value=datetime.date(1921, 1, 1),        max_value=datetime.date(2033, 12, 31))
    smoke   = st.checkbox("Do you smoke?")
    genre   = st.radio("Which movie genre do you like?",                                      options=['horror', 'adventure', 'romantic'])
    weight  = st.slider("Choose your weight", min_value=40., max_value=150., step=0.5)
    p_form  = st.selectbox("Select level of your physical condition",                 options=["Bad", "Normal", "Good"])
    colors  = st.multiselect('What are your favorite colors',                                options=['Green', 'Yellow', 'Red', 'Blue', 'Pink'])
    info    = st.text_area("Share some information about you", "Put information here",         help='You can write about your hobbies or family')
    image   = st.file_uploader("Upload your photo", type=['jpg', 'png'])

    click = st.sidebar.button('Click me!')
    if click:
        st.sidebar.write("You clicked the button")

    col1, col2 = st.columns(2)
    with col1:
        st.image("https://static.streamlit.io/examples/cat.jpg", width=300)
        st.button("Like cats")
    with col2:
        st.image("https://static.streamlit.io/examples/dog.jpg", width=355)
        st.button("Like dogs")

    submit = st.button("Submit")
    if submit:
        st.write("You submitted the form")    
    st.image("./images/zhang.gif")

def st_mongo():
    st.code("""
        pass
    """)

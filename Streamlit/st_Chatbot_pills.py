import streamlit as st
from streamlit_pills import pills

st.title("Echo Bot")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# State variable to store selected prompt
if "selected_prompt" not in st.session_state:
    st.session_state.selected_prompt = None

# Update selected prompt based on pill selection
selected = pills("Label", ["Option 1", "Option 2", "Option 3"], ["🍀", "🎈", "🌈"])
st.session_state.selected_prompt = selected

# Chat input with conditional placeholder
default_text = st.session_state.selected_prompt if st.session_state.selected_prompt else "What is up?"
prompt = st.chat_input(placeholder=default_text)

# React to user input
if prompt:  # Check if prompt has a value (avoids unnecessary processing)
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)

    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    response = f"Echo: {prompt}"
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(response)

    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})

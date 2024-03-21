import streamlit as st

st.set_page_config(page_title="Fullstack-GPT", page_icon="🚆")
st.title("FullStack-GPT")


if "messages" not in st.session_state:
    st.session_state["messages"] = []


def send_message(message, role, save=True):
    with st.chat_message(role):
        st.write(message)
    if save:
        st.session_state["messages"].append({"message": message, "role": role})


for message in st.session_state["messages"]:
    send_message(
        message=message["message"],
        role=message["role"],
        save=False,
    )

message = st.chat_input("input keyword")

if message:
    send_message(message, "human")

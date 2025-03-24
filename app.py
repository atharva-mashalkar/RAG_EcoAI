import sys
sys.modules["torch.classes"] = None

import time
import warnings
import streamlit as st
from source.utils import MAIN_CSS
from source.core import SessionManager, login, signup, process_query

warnings.simplefilter("ignore", category=DeprecationWarning)

def _clear_input():
    session_id = st.query_params["session_id"]
    st.session_state['session_data'][session_id]["query"] = st.session_state["user_input"]
    st.session_state["session_data"][session_id]["disabled_input"] = True

def _stream_response():
    session_id = st.query_params["session_id"]
    for word in st.session_state['session_data'][session_id]['ai_response'].split(" "):
        yield word + " "
        time.sleep(0.02)

def _get_response(query):
    response, sources = process_query(query)
    return response, sources
            
def _update_query_params(session_id, username, expires_at):
    st.query_params["session_id"] = session_id
    st.query_params["username"] = username
    st.query_params["expires_at"] = expires_at

def _show_signup_form():
    with st.form(key='signup_form', border=False):
        st.title("Signup")
        firstname = st.text_input("First Name", key="signup_firstname")
        lastname = st.text_input("Last Name", key="signup_lastname")
        username = st.text_input("Username", key="signup_username")
        password = st.text_input("Password", type="password", key="signup_password")
        submit_button = st.form_submit_button("Submit")

        if submit_button:
            success, message = signup(firstname, lastname, username, password)
            if success:
                session_id = session_manager.create_session(username)
                session = st.session_state['session_data'][session_id]
                _update_query_params(session_id, username, session["expires_at"])
                st.session_state["show_signup"] = False
                st.rerun()
            st.write(message)

def _show_login_form():
    with st.form(key='login_form', border=False):
        st.title("Login")
        username = st.text_input("Username", key="login_username")
        password = st.text_input("Password", type="password", key="login_password")
        submit_button = st.form_submit_button("Submit")

        if submit_button:
            success, message = login(username, password)
            st.write(message)
            if success:
                session_id = session_manager.create_session(username)
                session = st.session_state['session_data'][session_id]
                _update_query_params(session_id, username, session["expires_at"])
                st.session_state["show_login"] = False
                st.rerun()
            else:
                st.session_state["show_login"] = True


def _add_logout_button(session_manager, session_id):
    if session_id in st.session_state['session_data'] and st.session_state['session_data'][session_id]:
        if st.button("Logout"):
            session_manager.delete_session()
            _update_query_params('', '', '')
            st.rerun()

def _add_signup_login_buttons():
    login_col1, login_col2 = st.columns(2)
    with login_col1:
        if st.button("Login"):
            st.session_state["show_login"] = True
            st.session_state["show_signup"] = False
    with login_col2:
        if st.button("Signup"):
            st.session_state["show_signup"] = True
            st.session_state["show_login"] = False


if __name__ == "__main__":
    session_manager = SessionManager()
    session_id = session_manager.validate_session()

    st.set_page_config(page_title="EcoAI")

    st.markdown(MAIN_CSS, unsafe_allow_html=True)

    col1, col2 = st.columns([1, 5])
    with col1:
        st.image("./source/utils/logo.jpeg", width=100)

    with col2:
        st.markdown("# EcoAI")
        if not session_id or not session_id in st.session_state['session_data'] or not st.session_state['session_data'][session_id]['authenticated']:
            _add_signup_login_buttons()

    form_col1, form_col2 = st.columns(2)
    with form_col1:
        if st.session_state["show_login"]:
            _show_login_form()
    with form_col2:
        if st.session_state["show_signup"]:
            _show_signup_form()

    _add_logout_button(session_manager, session_id)

    if session_id and session_id in st.session_state['session_data'] and st.session_state['session_data'][session_id]['authenticated']:
        for session_message in st.session_state['session_data'][session_id]["messages"]:
            if session_message[0] == 'user':
                st.chat_message("Human", avatar=None).write(session_message[1], unsafe_allow_html=True)
            else:
                source = ""
                for citation in session_message[2]:
                    source += f"\n- {citation.pdf_path}, Page {citation.page_number}"
                ai_message = f"{session_message[1]}\n\nSource:{source}"
                if session_message[3]:
                    st.session_state['session_data'][session_id]["ai_response"] = ai_message
                    st.chat_message("AI", avatar="./source/utils/logo.jpeg").write_stream(_stream_response)
                    st.session_state['session_data'][session_id]["ai_response"] = ""
                    st.session_state['session_data'][session_id]["messages"][-1][3] = False
                else:
                    st.chat_message("AI", avatar="./source/utils/logo.jpeg").write(ai_message, unsafe_allow_html=True)

        if st.session_state['session_data'][session_id]['waiting_for_response'] and st.session_state['session_data'][session_id]["disabled_input"]:
            with st.status("Searching for relevant context in Database...."):
                ai_response, sources = _get_response(st.session_state['session_data'][session_id]['query'])
            st.session_state['session_data'][session_id]["disabled_input"] = False
            st.session_state['session_data'][session_id]['query'] = ""
            st.session_state['session_data'][session_id]['waiting_for_response'] = False
            st.session_state['session_data'][session_id]["messages"].append(["ai", ai_response, sources, True])
            st.rerun()

        if st.session_state['session_data'][session_id]["query"] and not st.session_state['session_data'][session_id]['waiting_for_response'] and st.session_state['session_data'][session_id]["disabled_input"]:
            st.session_state['session_data'][session_id]["messages"].append(["user", st.session_state['session_data'][session_id]["query"], "", ""])
            st.session_state['session_data'][session_id]['waiting_for_response'] = True
            st.rerun()

        st.chat_input(placeholder="Ask your question...", key="user_input", disabled=st.session_state['session_data'][session_id]["disabled_input"], on_submit=_clear_input)

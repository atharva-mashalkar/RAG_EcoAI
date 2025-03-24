import uuid
import time
import streamlit as st
from datetime import timedelta, datetime
from config import SESSION_DURATION

class SessionManager:
    def __init__(self, session_duration=timedelta(minutes=SESSION_DURATION)):
        self.session_duration = session_duration
        self._initialize_session_state()


    def create_session(self, username):
        session_id = str(uuid.uuid4())
        expires = datetime.now() + self.session_duration
        sessions = self._load_sessions()
        sessions = self._initialize_user_state(sessions, session_id, username, expires.timestamp())
        self._update_sessions(sessions[session_id], session_id)
        return session_id


    def validate_session(self):
        query_params = st.query_params
        session_id = query_params.get("session_id")
        username = query_params.get("username")
        expires_at = query_params.get("expires_at")
        if session_id:
            sessions = self._load_sessions()
            if float(expires_at) < time.time():
                self.delete_session(session_id)
                return None
            elif session_id not in st.session_state['session_data']:
                sessions = self._initialize_user_state(sessions, session_id, username, expires_at)
                self._update_sessions(sessions[session_id], session_id)
            return session_id
        return None


    def delete_session(self, session_id=None):
        if not session_id:
            session_id = st.query_params.get("session_id")
        if session_id:
            sessions = self._load_sessions()
            sessions.pop(session_id, None)
            self._save_sessions(sessions)


    @staticmethod
    def _initialize_user_state(sessions, session_id, username, expires_at):
        sessions[session_id] = {
            "username": username,
            "expires_at": expires_at,
            "messages" : [],
            "query" : "",
            "ai_response" : "",
            "disabled_input" : False,
            "waiting_for_response" : False,
            "authenticated" : True  
        }
        return sessions


    @staticmethod
    def _load_sessions():
        return st.session_state.get('session_data', {})


    @staticmethod
    def _update_sessions(session, session_id):
        if session_id:
            st.session_state['session_data'][session_id] = session
    

    @staticmethod
    def _save_sessions(sessions):
        st.session_state['session_data'] = sessions


    @staticmethod
    def _initialize_session_state():
        if "session_data" not in st.session_state:
            st.session_state['session_data'] = {}
        if "show_login" not in st.session_state:
            st.session_state["show_login"] = False
        if "show_signup" not in st.session_state:
            st.session_state["show_signup"] = False

MAIN_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Barlow:wght@300;400;600&family=Barlow+Condensed:wght@300;400;600&display=swap');
    
    body {
        font-family: 'Barlow', sans-serif;
        background: linear-gradient(135deg, #ff9a9e, #fad0c4);
        color: #3D3D3D;
    }
    .stChatMessage {
        font-family: 'Barlow Condensed', sans-serif;
        padding: 10px;
        border-radius: 10px;
    }
    .stChatMessage.Human {
        background-color: rgba(239, 232, 220, 0.8);
        color: #3D3D3D;
        text-align: left;
    }
    .stChatMessage.AI {
        background-color: #ffcc00;
        color: #3D3D3D;
        text-align: right;
    }
    .stChatInput input {
        font-family: 'Barlow', sans-serif;
        border: 2px solid #ff9a9e;
    }
    .login-button {
        width: 100%;
    }
    div.stApp {
        padding-top: 0rem;
    }
    .login-button {
        width: 100%;
    }
    .stForm {
        border: none;
        padding-top: 0px;
        width: 70% !important;
        margin: 0 auto;
    }
</style>
"""

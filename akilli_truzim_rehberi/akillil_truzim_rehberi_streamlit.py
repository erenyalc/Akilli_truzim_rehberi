"""
web'de çalışan chatbot ekranı
"""

import streamlit as st
from langchain_community.chat_models import ChatOllama
from langchain_classic.schema import SystemMessage, HumanMessage
from langchain_classic.memory import ConversationBufferMemory
from langchain_classic.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_classic.callbacks.base import BaseCallbackHandler
from typing import Any

#streamlit için özel streaming callback
class StreamHandler(BaseCallbackHandler):
    def __init__(self, placeholder):
        self.placeholder = placeholder
        self.final_text = ""

    def on_llm_new_token(self, token:str, **kwargs:Any) -> None:
        self.final_text += token #token'ları birleştir
        self.placeholder.markdown(self.final_text + " ")#canlı yazma

st.set_page_config(page_title = "Akıllı Truzim Rehberi", page_icon= "✈")
st.title("✈ Akıllı turizm rehberi")
st.markdown("Türkiye'nin dört bir yanındaki turistik yerler hakkında bilgi almak için iletişime geçebilirsiniz.")

#session state: kullanıcı geçmişini tutmak için

if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(return_messages= True)

user_input = st.chat_input("Bir şehir, mekan ya da yemek aktivitesi sorabilirsiniz:")

    #sohbet geçmişini arayüzde göster
for msg in st.session_state.memory.chat_memory.messages:
    if isinstance(msg, HumanMessage):
        with st.chat_message("Kullanıcı:"):
            st.markdown(msg.content)
    else:
        with st.chat_message("Akıllı rehber:"):
            st.markdown(msg.content)

if user_input:
    st.session_state.memory.chat_memory.add_user_message(user_input)
    with st.chat_message("Kullanıcı:"):
        st.markdown(user_input)

    with st.chat_message("Akıllı rehber:"):
        response_placeholder = st.empty()#geçici mesaj kutusu
        stream_handler = StreamHandler(response_placeholder)
        llm = ChatOllama(model = "gemma3:4b", streaming = True, callbacks = [stream_handler])

        #tüm konuşmayı modele verecek şekilde mesajları oluştur.
        messages = [
            SystemMessage(content= "Sen bir turizm rehberisin. Kullanıcıya Türkiye'deki tarihi yerler, şehirler, yemekler, ulaşım araçları gibi konularda öneriler ver.")
        ] + st.session_state.memory.load_memory_variables({})["history"] + [HumanMessage(content=user_input)]

        #yanıtlar ve yanıtı hafızaya kaydetme
        response = llm.invoke(messages)
        st.session_state.memory.chat_memory.add_ai_message(response.content)
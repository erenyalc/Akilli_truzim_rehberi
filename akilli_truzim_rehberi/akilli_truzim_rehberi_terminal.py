"""
Akıllı truzim rehberi (LLM)
Kullanıcı Türkiye truzmi ile ilgili yazılı sorular sorar- cevap alır.
Google Gemma3:4b modelini Ollama framework'ü üzerinden kullanıldı.
Streamlit ile arayüz oluşturuldu.
Local'de çalışıyor.

pip install langchain-community streamlit pydantic requests

Ollama: açık kaynaklı platform. LLM'leri Local'de çalışmamıza olanak sağlıyor.
https://ollama.com
https://ollama.com/library/gemma3:4b
"""

from langchain_community.chat_models import ChatOllama
from langchain_classic.schema import SystemMessage, HumanMessage # Chat mesaj sınıflandırıcı
from langchain_classic.memory import ConversationBufferMemory

llm = ChatOllama(model = "gemma3:4b")
#memory
memory = ConversationBufferMemory(return_messages= True)
print("Akıllı truzim rehberine hoşgeldiniz")
print("Size gezilecek yerler, tatil yerleri ve ulaşım bilgileri gibi konularda yardımcı olabilirim.")

while True:
    user_input = input("Siz:")

    if user_input.lower() in ["q", "bye"]:
        print("Program sonlandırıldı.")
        break

    #save memory
    memory.chat_memory.add_user_message(user_input)

    #sistem mesajı, memory, human message
    message = [
        SystemMessage(content= "Sen bir truzim rehberisin. Kullanıcılara Türkiye'deki şehirler, tarihi yerler, yöresel yemekler, ulaşım ve tatil sorularu hakkında yardımcı ol")
    ] + memory.load_memory_variables({})["history"] + [HumanMessage(content=user_input)]

    response = llm.invoke(message)

    #model mesajını hafızaya ekle
    memory.chat_memory.add_ai_message(response.content)
    print(f"Akıllı truzim rehberi: {response.content}")


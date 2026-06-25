from backend.core import run_llm
from typing import List, Any
import streamlit as st

def format_sources(context_docs:List[Any])-> List[str]:
    sources: List[str] = []
    for doc in (context_docs or []):
        docs = doc if isinstance(doc, list) else [doc]
        for item in docs:
            meta = getattr(item, "metadata", None) or {}
            sources.append(str(meta.get("source", "Unknown")))
    return sources

st.set_page_config(page_title="LangChain Documentation Retriever", layout="centered")
st.title("LangChain Documentation Retriever")


with st.sidebar:
    st.subheader("Session")
    if st.button("Clear Chat", use_container_width=True):
        st.session_state.pop("messages", None)
        st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = [{
        "role":"assistant",
        "content":"Hello! I am a LangChain documentation assistant. Ask me anything about LangChain and I will try to help you.",
        "sources":[],
    }
    ]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["sources"]:
            for source in msg["sources"]:
                st.markdown(f"- {source}")

prompt = st.chat_input("Ask me anything about LangChain documentation...")

if prompt:
    st.session_state.messages.append({"role":"user","content":prompt,"sources":[]})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):

        try:
            with st.spinner("Retrieving answer from LangChain documentation..."):
                response = run_llm(prompt)
                answer = str(response.get("answer", "")).strip() or "(No answer returned)"
                print("length of context_docs:", len(response.get("context")))
                sources = format_sources(response.get("context"))
            
            st.markdown(answer)
            if sources:
                with st.expander("Sources"):
                    for source in sources:
                        st.markdown(f"- {source}")
            
            st.session_state.messages.append({"role":"assistant","content":answer,"sources":sources})
        except Exception as e:
            st.error(f"Error: {e}")
            st.exception(e)



    





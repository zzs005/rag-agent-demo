from __future__ import annotations

import requests
import streamlit as st


API_BASE = "http://127.0.0.1:8000"


st.set_page_config(page_title="KB Agent Demo", layout="wide")
st.title("KB Question Answering Agent")

if "conversation_id" not in st.session_state:
    st.session_state.conversation_id = None
if "messages" not in st.session_state:
    st.session_state.messages = []

tab_upload, tab_chat, tab_search = st.tabs(["Upload", "Agent Chat", "Search Debug"])


with tab_upload:
    st.subheader("Upload PDF")
    uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])

    if uploaded_file is not None and st.button("Upload and Rebuild Index"):
        with st.spinner("Uploading document and rebuilding index..."):
            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
            try:
                resp = requests.post(f"{API_BASE}/upload", files=files, timeout=600)
                if resp.ok:
                    st.success("Upload succeeded.")
                    st.json(resp.json())
                else:
                    st.error(f"Upload failed: {resp.status_code}")
                    st.json(resp.json())
            except Exception as exc:
                st.error(f"Request failed: {exc}")


with tab_chat:
    st.subheader("Multi-turn knowledge base chat")

    col1, col2 = st.columns([4, 1])
    with col2:
        if st.button("Clear Session"):
            st.session_state.conversation_id = None
            st.session_state.messages = []
            st.rerun()

    with col1:
        if st.session_state.conversation_id:
            st.caption(f"conversation_id: {st.session_state.conversation_id}")
        else:
            st.caption("conversation_id will be created automatically on the first turn.")

    top_k = st.slider("Top K", min_value=1, max_value=10, value=5)

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("meta"):
                st.caption(message["meta"])

    user_query = st.chat_input("Ask a question, for example: How early should a leave request be submitted?")

    if user_query:
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        payload = {
            "query": user_query,
            "top_k": top_k,
            "conversation_id": st.session_state.conversation_id,
        }

        with st.chat_message("assistant"):
            with st.spinner("Retrieving evidence and generating the answer..."):
                try:
                    resp = requests.post(f"{API_BASE}/chat", json=payload, timeout=600)
                    if resp.ok:
                        data = resp.json()
                        st.session_state.conversation_id = data.get("conversation_id")

                        st.markdown(data["answer"])

                        status_lines = [
                            f"answer_type: `{data['answer_type']}`",
                            f"confidence: `{data['confidence']}`",
                            f"refused: `{data['refused']}`",
                        ]
                        if data.get("needs_clarification"):
                            status_lines.append("waiting for more user detail")
                        st.caption(" | ".join(status_lines))

                        if data.get("clarification_question"):
                            st.info(data["clarification_question"])

                        citations = data.get("citations", [])
                        if citations:
                            st.markdown("#### Citations")
                            for index, citation in enumerate(citations, start=1):
                                title = (
                                    f"{index}. {citation['source']} | "
                                    f"page {citation['page_start']}-{citation['page_end']}"
                                )
                                with st.expander(title):
                                    st.write(f"section: {citation.get('section_title')}")
                                    st.write(f"chunk_id: {citation['chunk_id']}")

                        trace = data.get("agent_trace", [])
                        if trace:
                            st.markdown("#### Agent Trace")
                            for step in trace:
                                st.write(f"- {step}")

                        meta = (
                            f"type={data['answer_type']}, confidence={data['confidence']}, "
                            f"clarification={data.get('needs_clarification', False)}"
                        )
                        st.session_state.messages.append(
                            {
                                "role": "assistant",
                                "content": data["answer"],
                                "meta": meta,
                            }
                        )
                    else:
                        st.error(f"Chat failed: {resp.status_code}")
                        st.json(resp.json())
                except Exception as exc:
                    st.error(f"Request failed: {exc}")


with tab_search:
    st.subheader("Search Debug")
    search_query = st.text_input(
        "Search query",
        placeholder="For example: What are the scholarship evaluation criteria?",
    )
    search_top_k = st.slider("Search Top K", min_value=1, max_value=10, value=5, key="search_top_k")

    if st.button("Run Search"):
        if not search_query.strip():
            st.warning("Please enter a search query first.")
        else:
            with st.spinner("Searching..."):
                payload = {"query": search_query, "top_k": search_top_k}
                try:
                    resp = requests.post(f"{API_BASE}/search", json=payload, timeout=600)
                    if resp.ok:
                        data = resp.json()
                        st.markdown("### Search Results")
                        for i, item in enumerate(data["results"], start=1):
                            title = (
                                f"Top {i} | score={item['score']:.4f} | "
                                f"{item['source']} | page {item['page_start']}-{item['page_end']}"
                            )
                            with st.expander(title):
                                st.write(f"chunk_id: {item['chunk_id']}")
                                st.write(f"section: {item.get('section_title')}")
                                st.write(item["text_preview"])
                        st.markdown("### Raw Response")
                        st.json(data)
                    else:
                        st.error(f"Search failed: {resp.status_code}")
                        st.json(resp.json())
                except Exception as exc:
                    st.error(f"Request failed: {exc}")

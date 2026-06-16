import streamlit as st
import os

from src.rag_engine import (
    ask_question,
    research_action
)

from src.process_pdf import process_pdf


# -----------------------
# Page Config
# -----------------------
st.set_page_config(
    page_title="ResearchGPT",
    page_icon="📚",
    layout="wide"
)

# -----------------------
# Session State
# -----------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "chunk_count" not in st.session_state:
    st.session_state.chunk_count = 0

# -----------------------
# Header
# -----------------------
st.title("📚 ResearchGPT")
st.caption(
    "AI-Powered Research Paper Assistant"
)

# -----------------------
# Sidebar
# -----------------------
with st.sidebar:

    st.header("📄 Research Paper")

    uploaded_file = st.file_uploader(
        "Upload PDF",
        type=["pdf"]
    )

    # -----------------------
    # Process PDF
    # -----------------------
    if uploaded_file is not None:

        os.makedirs(
            "uploads",
            exist_ok=True
        )

        save_path = os.path.join(
            "uploads",
            "current_paper.pdf"
        )

        with open(save_path, "wb") as f:

            f.write(
                uploaded_file.getbuffer()
            )

        if (
            "processed_file"
            not in st.session_state
            or
            st.session_state.processed_file
            != uploaded_file.name
        ):

            with st.spinner(
                "Processing PDF..."
            ):

                chunk_count = process_pdf(
                    save_path
                )

            st.session_state.processed_file = (
                uploaded_file.name
            )

            st.session_state.chunk_count = (
                chunk_count
            )

            st.success(
                "PDF processed successfully!"
            )

    st.divider()

    # -----------------------
    # Statistics
    # -----------------------
    st.subheader("📊 Statistics")

    st.metric(
        "Chunks",
        st.session_state.chunk_count
    )

    st.metric(
        "Vector Store",
        "Ready"
        if st.session_state.chunk_count > 0
        else "Not Ready"
    )

    st.divider()

    # -----------------------
    # Current Paper
    # -----------------------
    st.subheader("📁 Current Paper")

    if "processed_file" in st.session_state:

        st.write(
            f"**File:** {st.session_state.processed_file}"
        )

    else:

        st.write(
            "No paper uploaded"
        )

    st.divider()

    # -----------------------
    # Quick Actions
    # -----------------------
    st.subheader("⚡ Quick Actions")

    if st.button(
        "📝 Generate Summary"
    ):

        if (
            "processed_file"
            not in st.session_state
        ):

            st.warning(
                "Please upload a PDF first."
            )

        else:

            with st.spinner(
                "Generating Summary..."
            ):

                summary = research_action(
                    "Summarize this research paper."
                )

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": summary
                }
            )

            st.rerun()

    if st.button(
        "🔍 Key Findings"
    ):

        if (
            "processed_file"
            not in st.session_state
        ):

            st.warning(
                "Please upload a PDF first."
            )

        else:

            with st.spinner(
                "Extracting Findings..."
            ):

                findings = research_action(
                    "List the key findings of this paper in bullet points."
                )

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": findings
                }
            )

            st.rerun()

    if st.button(
        "⚠️ Limitations"
    ):

        if (
            "processed_file"
            not in st.session_state
        ):

            st.warning(
                "Please upload a PDF first."
            )

        else:

            with st.spinner(
                "Finding Limitations..."
            ):

                limitations = research_action(
                    "What limitations are mentioned in this paper?"
                )

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": limitations
                }
            )

            st.rerun()

    if st.button(
        "🚀 Future Work"
    ):

        if (
            "processed_file"
            not in st.session_state
        ):

            st.warning(
                "Please upload a PDF first."
            )

        else:

            with st.spinner(
                "Analyzing Future Work..."
            ):

                future = research_action(
                    "What future work is suggested in this paper?"
                )

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": future
                }
            )

            st.rerun()

    st.divider()

    # -----------------------
    # Clear Chat
    # -----------------------
    if st.button(
        "🧹 Clear Chat"
    ):

        st.session_state.messages = []

        st.rerun()

# -----------------------
# Welcome Screen
# -----------------------
if len(
    st.session_state.messages
) == 0:

    st.info(
        """
👋 Welcome to ResearchGPT

Upload a research paper and ask questions.

Example Questions:

• What is the Transformer?

• What are the main contributions?

• Explain Multi-Head Attention

• Summarize the paper

• What are the limitations?
"""
    )

# -----------------------
# Chat History
# -----------------------
for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):

        st.write(
            message["content"]
        )

# -----------------------
# Chat Input
# -----------------------
question = st.chat_input(
    "Ask something about the research paper..."
)

if question:

    if (
        "processed_file"
        not in st.session_state
    ):

        st.error(
            "Please upload a PDF first."
        )

        st.stop()

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )

    with st.chat_message(
        "user"
    ):

        st.write(question)

    with st.chat_message(
        "assistant"
    ):

        with st.spinner(
            "Analyzing paper..."
        ):

            answer = ask_question(
                question
            )

            st.write(answer)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer
        }
    )
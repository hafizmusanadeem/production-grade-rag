import time
import uuid

import requests
import streamlit as st
import logfire

from app.config import settings


# ============================================================
# LOGFIRE CONFIGURATION
# ============================================================
try:
    logfire_token = settings.LOGFIRE_TOKEN

    if not logfire_token:
        raise RuntimeError("LOGFIRE_TOKEN is missing or empty.")

    logfire.configure(token=logfire_token)
    logfire.instrument_requests()

    LOGFIRE_STATUS = "Connected & Tracing"

except Exception as e:
    print(f"Logfire initialization failed: {e}")
    LOGFIRE_STATUS = f"Standby (Error: {e})"


# ============================================================
# BACKEND CONFIGURATION
# ============================================================
BASE_URL = settings.BACKEND_URL

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Enterprise Agentic RAG",
    page_icon="🤖",
    layout="wide",
)


# ============================================================
# AVATARS
# ============================================================

AI_AVATAR = "🤖"
USER_AVATAR = "👤"


# ============================================================
# SESSION MANAGEMENT
# ============================================================

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

    logfire.info(
        f"✨ New User Session Created: "
        f"{st.session_state.session_id}"
    )

if "messages" not in st.session_state:
    st.session_state.messages = []


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.title("🧠 Agent OS")
    st.markdown("---")

    st.success(f"Logfire: {LOGFIRE_STATUS}")
    st.info(
        f"Memory ID: {st.session_state.session_id[:8]}"
    )

    st.caption(f"Backend: {BASE_URL}")

    st.markdown("---")

    if st.button(
        "🗑️ Clear History & Memory",
        width="stretch",
        type="primary"
    ):
        logfire.warning(
            "🗑️ Memory Wipe Triggered for session: "
            f"{st.session_state.session_id}"
        )

        st.session_state.messages = []
        st.session_state.session_id = str(uuid.uuid4())

        logfire.info(
            f"✨ New User Session Created: "
            f"{st.session_state.session_id}"
        )

        st.rerun()


# ============================================================
# MAIN CHAT
# ============================================================

st.title("🤖 Enterprise Agentic Assistant")


# ============================================================
# DISPLAY CHAT HISTORY
# ============================================================

for message in st.session_state.messages:

    avatar = (
        AI_AVATAR
        if message["role"] == "assistant"
        else USER_AVATAR
    )

    with st.chat_message(
        message["role"],
        avatar=avatar
    ):
        st.markdown(message["content"])


# ============================================================
# CHAT INPUT
# ============================================================

if prompt := st.chat_input(
    "Please feel free to ask questions about your provided documentation..."
):

    # ========================================================
    # USER INTERACTION TRACE
    # ========================================================

    with logfire.span(
        "💬 User Chat Interaction",
        user_query=prompt,
        session_id=st.session_state.session_id,
    ):

        # ----------------------------------------------------
        # Store and display user message
        # ----------------------------------------------------

        st.session_state.messages.append(
            {
                "role": "user",
                "content": prompt,
            }
        )

        with st.chat_message(
            "user",
            avatar=USER_AVATAR
        ):
            st.markdown(prompt)


        # ====================================================
        # ASSISTANT RESPONSE
        # ====================================================

        with st.chat_message(
            "assistant",
            avatar=AI_AVATAR
        ):

            data = {}

            # ------------------------------------------------
            # Agent Processing Status
            # ------------------------------------------------

            with st.status(
                "🔍 Agent is thinking...",
                expanded=True
            ) as status:

                try:

                    # ========================================
                    # CALL RAG BACKEND
                    # ========================================

                    with logfire.span(
                        "📡 Calling RAG Backend",
                        backend_url=BASE_URL,
                        session_id=st.session_state.session_id,
                    ):

                        url = f"{BASE_URL}/query"

                        payload = {
                            "q": prompt,
                            "thread_id": (
                                st.session_state.session_id
                            ),
                        }

                        response = requests.post(
                            url,
                            json=payload,
                            timeout=60,
                        )


                        # ====================================
                        # HTTP ERROR HANDLING
                        # ====================================

                        if response.status_code != 200:

                            logfire.error(
                                "❌ Backend returned HTTP error",
                                status_code=response.status_code,
                                response=response.text,
                            )

                            st.error(
                                f"Backend Error: "
                                f"{response.status_code}"
                            )

                            st.code(response.text)

                            status.update(
                                label="❌ Backend Error",
                                state="error",
                            )

                            st.stop()


                        # ====================================
                        # PARSE RESPONSE
                        # ====================================

                        try:
                            data = response.json()

                        except ValueError as e:

                            logfire.error(
                                f"❌ Invalid JSON from backend: {e}"
                            )

                            st.error(
                                "Backend returned an invalid "
                                "JSON response."
                            )

                            status.update(
                                label="❌ Invalid Backend Response",
                                state="error",
                            )

                            st.stop()


                    # ========================================
                    # DISPLAY AGENT THOUGHT PROCESS
                    # ========================================

                    steps = data.get(
                        "thought_process",
                        []
                    )

                    if steps:

                        st.markdown(
                            "**Agent Processing:**"
                        )

                        for step in steps:
                            st.markdown(
                                f"⚙️ {step}",
                                unsafe_allow_html=False,
                            )


                    # ========================================
                    # MARK PROCESSING COMPLETE
                    # ========================================

                    status.update(
                        label="✅ Answer Synthesized",
                        state="complete",
                        expanded=False,
                    )


                except requests.exceptions.Timeout:

                    logfire.error(
                        "❌ RAG Backend request timed out"
                    )

                    status.update(
                        label="❌ Backend Timeout",
                        state="error",
                    )

                    st.error(
                        "The backend took too long to respond. "
                        "Please try again."
                    )

                    st.stop()


                except requests.exceptions.ConnectionError as e:

                    logfire.error(
                        f"❌ Cannot connect to RAG Backend: {e}"
                    )

                    status.update(
                        label="❌ Backend Offline",
                        state="error",
                    )

                    st.error(
                        "Backend Offline. "
                        "Please make sure the FastAPI server "
                        "is running."
                    )

                    st.stop()


                except Exception as e:

                    logfire.error(
                        f"❌ UI-Backend Connection Failed: {e}"
                    )

                    status.update(
                        label="❌ Connection Failed",
                        state="error",
                    )

                    st.error(
                        f"An unexpected error occurred: {e}"
                    )

                    st.stop()


            # =================================================
            # FINAL ANSWER
            # =================================================

            answer_placeholder = st.empty()

            full_answer = data.get(
                "answer",
                "No response."
            )

            # Simulated streaming effect
            curr_text = ""

            for char in full_answer:

                curr_text += char

                answer_placeholder.markdown(
                    curr_text + "▌"
                )

                time.sleep(0.005)

            # Remove cursor
            answer_placeholder.markdown(
                full_answer
            )


            # =================================================
            # RETRIEVED SOURCES
            # =================================================

            sources = data.get(
                "sources",
                []
            )

            if sources:

                with st.expander(
                    f"📄 Retrieved Context "
                    f"({len(sources)} chunks)"
                ):

                    for i, source in enumerate(sources):

                        # Short preview
                        preview = (
                            source[:100]
                            .replace("\n", " ")
                        )

                        if len(source) > 100:
                            preview += "..."

                        # Each chunk gets its own expandable section
                        with st.expander(
                            f"Chunk {i + 1}: {preview}"
                        ):

                            st.info(source)

            else:

                st.caption(
                    "ℹ️ No context retrieved — "
                    "conversational response."
                )


            # =================================================
            # SAVE ASSISTANT MESSAGE
            # =================================================

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": full_answer,
                }
            )


            # =================================================
            # COMPLETION LOG
            # =================================================

            logfire.info(
                "✅ Chat cycle completed successfully.",
                session_id=st.session_state.session_id,
            )
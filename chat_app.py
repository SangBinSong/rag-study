"""
RAG Study Chat App - Simple Chat UI following Streamlit best practices
"""
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.messages import (
    AIMessage,
    HumanMessage
)

from dotenv import load_dotenv

from module.simple_rag_chat import SimpleRAGChat
from module.vector_db import VectorDB

load_dotenv()

# Page config
st.set_page_config(
    page_title="RAG Chat Demo",
    page_icon="💬",
    layout="wide"
)

def initialize_session():
    """Initialize session state variables"""
    if "session_input" not in st.session_state:
        st.session_state.session_input = ""

@st.cache_resource
def get_rag_instance():
    return SimpleRAGChat(
        llm=ChatOpenAI(
            model="gpt-4.1-mini",
            temperature=0.1
        ),
        vector_store=VectorDB(storage_path="./db/faiss"),
        summarizer_llm=ChatOpenAI(
            model="gpt-4.1-nano",
            temperature=0.1
        )
    )

def main():
    st.title("💬 RAG Chat Demo")
    st.caption("🚀 문서 기반 질의응답 시스템")
    
    initialize_session()

    simple_rag = get_rag_instance();
    # Display chat messages from history on app rerun
    for message in simple_rag.get_history_list() :
        role = ""
        if isinstance(message, HumanMessage):
            role = "user"
        elif isinstance(message, AIMessage):
            role = "assistant"
        else:
            # 지원하지 않는 타입은 건너뛰거나 에러 처리
            continue

        with st.chat_message(role):
            st.markdown(message.content)

    # React to user input
    prompt = st.chat_input("메시지를 입력하세요...", disabled=st.session_state.session_input != "")

    if prompt:
        st.session_state.session_input = prompt
        st.rerun()
    
    if st.session_state.session_input:
        with st.chat_message("user"):
            st.markdown(st.session_state.session_input)

        simple_rag.send(st.session_state.session_input)
        st.session_state.session_input = ""
        st.rerun()
        
    # Sidebar with info
    with st.sidebar:
        st.header("ℹ️ 정보")
        
        # Clear chat button
        if st.button("🗑️ 대화 초기화", type="secondary", use_container_width=True):
            simple_rag.new_history_session()
            st.rerun()
        
        st.markdown("**현재 상태:**")
        if simple_rag.get_history_length():
            st.success(f"💬 {simple_rag.get_history_length()}개 메시지")
        else:
            st.info("대화를 시작해보세요!")
        
        st.markdown("---")
        
        st.markdown("**사용 가능한 질문 예시:**")
        example_questions = [
            "AI 도입 전략은 무엇인가요?",
            "공공부문의 디지털 전환에 대해 설명해주세요",
            "인공지능 정책의 주요 방향은?",
            "디지털 정부 혁신 방안은?"
        ]
        
        for question in example_questions:
            if st.button(f"💡 {question[:20]}...", key=f"example_{hash(question)}", use_container_width=True):
                # st.input에 입력된 내용을 초기화
                st.session_state.session_input = question
                st.rerun()
        
        st.markdown("---")
        
        st.markdown("**💡 팁:**")
        st.markdown(
            """
            - 구체적인 질문을 하면 더 정확한 답변을 받을 수 있습니다
            - 한 번에 하나의 주제에 대해 물어보세요
            - 현재는 테스트 모드로 입력한 내용을 그대로 답변으로 반환합니다
            """
        )

if __name__ == "__main__":
    main()
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
            model="gpt-4o-mini",
            temperature=0.1,
            streaming=True,
            max_tokens=1000,
            request_timeout=30
        ),
        vector_store=VectorDB(storage_path="./db/faiss"),
        summarizer_llm=ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0.1,
            max_tokens=200,
            request_timeout=15
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
        # st.rerun() 제거 - 스트리밍 처리에서 직접 처리
    
    if st.session_state.session_input:
        with st.chat_message("user"):
            st.markdown(st.session_state.session_input)

        # 스트리밍 응답 처리
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            
            # 스트리밍 응답 생성
            try:
                for chunk in simple_rag.send_stream(st.session_state.session_input):
                    if chunk:  # 빈 청크가 아닌 경우만 처리
                        full_response += chunk
                        message_placeholder.markdown(full_response + "▌")

                # 최종 응답 표시 (커서 제거)
                message_placeholder.markdown(full_response)
                
            except Exception as e:
                message_placeholder.error(f"스트리밍 중 오류가 발생했습니다: {str(e)}")
        
        # 입력 초기화 후 페이지 새로고침
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
            - 응답이 실시간으로 스트리밍됩니다
            - 문서 기반 RAG 시스템으로 정확한 답변을 제공합니다
            """
        )

if __name__ == "__main__":
    main()
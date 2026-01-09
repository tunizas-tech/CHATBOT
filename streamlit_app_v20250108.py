import os
import hashlib
from pathlib import Path
from typing import Optional, List

import streamlit as st

from langchain.text_splitter import RecursiveCharacterTextSplitter
# 커뮤니티 경로의 FAISS 사용을 권장합니다.
from langchain_community.vectorstores import FAISS
from langchain.schema import Document

# OpenAI LLM/Embedding
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

# Agent 관련
from langchain.tools.retriever import create_retriever_tool
from langchain.prompts import ChatPromptTemplate
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain.agents import Tool

# 검색(옵션)
from langchain_community.utilities import SerpAPIWrapper

# ========= 고정 경로/옵션 =========
PDF_DIR = "./pdf"                 # <- 여기만 바꾸면 됩니다
FAISS_DIR = "./faiss_index"       # 인덱스 저장 폴더
EMBED_MODEL = "text-embedding-3-small"  # 경량 임베딩 모델 권장
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150

# 환경 설정
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


# ========== SerpAPI 웹검색 툴(키가 없으면 추가 안 함) ==========
def search_web_tool_or_none():
    api_key = os.environ.get("SERPAPI_API_KEY") or st.session_state.get("SERPAPI_API")
    if not api_key:
        return None

    search = SerpAPIWrapper()  # 내부에서 키를 환경변수로 읽음

    def run_with_source(query: str) -> str:
        try:
            results = search.results(query)
            organic = results.get("organic_results", [])
            formatted = []
            for r in organic[:5]:
                title = r.get("title")
                link = r.get("link")
                source = r.get("source")
                snippet = r.get("snippet")
                if link:
                    formatted.append(f"- [{title}]({link}) ({source})\n  {snippet}")
                else:
                    formatted.append(f"- {title} (출처: {source})\n  {snippet}")
            return "\n".join(formatted) if formatted else "검색 결과가 없습니다."
        except Exception as e:
            return f"검색 중 오류가 발생했습니다: {e}"

    return Tool(
        name="web_search",
        func=run_with_source,
        description="실시간 뉴스 및 웹 정보를 검색할 때 사용합니다. 결과는 제목+출처+링크+간단요약(snippet) 형태로 반환됩니다."
    )


# ========== PDF 로더: OCR Fallback 포함(선택) ==========
def load_with_ocr_fallback(pdf_path: Path) -> List[Document]:
    """PyPDFLoader 실패 시 Unstructured OCR로 대체 시도"""
    try:
        from langchain.document_loaders import PyPDFLoader
        return PyPDFLoader(str(pdf_path)).load()
    except Exception:
        # OCR fallback (선택): 필요 시 의존성 설치
        try:
            # pip install "unstructured[local-inference]"
            from langchain_community.document_loaders import UnstructuredPDFLoader
            loader = UnstructuredPDFLoader(str(pdf_path), mode="single")
            return loader.load()
        except Exception as e:
            st.error(f"OCR fallback 실패: {pdf_path.name} - {e}")
            return []


# ========== 폴더 변경 감지(파일 경로+수정시각으로 해시) ==========
def _folder_signature(pdf_dir: Path) -> str:
    files = sorted(pdf_dir.glob("**/*.pdf"))
    h = hashlib.md5()
    for f in files:
        try:
            stat = f.stat()
            h.update(str(f.resolve()).encode())
            h.update(str(int(stat.st_mtime)).encode())
        except Exception:
            # 접근 불가/일시 오류는 스킵
            continue
    return h.hexdigest()


# ========== 캐시: 인덱스 빌드 또는 로딩 ==========
@st.cache_resource(show_spinner=True)
def build_or_load_faiss_index(pdf_dir_str: str) -> Optional[FAISS]:
    pdf_dir = Path(pdf_dir_str)
    if not pdf_dir.exists() or not pdf_dir.is_dir():
        st.warning(f"지정한 폴더가 존재하지 않습니다: {pdf_dir.resolve()}")
        return None

    sig = _folder_signature(pdf_dir)
    faiss_sig_file = Path(FAISS_DIR) / "signature.txt"

    # 저장된 인덱스가 있고, 시그니처가 동일하면 로드
    if Path(FAISS_DIR).exists() and faiss_sig_file.exists():
        old_sig = faiss_sig_file.read_text(encoding="utf-8").strip()
        if old_sig == sig:
            try:
                vs = FAISS.load_local(
                    FAISS_DIR,
                    OpenAIEmbeddings(model=EMBED_MODEL),
                    allow_dangerous_deserialization=True
                )
                return vs
            except Exception:
                # 인덱스가 깨졌거나 버전 충돌 등: 재빌드로 폴백
                pass

    # 재빌드
    files = list(pdf_dir.glob("**/*.pdf"))
    if not files:
        st.warning(f"PDF 파일을 찾을 수 없습니다: {pdf_dir.resolve()}")
        return None

    all_docs: List[Document] = []
    for pdf in files:
        docs = load_with_ocr_fallback(pdf)
        if not docs:
            st.error(f"PDF 로딩 실패: {pdf.name}")
            continue
        # 출처를 metadata에 기록
        for d in docs:
            d.metadata = d.metadata or {}
            d.metadata["source"] = pdf.name
            d.metadata["path"] = str(pdf.resolve())
        all_docs.extend(docs)

    if not all_docs:
        st.warning("문서 로딩에 실패했습니다.")
        return None

    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = splitter.split_documents(all_docs)

    vs = FAISS.from_documents(chunks, OpenAIEmbeddings(model=EMBED_MODEL))

    Path(FAISS_DIR).mkdir(parents=True, exist_ok=True)
    vs.save_local(FAISS_DIR)
    faiss_sig_file.write_text(sig, encoding="utf-8")
    return vs


# ========== retriever tool 생성 ==========
def make_pdf_search_tool():
    vs = build_or_load_faiss_index(PDF_DIR)
    if not vs:
        return None
    retriever = vs.as_retriever(search_kwargs={"k": 5})
    return create_retriever_tool(
        retriever,
        name="pdf_search",
        description="Use this tool to search information from the pdf document"
    )


# ========== Agent 대화 실행 ==========
def chat_with_agent(user_input, agent_executor):
    result = agent_executor({"input": user_input})
    return result["output"]


# ========== Streamlit App ==========
def main():
    st.set_page_config(page_title="AI 비서", layout="wide", page_icon="🤖")

    with st.container():
        if Path("./chatbot_logo.png").exists():
            st.image("./chatbot_logo.png", use_container_width=True)
        st.markdown("---")
        st.title("안녕하세요! RAG를 활용한 'AI 비서 톡톡이' 입니다")

    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    # 사이드바: 키만 입력(폴더/파일 입력 UI는 없음)
    with st.sidebar:
        st.session_state["OPENAI_API"] = st.text_input("OPENAI API 키", placeholder="Enter Your API Key", type="password")
        st.session_state["SERPAPI_API"] = st.text_input("SERPAPI_API 키 (선택)", placeholder="Enter Your API Key", type="password")
        if st.session_state.get("OPENAI_API"):
            os.environ["OPENAI_API_KEY"] = st.session_state["OPENAI_API"]
        if st.session_state.get("SERPAPI_API"):
            os.environ["SERPAPI_API_KEY"] = st.session_state["SERPAPI_API"]

        st.caption(f"📁 PDF 경로: `{Path(PDF_DIR).resolve()}`")
        st.caption(f"💾 FAISS 저장 경로: `{Path(FAISS_DIR).resolve()}`")
        st.caption(f"🧠 임베딩 모델: `{EMBED_MODEL}` / 청크 {CHUNK_SIZE} (+{CHUNK_OVERLAP})")

    # 키 확인
    if not st.session_state.get("OPENAI_API"):
        st.warning("OpenAI API 키를 입력하세요.")
        return

    # 도구 구성
    tools = []

    pdf_search_tool = make_pdf_search_tool()
    if pdf_search_tool:
        tools.append(pdf_search_tool)

    web_tool = search_web_tool_or_none()
    if web_tool:
        tools.append(web_tool)

    # LLM
    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)

    # 에이전트 프롬프트
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system",
             "Be sure to answer in Korean. You are a helpful assistant. "
             "Make sure to use the `pdf_search` tool for searching information from the pdf document. "
             "If you can't find the information from the PDF document, use the `web_search` tool for searching information from the web. "
             "If the user’s question contains words like '최신', '현재', or '오늘', you must ALWAYS use the `web_search` tool to ensure real-time information is retrieved. "             "IMPORTANT: When you use the `pdf_search` tool to find information, you MUST always cite the source PDF file name in your response. "
             "Include the source information in a format like '(출처: [파일명])' at the end of the relevant sentence or paragraph. "             "Please always include emojis in your responses with a friendly tone. "
             "Your name is `AI 비서 톡톡이`. Please introduce yourself at the beginning of the conversation."
             ),
            ("placeholder", "{chat_history}"),
            ("human", "{input} \n\n Be sure to include emoji in your responses."),
            ("placeholder", "{agent_scratchpad}"),
        ]
    )

    agent = create_tool_calling_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

    # 입력창
    user_input = st.chat_input("질문이 무엇인가요?")
    if user_input:
        # 단순 메시지 스택 유지(초기 로직 유지)
        st.session_state["messages"].append({"role": "user", "content": user_input})
        prev_msgs = [{"role": m["role"], "content": m["content"]} for m in st.session_state["messages"][:-1]]
        response = chat_with_agent(user_input + "\n\nPrevious Messages: " + str(prev_msgs), agent_executor)
        st.session_state["messages"].append({"role": "assistant", "content": response})

    # 대화 출력
    for msg in st.session_state["messages"]:
        st.chat_message(msg["role"]).write(msg["content"])


if __name__ == "__main__":
    main()

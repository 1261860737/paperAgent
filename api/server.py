from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from api.session_manager import SessionManager
import os
import shutil

app = FastAPI()
session_manager = SessionManager()


# 定义允许的源（即前端应用所在的地址）
origins = [
    "http://localhost",  # 允许 localhost 访问
    "http://localhost:5173",  # 如果你的前端是 Vite 的默认端口
    "http://paperagent.com",  # 如果部署了前端到其他地方，请在此添加
]

# 添加 CORS 中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # 允许的来源
    allow_credentials=True,
    allow_methods=["*"],  # 允许所有请求方法（GET, POST, PUT, DELETE 等）
    allow_headers=["*"],  # 允许所有请求头
)

UPLOAD_DIR = "./uploaded_pdfs"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.post("/api/upload_pdf")
async def upload_pdf(file: UploadFile = File(...)):
    """上传 PDF 并创建 Session"""
    save_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    session_id = session_manager.create_session(save_path)
    wf = session_manager.get_workflow(session_id)

    collection_info = wf.retriever.vector_db.get_collection_info()
    chunks_count = collection_info.get("row_count", 0)

    return {
        "session_id": session_id,
        "chunks": chunks_count,
        "status": "ready"
    }


@app.post("/api/ask")
async def ask(session_id: str, query: str, top_k: int = 3):
    """向某个 Session 提问"""
    workflow = session_manager.get_workflow(session_id)


    # 确保将 top_k 参数传递给 run_workflow
    result = await workflow.run_workflow(query, top_k=top_k)

    # 存历史，用于前端显示
    session_manager.append_history(session_id, "user", query)
    session_manager.append_history(session_id, "assistant", result["answer"])

    return result


@app.get("/api/history/{session_id}")
async def history(session_id: str):
    return session_manager.get_history(session_id)

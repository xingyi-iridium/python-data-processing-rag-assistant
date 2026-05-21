#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI智能客服 API 服务器 v4.0（双模型RAG优化版）
符合竞赛规范的 REST API 接口（支持中英文双语 + 多问题完整回答 + 拟人化客服风格）

============================================
【设计理念 - 双模型协同架构】
============================================
模型分工：
1. Qwen/Qwen3-VL-8B-Instruct：负责视觉理解
   - 解析用户上传的图片
   - 生成详细的图片描述文本

2. Pro/deepseek-ai/DeepSeek-V3.2：负责文本理解和答案生成
   - 接收纯文本问题 → 直接生成答案
   - 接收图片描述 + RAG知识 → 生成综合答案

优势：
1. 专业化分工：视觉模型专注图像理解，文本模型专注语言生成
2. 成本优化：使用性价比更高的DeepSeek-V3.2处理主要任务
3. 性能提升：DeepSeek-V3.2在中文理解和客服对话方面表现优异
4. 保持兼容：完全兼容原有API接口和数据结构
"""
import os
import sys
import logging
import warnings
import base64
import uuid
import json
import re
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Header, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager
import uvicorn

# 忽略所有警告信息
warnings.filterwarnings("ignore")

# 加载环境变量
load_dotenv()


# ==============================================
# 【配置管理】
# ==============================================
class Config:
    """全局配置类"""

    # --- API 服务配置 ---
    API_HOST = "0.0.0.0"
    API_PORT = 8000
    KAFU_API_TOKEN = os.getenv("KAFU_API_TOKEN", "your-secret-token-here")

    # --- SiliconFlow API 配置 ---
    SILICON_API_KEY = os.getenv("SILICON_API_KEY")
    BASE_URL = "https://api.siliconflow.cn/v1"

    # --- 模型配置（双模型架构）---
    # 视觉模型：专门用于图片理解
    VISION_MODEL = "Qwen/Qwen3-VL-32B-Instruct"

    # 文本模型：用于纯文本问答和图片+文本综合问答
    TEXT_MODEL = "Pro/deepseek-ai/DeepSeek-V3.2"

    # 文档精炼模型（可选，如需实时精炼）
    REFINEMENT_MODEL = "Pro/deepseek-ai/DeepSeek-V3.2"

    # --- 超时控制 ---
    TEXT_REQUEST_TIMEOUT = 20  # DeepSeek-V3.2响应较快
    VISION_REQUEST_TIMEOUT = 25  # 视觉模型需要更长时间
    REFINEMENT_REQUEST_TIMEOUT = 30

    # --- RAG 知识库配置 ---
    KNOWLEDGE_BASE_DIR = "knowledge_base"
    VECTOR_DB_PATH = "vector_db_agent3"
    # 新增：精炼文档向量库路径
    REFINED_VECTOR_DB_PATH = "refined_vector_db"
    IMAGES_DIR = "extracted_images"
    CHUNK_SIZE = 256
    CHUNK_OVERLAP = 30
    MAX_IMAGE_SIZE_MB = 5
    # 优化：降低top_k到2，提高检索速度
    RAG_TOP_K = 2

    # --- 模型参数 ---
    TEXT_TEMPERATURE = 0.3
    VISION_TEMPERATURE = 0.1
    REFINEMENT_TEMPERATURE = 0.2
    MAX_RETRIES = 1


config = Config()

# ==============================================
# 【日志系统】
# ==============================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("api_server.log", encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

# ==============================================
# 【核心依赖导入】
# ==============================================
try:
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_community.vectorstores import FAISS
    from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
    import fitz  # PyMuPDF
    import docx  # python-docx
except ImportError as e:
    logger.error(f"缺少依赖包：{e}")
    logger.info(
        "请安装：pip install langchain-openai langchain-text-splitters langchain-community pymupdf python-docx fastapi uvicorn python-dotenv")
    sys.exit(1)

# ==============================================
# 【全局变量】
# ==============================================
text_agent = None
vision_agent = None
refinement_agent = None
vector_store = None
refined_vector_store = None
image_metadata = {}
session_histories = {}


# ==============================================
# 【RAG知识库系统】
# ==============================================
def extract_text_from_word(docx_path: str) -> str:
    """从Word文档中提取纯文本"""
    try:
        doc = docx.Document(docx_path)
        full_text = [para.text for para in doc.paragraphs]
        text = '\n'.join(full_text)
        logger.debug(f"Word文档提取完成：{len(text)}字符")
        return text
    except Exception as e:
        logger.error(f"Word解析失败：{str(e)}")
        raise


def extract_text_from_txt(txt_path: str) -> str:
    """从TXT文件中提取文本"""
    try:
        for encoding in ['utf-8', 'gbk', 'gb2312', 'latin-1']:
            try:
                with open(txt_path, 'r', encoding=encoding) as f:
                    text = f.read()
                logger.debug(f"TXT文件提取完成（{encoding}）：{len(text)}字符")
                return text
            except UnicodeDecodeError:
                continue
        raise ValueError(f"无法解析文件编码：{txt_path}")
    except Exception as e:
        logger.error(f"TXT解析失败：{str(e)}")
        raise


def extract_text_and_images_from_pdf(pdf_path: str) -> Tuple[List[Dict], List[Dict]]:
    """从PDF中提取文本和图片

    Returns:
        (文本块列表 [{"content", "page", "source"}, ...],
         图片信息列表 [{"path", "page", "source"}, ...])
    """
    try:
        doc = fitz.open(pdf_path)
        text_chunks = []
        image_info = []
        source_name = os.path.basename(pdf_path)

        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text()

            if text.strip():
                text_chunks.append({
                    "content": text,
                    "page": page_num + 1,
                    "source": source_name
                })

            # 提取图片
            image_list = page.get_images(full=True)
            for img_index, img in enumerate(image_list):
                xref = img[0]
                try:
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    image_ext = base_image["ext"]

                    img_filename = f"page{page_num + 1}_img{img_index}.png"
                    img_path = os.path.join(config.KNOWLEDGE_BASE_DIR, config.IMAGES_DIR, img_filename)
                    os.makedirs(os.path.dirname(img_path), exist_ok=True)

                    with open(img_path, "wb") as img_file:
                        img_file.write(image_bytes)

                    image_info.append({
                        "path": img_path,
                        "page": page_num + 1,
                        "source": source_name
                    })
                except Exception as img_err:
                    logger.warning(f"PDF图片提取失败（页{page_num + 1}，图{img_index + 1}）：{img_err}")

        doc.close()
        logger.info(f"PDF提取完成：{len(text_chunks)}个文本块，{len(image_info)}张图片")
        return text_chunks, image_info

    except Exception as e:
        logger.error(f"PDF解析失败：{str(e)}")
        raise


def load_refined_knowledge_from_jsonl(jsonl_path: str) -> List[Dict]:
    """
    从JSONL文件加载精炼知识库

    Args:
        jsonl_path: JSONL文件路径

    Returns:
        精炼文档块列表 [{"content", "page", "source", "category", "pic_ids"}, ...]
    """
    refined_chunks = []

    if not os.path.exists(jsonl_path):
        logger.warning(f"JSONL文件不存在：{jsonl_path}")
        return refined_chunks

    try:
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue

                try:
                    record = json.loads(line)

                    # 提取字段
                    question_keyword = record.get("question_keyword", "")
                    answer = record.get("answer", "")
                    source = record.get("source", "未知")

                    # 合并为完整内容
                    content = f"{question_keyword}\n{answer}"

                    # 尝试从答案中提取页码（如果有 [第X页] 标记）
                    page_match = re.search(r'\[第(\d+)页\]', answer)
                    page = int(page_match.group(1)) if page_match else 1

                    # 提取图片ID（ManualXX_YY格式）
                    pic_ids = re.findall(r'<PIC>(Manual\d+_\d+)</PIC>', answer)

                    refined_chunks.append({
                        "content": content,
                        "page": page,
                        "source": source,
                        "category": "refined",
                        "pic_ids": pic_ids
                    })

                except json.JSONDecodeError as e:
                    logger.warning(f"JSONL第{line_num}行解析失败：{e}")
                    continue

        logger.info(f"✅ 从JSONL加载 {len(refined_chunks)} 条精炼知识")
        return refined_chunks

    except Exception as e:
        logger.error(f"JSONL加载失败：{e}")
        return refined_chunks


def build_refined_vector_database(refined_chunks: List[Dict]) -> Any:
    """
    构建精炼文档的FAISS向量数据库

    参数：
        refined_chunks: 精炼后的文档块 [{"content", "page", "source", "category", "pic_ids"}, ...]

    返回：
        FAISS向量数据库实例
    """
    try:
        embeddings = OpenAIEmbeddings(
            api_key=config.SILICON_API_KEY,
            base_url=config.BASE_URL,
            model="BAAI/bge-large-zh-v1.5"
        )

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP
        )

        all_texts = []
        all_metadatas = []

        for chunk in refined_chunks:
            sub_chunks = text_splitter.split_text(chunk["content"])
            for sub_chunk in sub_chunks:
                all_texts.append(sub_chunk)
                all_metadatas.append({
                    "page": chunk["page"],
                    "source": chunk["source"],
                    "category": chunk.get("category", "general"),
                    "pic_ids": json.dumps(chunk.get("pic_ids", []))  # 存储图片ID列表
                })

        logger.info(f"开始构建精炼向量数据库，共 {len(all_texts)} 个文本块")

        batch_size = 32
        vector_db = None

        for i in range(0, len(all_texts), batch_size):
            batch_texts = all_texts[i:i + batch_size]
            batch_metadatas = all_metadatas[i:i + batch_size]

            if vector_db is None:
                vector_db = FAISS.from_texts(
                    texts=batch_texts,
                    embedding=embeddings,
                    metadatas=batch_metadatas
                )
            else:
                batch_db = FAISS.from_texts(
                    texts=batch_texts,
                    embedding=embeddings,
                    metadatas=batch_metadatas
                )
                vector_db.merge_from(batch_db)

            logger.info(f"已处理 {min(i + batch_size, len(all_texts))}/{len(all_texts)} 个精炼文本块")

        os.makedirs(config.REFINED_VECTOR_DB_PATH, exist_ok=True)
        vector_db.save_local(config.REFINED_VECTOR_DB_PATH)

        logger.info(f"✅ 精炼向量数据库构建完成：{len(all_texts)}个精炼文档片段")
        return vector_db
    except Exception as e:
        logger.error(f"精炼向量数据库构建失败：{str(e)}")
        raise


def build_vector_database(text_chunks: List[Dict]) -> Any:
    """构建原始文档的 FAISS 向量数据库（备用）"""
    try:
        embeddings = OpenAIEmbeddings(
            api_key=config.SILICON_API_KEY,
            base_url=config.BASE_URL,
            model="BAAI/bge-large-zh-v1.5"
        )

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP
        )

        all_texts = []
        all_metadatas = []

        for chunk in text_chunks:
            sub_chunks = text_splitter.split_text(chunk["content"])
            for sub_chunk in sub_chunks:
                all_texts.append(sub_chunk)
                all_metadatas.append({
                    "page": chunk["page"],
                    "source": chunk["source"]
                })

        logger.info(f"开始构建原始向量数据库，共 {len(all_texts)} 个文本块")

        batch_size = 32
        vector_db = None

        for i in range(0, len(all_texts), batch_size):
            batch_texts = all_texts[i:i + batch_size]
            batch_metadatas = all_metadatas[i:i + batch_size]

            if vector_db is None:
                vector_db = FAISS.from_texts(
                    texts=batch_texts,
                    embedding=embeddings,
                    metadatas=batch_metadatas
                )
            else:
                batch_db = FAISS.from_texts(
                    texts=batch_texts,
                    embedding=embeddings,
                    metadatas=batch_metadatas
                )
                vector_db.merge_from(batch_db)

            logger.info(f"已处理 {min(i + batch_size, len(all_texts))}/{len(all_texts)} 个原始文本块")

        os.makedirs(config.VECTOR_DB_PATH, exist_ok=True)
        vector_db.save_local(config.VECTOR_DB_PATH)

        logger.info(f"✅ 原始向量数据库构建完成：{len(all_texts)}个文档片段")
        return vector_db
    except Exception as e:
        logger.error(f"原始向量数据库构建失败：{str(e)}")
        raise


def load_knowledge_base(pdf_folder: str = "manuals") -> bool:
    """
    加载知识库（三级优先级策略）

    优先级：
    1. 已存在的精炼向量库（最快）
    2. 预处理的JSONL文件（快速）
    3. 实时精炼原始文档（较慢，仅首次或无预处理时）
    """
    global vector_store, refined_vector_store, image_metadata

    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))

        # 优先级1：检查是否存在精炼向量库
        refined_db_path = os.path.join(script_dir, config.REFINED_VECTOR_DB_PATH)
        if os.path.exists(refined_db_path):
            logger.info("✅ 检测到精炼向量库，直接加载...")
            try:
                embeddings = OpenAIEmbeddings(
                    api_key=config.SILICON_API_KEY,
                    base_url=config.BASE_URL,
                    model="BAAI/bge-large-zh-v1.5"
                )
                refined_vector_store = FAISS.load_local(
                    refined_db_path,
                    embeddings,
                    allow_dangerous_deserialization=True
                )
                logger.info(f"✅ 精炼向量库加载成功")
                return True
            except Exception as e:
                logger.warning(f"精炼向量库加载失败：{e}，尝试其他方案")

        # 优先级2：检查是否存在预处理的JSONL文件
        jsonl_path = os.path.join(script_dir, "processed_knowledge.jsonl")
        if os.path.exists(jsonl_path):
            logger.info("✅ 检测到预处理JSONL文件，加载并构建向量库...")
            try:
                refined_chunks = load_refined_knowledge_from_jsonl(jsonl_path)
                if refined_chunks:
                    refined_vector_store = build_refined_vector_database(refined_chunks)
                    logger.info(f"✅ 从JSONL构建精炼向量库成功")
                    return True
            except Exception as e:
                logger.warning(f"JSONL加载失败：{e}，回退到原始文档")

        # 优先级3：从原始文档实时精炼（兼容模式）
        logger.info("⚠️ 未找到预处理文件，从原始文档加载...")

        all_text_chunks = []
        all_image_info = []

        if not os.path.exists(pdf_folder):
            logger.error(f"知识库目录不存在：{pdf_folder}")
            return False

        for filename in os.listdir(pdf_folder):
            file_path = os.path.join(pdf_folder, filename)
            ext = os.path.splitext(filename)[1].lower()

            try:
                if ext == '.pdf':
                    chunks, images = extract_text_and_images_from_pdf(file_path)
                    all_text_chunks.extend(chunks)
                    all_image_info.extend(images)
                elif ext == '.docx':
                    text = extract_text_from_word(file_path)
                    if text.strip():
                        all_text_chunks.append({
                            "content": text,
                            "page": 1,
                            "source": filename
                        })
                elif ext == '.txt':
                    text = extract_text_from_txt(file_path)
                    if text.strip():
                        all_text_chunks.append({
                            "content": text,
                            "page": 1,
                            "source": filename
                        })
            except Exception as e:
                logger.error(f"处理文件失败 {filename}: {str(e)}")
                continue

        if not all_text_chunks:
            return False

        # 保存原始向量库（备用）
        vector_store = build_vector_database(all_text_chunks)
        for img in all_image_info:
            key = f"{img['source']}_page{img['page']}"
            if key not in image_metadata:
                image_metadata[key] = []
            image_metadata[key].append(img["path"])

        logger.warning("⚠️ 建议使用 refine_knowledge_base.py 预先生成精炼知识库以提高性能")
        return True

    except Exception as e:
        logger.error(f"知识库加载失败：{str(e)}")
        return False


def retrieve_relevant_content(query: str, top_k: int = None) -> Tuple[str, List[str]]:
    """
    检索相关知识（优先使用精炼向量库）

    策略：
    1. 如果有精炼向量库，优先检索精炼文档
    2. 否则回退到原始向量库

    Returns:
        (相关文本, 相关图片ID列表)
    """
    global refined_vector_store, vector_store

    if top_k is None:
        top_k = config.RAG_TOP_K

    # 优先使用精炼向量库
    active_store = refined_vector_store if refined_vector_store else vector_store

    if not active_store:
        return "", []

    try:
        docs = active_store.similarity_search(query, k=top_k)

        relevant_texts = []
        related_image_ids = []

        for doc in docs:
            page = doc.metadata.get("page", "未知")
            source = doc.metadata.get("source", "未知")
            content = doc.page_content

            # 添加页码和来源信息
            relevant_texts.append(f"【来源：{source} 第{page}页】\n{content}")

            # 从metadata提取图片ID
            pic_ids_json = doc.metadata.get("pic_ids", "[]")
            try:
                pic_ids = json.loads(pic_ids_json)
                related_image_ids.extend(pic_ids)
            except:
                pass

            # 也从content中提取<PIC>标签
            pic_tags = re.findall(r'<PIC>([^<]+)</PIC>', content)
            related_image_ids.extend(pic_tags)

        # 去重并保持顺序
        related_image_ids = list(dict.fromkeys(related_image_ids))

        combined_text = "\n\n---\n\n".join(relevant_texts)
        return combined_text, related_image_ids

    except Exception as e:
        logger.error(f"检索失败：{str(e)}")
        return "", []


# ==============================================
# 【Agent初始化】
# ==============================================
def init_agents():
    """初始化AI模型实例（双模型架构）"""
    global text_agent, vision_agent, refinement_agent

    # 文本生成模型：DeepSeek-V3.2（处理纯文本和图片+文本综合问答）
    text_agent = ChatOpenAI(
        api_key=config.SILICON_API_KEY,
        base_url=config.BASE_URL,
        model=config.TEXT_MODEL,
        temperature=config.TEXT_TEMPERATURE,
        request_timeout=config.TEXT_REQUEST_TIMEOUT,
        max_retries=config.MAX_RETRIES
    )

    # 视觉理解模型：Qwen3-VL-8B（专门处理图片）
    vision_agent = ChatOpenAI(
        api_key=config.SILICON_API_KEY,
        base_url=config.BASE_URL,
        model=config.VISION_MODEL,
        temperature=config.VISION_TEMPERATURE,
        request_timeout=config.VISION_REQUEST_TIMEOUT,
        max_retries=config.MAX_RETRIES
    )

    # 文档精炼模型：DeepSeek-V3.2（如需实时精炼）
    refinement_agent = ChatOpenAI(
        api_key=config.SILICON_API_KEY,
        base_url=config.BASE_URL,
        model=config.REFINEMENT_MODEL,
        temperature=config.REFINEMENT_TEMPERATURE,
        request_timeout=config.REFINEMENT_REQUEST_TIMEOUT,
        max_retries=config.MAX_RETRIES
    )

    logger.info("✅ Agents 初始化完成")
    logger.info(f"   📝 文本模型: {config.TEXT_MODEL}")
    logger.info(f"   👁️  视觉模型: {config.VISION_MODEL}")


# ==============================================
# 【FastAPI应用】
# ==============================================
security = HTTPBearer(auto_error=False)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    logger.info("🚀 启动 AI 智能客服 API 服务 v4.0（双模型RAG优化版）...")

    init_agents()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    manuals_path = os.path.join(script_dir, "manuals")
    kb_loaded = load_knowledge_base(manuals_path)

    if kb_loaded:
        if refined_vector_store:
            logger.info("✅ 精炼知识库加载成功（快速模式）")
        else:
            logger.info("✅ 原始知识库加载成功（兼容模式）")
    else:
        logger.warning("⚠️ 知识库未加载，请检查 manuals 目录")

    yield

    logger.info("🛑 关闭 API 服务...")


app = FastAPI(
    title="AI智能客服API v4.0",
    description="符合竞赛规范的REST API接口（双模型RAG优化版）",
    version="4.0.0",
    lifespan=lifespan
)


# ==============================================
# 【认证中间件】
# ==============================================
async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """验证 Bearer Token"""
    if credentials is None:
        raise HTTPException(status_code=401, detail="缺少认证令牌")

    if credentials.credentials != config.KAFU_API_TOKEN:
        raise HTTPException(status_code=403, detail="无效的认证令牌")

    return credentials.credentials


# ==============================================
# 【数据模型】
# ==============================================
class ChatRequest(BaseModel):
    """聊天请求数据模型"""
    question: Optional[str] = Field(None, description="用户问题")
    session_id: Optional[str] = Field(None, description="会话ID")
    image: Optional[str] = Field(None, description="图片Base64编码")


class ChatResponse(BaseModel):
    """聊天响应数据模型"""
    code: int = Field(0, description="状态码")
    data: Dict[str, Any] = Field(..., description="响应数据")


# ==============================================
# 【核心业务逻辑】
# ==============================================
def detect_question_type(question: str) -> str:
    """
    检测问题类型

    返回：
    - technical: 技术说明类（指示灯、按钮含义等）
    - accessory: 配件咨询类（表带、电池等）
    - logistics: 物流快递类
    - after_sales: 售后服务类
    - general: 通用咨询类
    """
    question_lower = question.lower()

    # 技术说明类关键词
    technical_keywords = ["指示灯", "闪烁", "标识", "含义", "代表", "按钮", "图标", "符号"]
    # 配件咨询类关键词
    accessory_keywords = ["表带", "电池", "配件", "更换", "尺寸", "可选", "其他"]
    # 物流类关键词
    logistics_keywords = ["物流", "快递", "运费", "配送", "送到", "乡镇", "多久", "揽收"]
    # 售后类关键词
    after_sales_keywords = ["售后", "维修", "退款", "退货", "换货", "故障", "质保"]

    if any(kw in question_lower for kw in technical_keywords):
        return "technical"
    elif any(kw in question_lower for kw in accessory_keywords):
        return "accessory"
    elif any(kw in question_lower for kw in logistics_keywords):
        return "logistics"
    elif any(kw in question_lower for kw in after_sales_keywords):
        return "after_sales"
    else:
        return "general"


def detect_language(text: str) -> str:
    """检测文本的主要语言"""
    if not text:
        return 'zh'

    chinese_chars = sum(1 for char in text if '\u4e00' <= char <= '\u9fff')
    english_chars = sum(1 for char in text if char.isascii() and char.isalpha())

    if chinese_chars > english_chars:
        return 'zh'
    elif english_chars > chinese_chars:
        return 'en'
    else:
        return 'zh'


def build_system_prompt(question_type: str, language: str) -> str:
    """
    根据问题类型和语言构建 System Prompt（标准答案风格版）

    基于5个标准答案样本总结的风格：
    1. technical: 直接列举状态/含义，用<PIC>分隔
    2. accessory: 结构化信息 + 配图标注
    3. logistics/after_sales/general: 亲切问候 + 具体细节 + 主动帮助
    """

    if language == 'en':
        prompts = {
            "technical": (
                "You explain technical indicators. Be concise and structured.\n\n"
                "**Rules**:\n"
                "1. List each state/meaning directly\n"
                "2. Use <PIC>pic_ID</PIC> to separate different states\n"
                "3. NO greetings or filler words\n"
                "4. Max 60 words\n\n"
                "**Example style**:\n"
                '"State A<PIC>State B<PIC>State C"'
            ),

            "accessory": (
                "You answer accessory questions. Be informative.\n\n"
                "**Rules**:\n"
                "1. Provide specific information (sizes, options, etc.)\n"
                "2. Use <PIC>pic_ID</PIC> for diagrams/tables\n"
                "3. Structure with clear sections\n"
                "4. Max 80 words\n\n"
                "**Example style**:\n"
                '"Size information\\n\\nSizes are shown below.<PIC>\\n\\nAdditional notes<PIC>"'
            ),

            "logistics": (
                "You handle logistics inquiries. Be friendly and detailed.\n\n"
                "**Rules**:\n"
                "1. Start with 'Hello' or greeting\n"
                "2. Give specific details (timeframes, conditions, costs)\n"
                "3. Offer proactive help\n"
                "4. Sound natural like a real customer service agent\n"
                "5. Max 120 words\n\n"
                "**Tone**: Warm, helpful, conversational"
            ),

            "after_sales": (
                "You handle after-sales issues. Be empathetic and solution-oriented.\n\n"
                "**Rules**:\n"
                "1. Start with apology if issue occurred\n"
                "2. Acknowledge the problem specifically\n"
                "3. Provide clear solution steps\n"
                "4. Mention timelines and what info needed\n"
                "5. Sound sincere and professional\n"
                "6. Max 120 words\n\n"
                "**Tone**: Empathetic, responsible, helpful"
            ),

            "general": (
                "You answer general product questions. Be friendly and helpful.\n\n"
                "**Rules**:\n"
                "1. Start with appropriate greeting\n"
                "2. Answer ALL questions asked\n"
                "3. Provide specific, concrete information\n"
                "4. Offer additional assistance\n"
                "5. Sound natural, not robotic\n"
                "6. Max 100 words\n\n"
                "**Tone**: Friendly, professional, conversational"
            )
        }
    else:
        prompts = {
            "technical": (
                "解释技术指标。简洁结构化。\n\n"
                "**要求**：\n"
                "1. 直接列举每个状态/含义\n"
                "2. 用<PIC>pic_ID</PIC>分隔不同状态\n"
                "3. 不要问候语或填充词\n"
                "4. 60字以内\n\n"
                "**示例风格**：\n"
                '"状态A<PIC>状态B<PIC>状态C"'
            ),

            "accessory": (
                "回答配件咨询。提供详细信息。\n\n"
                "**要求**：\n"
                "1. 给出具体信息（尺寸、选项等）\n"
                "2. 用<PIC>pic_ID</PIC>标注图表/表格\n"
                "3. 清晰分段\n"
                "4. 80字以内\n\n"
                "**示例风格**：\n"
                '"尺寸信息\\n\\n尺寸如下所示。<PIC>\\n\\n补充说明<PIC>"'
            ),

            "logistics": (
                "处理物流咨询。亲切详细。\n\n"
                "**要求**：\n"
                "1. 以'您好'开头\n"
                "2. 给出具体细节（时间、条件、费用）\n"
                "3. 主动提供帮助\n"
                "4. 像真人客服一样自然\n"
                "5. 120字以内\n\n"
                "**语气**：温暖、乐于助人、对话式"
            ),

            "after_sales": (
                "处理售后问题。共情且以解决方案为导向。\n\n"
                "**要求**：\n"
                "1. 如有问题先道歉\n"
                "2. 具体承认问题\n"
                "3. 提供清晰的解决步骤\n"
                "4. 说明时间和所需信息\n"
                "5. 真诚专业\n"
                "6. 120字以内\n\n"
                "**语气**：共情、负责任、乐于助人"
            ),

            "general": (
                "回答一般产品问题。友好有帮助。\n\n"
                "**要求**：\n"
                "1. 适当问候开头\n"
                "2. 回答所有提出的问题\n"
                "3. 提供具体明确的信息\n"
                "4. 主动提供额外帮助\n"
                "5. 自然不生硬\n"
                "6. 100字以内\n\n"
                "**语气**：友好、专业、对话式"
            ),

            "general": (
                "回答一般产品问题。\n\n"
                "**关键规则**：\n"
                "1. 如果【检索到的精炼知识】中有相关内容，直接原样输出，不要改写\n"
                "2. 禁止添加'根据手册'、'您好'等额外文字\n"
                # ... 其他要求不变 ...
            ),
            "technical": (
                "解释技术指标。\n\n"
                "**关键规则**：\n"
                "1. 直接从【检索到的精炼知识】中复制相关内容\n"
                "2. 禁止添加任何解释性文字\n"
                # ... 其他要求不变 ...
            )
        }

    return prompts.get(question_type, prompts["general"])


def process_chat(question: str, session_id: str = None, image_base64: str = None) -> Dict[str, Any]:
    """
    处理聊天请求的核心逻辑（双模型RAG优化版）

    处理流程：
    1. 视觉分析（如果有图片）→ Qwen3-VL-8B
    2. RAG 检索精炼知识（快速匹配）
    3. 检测问题类型 + 语言 → 构建针对性 System Prompt
    4. 文本模型生成回答 → DeepSeek-V3.2
       - 纯文本问题：直接使用问题
       - 图片+文本：使用图片描述 + 问题
    5. 提取 <PIC> 标签 → 构建 ret 列表
    6. 更新会话历史
    """

    is_batch_mode = (session_id is None)

    if not session_id:
        session_id = str(uuid.uuid4())

    if not is_batch_mode:
        if session_id not in session_histories:
            session_histories[session_id] = []
        history = session_histories[session_id]
    else:
        history = []

    # Step 1: 视觉分析（如果有图片）→ 使用Qwen3-VL-8B
    image_description = ""
    if image_base64:
        try:
            logger.info("🔍 调用视觉模型分析图片...")
            vision_messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}},
                        {"type": "text", "text": "请详细描述这张图片的内容，包括所有可见的细节、文字、按钮、指示灯等"}
                    ]
                }
            ]
            vision_response = vision_agent.invoke(vision_messages)
            image_description = f"【图片内容描述】{vision_response.content}\n\n"
            logger.info("✅ 视觉分析完成")
        except Exception as e:
            logger.error(f"视觉分析失败：{str(e)}")
            image_description = ""

    # Step 2: RAG检索（优先使用精炼向量库）
    rag_context, related_image_ids = retrieve_relevant_content(question, top_k=config.RAG_TOP_K)
    rag_info = ""

    if rag_context:
        rag_info = f"\n【检索到的精炼知识】\n{rag_context}\n"

        if related_image_ids:
            for img_id in related_image_ids:
                rag_info += f"<PIC>{img_id}</PIC>\n"

    # Step 3: 检测问题类型并构建针对性Prompt
    question_language = detect_language(question)
    question_type = detect_question_type(question)
    system_prompt = build_system_prompt(question_type, question_language)

    # 组合图片描述、RAG结果、用户问题
    enhanced_query = f"{image_description}{rag_info}\n\n用户问题：{question}"

    messages = [
        SystemMessage(content=system_prompt),
        *history[-6:],
        HumanMessage(content=enhanced_query)
    ]

    # Step 4: 调用文本模型生成回答 → 使用DeepSeek-V3.2
    logger.info("📝 调用文本模型生成答案...")
    response = text_agent.invoke(messages)
    answer = response.content
    logger.info("✅ 答案生成完成")

    # Step 5: 空答案兜底处理
    if not answer or not answer.strip():
        if question_language == 'en':
            answer = "Thank you for your inquiry. Please contact our support team at 400-XXX-XXXX for assistance."
        else:
            answer = "感谢您的咨询。如需帮助，请拨打400-XXX-XXXX联系我们的支持团队。"
        logger.warning(f"模型返回空答案，使用默认回复。问题：{question[:50]}")

    # Step 6: 正则提取<PIC>标签
    pic_tags = re.findall(r'<PIC>([^<]+)</PIC>', answer)
    ret_list = list(dict.fromkeys(pic_tags))

    # Step 7: 更新会话历史（仅非批量模式）
    if not is_batch_mode:
        history.append(HumanMessage(content=question))
        history.append(AIMessage(content=answer))

        if len(history) > 12:
            session_histories[session_id] = history[-12:]

    # Step 8: 构建标准响应
    response_data = {
        "answer": answer,
        "session_id": session_id,
        "ret": ret_list
    }

    return response_data


# ==============================================
# 【API端点】
# ==============================================
@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest, token: str = Depends(verify_token)):
    """
    智能客服对话接口（核心端点）

    鉴权：需在 Header 中携带 Authorization: Bearer {KAFU_API_TOKEN}
    """
    try:
        logger.info(
            f"收到请求：session_id={request.session_id}, question={request.question[:50] if request.question else 'N/A'}")

        result = process_chat(
            question=request.question,
            session_id=request.session_id,
            image_base64=request.image
        )

        return ChatResponse(code=0, data=result)

    except Exception as e:
        logger.error(f"处理请求失败：{str(e)}")
        raise HTTPException(status_code=500, detail=f"服务器内部错误：{str(e)}")


@app.get("/health")
async def health_check():
    """健康检查接口"""
    return {
        "status": "healthy",
        "kb_loaded": refined_vector_store is not None or vector_store is not None,
        "sessions": len(session_histories)
    }


@app.get("/stats")
async def get_stats(token: str = Depends(verify_token)):
    """统计信息接口"""
    return {
        "code": 0,
        "data": {
            "total_sessions": len(session_histories),
            "kb_status": "refined_loaded" if refined_vector_store else ("loaded" if vector_store else "not_loaded"),
            "image_metadata_count": len(image_metadata)
        }
    }


# ==============================================
# 【文档精炼工具函数】
# ==============================================
@app.post("/refine-kb")
async def refine_knowledge_base(token: str = Depends(verify_token)):
    """
    手动触发知识库精炼（用于更新或重新构建精炼向量库）

    功能：
    1. 删除现有的精炼向量库
    2. 重新从原始文档精炼并构建

    返回：
        精炼任务状态
    """
    try:
        logger.info("开始手动触发文稿精炼...")

        # 删除现有的精炼向量库
        import shutil
        if os.path.exists(config.REFINED_VECTOR_DB_PATH):
            shutil.rmtree(config.REFINED_VECTOR_DB_PATH)
            logger.info(f"已删除旧的精炼向量库：{config.REFINED_VECTOR_DB_PATH}")

        # 重新加载知识库（会自动触发精炼流程）
        script_dir = os.path.dirname(os.path.abspath(__file__))
        manuals_path = os.path.join(script_dir, "manuals")

        global refined_vector_store
        refined_vector_store = None

        kb_loaded = load_knowledge_base(manuals_path)

        if kb_loaded:
            return {
                "code": 0,
                "message": "知识库精炼完成",
                "data": {
                    "status": "success",
                    "refined_db_path": config.REFINED_VECTOR_DB_PATH
                }
            }
        else:
            raise HTTPException(status_code=500, detail="知识库精炼失败")

    except Exception as e:
        logger.error(f"知识库精炼失败：{str(e)}")
        raise HTTPException(status_code=500, detail=f"精炼失败：{str(e)}")


@app.get("/kb-status")
async def get_kb_status(token: str = Depends(verify_token)):
    """
    查询知识库状态

    返回：
        原始向量库和精炼向量库的状态信息
    """
    original_exists = os.path.exists(config.VECTOR_DB_PATH)
    refined_exists = os.path.exists(config.REFINED_VECTOR_DB_PATH)

    # 检查JSONL文件
    script_dir = os.path.dirname(os.path.abspath(__file__))
    jsonl_path = os.path.join(script_dir, "processed_knowledge.jsonl")
    jsonl_exists = os.path.exists(jsonl_path)

    return {
        "code": 0,
        "data": {
            "original_vector_db": {
                "exists": original_exists,
                "path": config.VECTOR_DB_PATH
            },
            "refined_vector_db": {
                "exists": refined_exists,
                "path": config.REFINED_VECTOR_DB_PATH
            },
            "preprocessed_jsonl": {
                "exists": jsonl_exists,
                "path": jsonl_path
            },
            "active_store": "refined" if refined_exists else (
                "jsonl" if jsonl_exists else ("original" if original_exists else "none"))
        }
    }


# ==============================================
# 【服务启动入口】
# ==============================================
if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("🚀 AI智能客服 API 服务 v4.0（双模型RAG优化版）启动")
    logger.info(f"📍 地址: http://{config.API_HOST}:{config.API_PORT}")
    logger.info(f"🔑 Token: {config.KAFU_API_TOKEN}")
    logger.info(f"🌍 支持语言: 中文 / English")
    logger.info(f"✨ 视觉模型: {config.VISION_MODEL}")
    logger.info(f"✨ 文本模型: {config.TEXT_MODEL}")
    logger.info(f"✨ 特性: 双模型协同 + 两阶段RAG + 文档精炼")
    logger.info(f"✨ 优势: 视觉理解(Qwen3-VL) + 文本生成(DeepSeek-V3.2)")
    logger.info("=" * 60)

    logger.info("💡 提示：首次启动会进行文档精炼，可能需要较长时间")
    logger.info("💡 提示：后续启动会直接加载精炼向量库，速度更快")
    logger.info("💡 提示：如需重新精炼，调用 POST /refine-kb 接口")

    uvicorn.run(
        app,
        host=config.API_HOST,
        port=config.API_PORT,
        log_level="info"
    )

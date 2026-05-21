#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
离线文档预处理脚本 V2.0（标准答案样本对齐版 - 完整实现）
根据官方提供的5个标准答案样本，将产品手册提炼为标准化知识点。

特性：
1. 支持 PDF/DOCX/TXT 多格式文档
2. 视觉模型图片描述
3. 两阶段RAG优化：离线精炼 + 在线检索
4. 断点续传机制
5. 双格式输出：TXT + JSONL
6. 自动重试 + 进度可视化
7. 符合项目规范：2秒延迟、详细错误日志
8. ✅ 保留页码信息和图片ID映射（ManualXX_YY格式）
9. ✅ 自动清洗JSON格式残留脏数据

使用方法：
    python refine_knowledge_base.py                  # 正常模式（跳过已处理）
    python refine_knowledge_base.py --force          # 强制重新处理所有文件
    python refine_knowledge_base.py --reset          # 重置进度并重新处理
"""

import os
import sys
import time
import base64
import logging
import warnings
import json
import re
import argparse
from typing import List, Dict, Tuple
from pathlib import Path
from dotenv import load_dotenv

warnings.filterwarnings("ignore")
load_dotenv()


# ---------------------------
# 配置管理
# ---------------------------
class Config:
    """系统配置类"""
    # 输入输出路径
    INPUT_DIR = "manuals"
    OUTPUT_TXT = "processed_knowledge.txt"
    OUTPUT_JSONL = "processed_knowledge.jsonl"
    PROGRESS_FILE = "preprocess_progress.txt"
    ERROR_LOG = "preprocess_errors.log"
    TEMP_IMAGE_DIR = "temp_images"

    # 视觉模型配置（用于描述图片）
    VISION_MODEL = "Qwen/Qwen2-VL-72B-Instruct"
    VISION_TIMEOUT = 120

    # 精炼模型配置（用聪明模型做离线提炼）
    REFINE_MODEL = "Pro/deepseek-ai/DeepSeek-V3.2"
    REFINE_TIMEOUT = 300
    REFINE_TEMPERATURE = 0.1

    # 文本切分配置
    CHUNK_SIZE = 6000
    CHUNK_OVERLAP = 300

    # 批量处理配置（符合项目规范）
    REQUEST_DELAY = 2.0  # ✅ 2秒延迟，避免API限流
    MAX_RETRIES = 2  # 最多重试2次
    RETRY_WAIT = 2  # 重试间隔2秒

    # API配置
    SILICON_API_BASE = "https://api.siliconflow.cn/v1"


config = Config()

# ---------------------------
# 手册编号映射表（需根据实际情况填写）
# ---------------------------
MANUAL_ID_MAP = {
    # 示例映射，请根据实际文件修改
    # "VR头显手册": "Manual38",
    # "冰箱手册": "Manual01",
    # "空调手册": "Manual02",
}


def get_manual_id(filename: str) -> str:
    """获取手册的编号ID

    Args:
        filename: 文件名（不含扩展名）

    Returns:
        手册ID，如 Manual38
    """
    manual_name = Path(filename).stem

    # 优先使用映射表
    if manual_name in MANUAL_ID_MAP:
        return MANUAL_ID_MAP[manual_name]

    # 如果没有映射，使用哈希生成（仅作临时方案）
    hash_id = abs(hash(manual_name)) % 100
    logger.warning(f"未找到 {manual_name} 的手册ID映射，使用自动生成：Manual{hash_id:02d}")
    return f"Manual{hash_id:02d}"


# ---------------------------
# 日志配置
# ---------------------------
def setup_logging():
    """配置日志系统"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("preprocess.log", encoding='utf-8')
        ]
    )
    return logging.getLogger(__name__)


logger = setup_logging()

# ---------------------------
# 依赖导入
# ---------------------------
try:
    from langchain_openai import ChatOpenAI
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_core.messages import HumanMessage, SystemMessage
    import fitz  # PyMuPDF
    import docx
    from tqdm import tqdm
except ImportError as e:
    logger.error(f"缺少依赖包：{e}")
    logger.info("请安装：pip install langchain-openai langchain-text-splitters pymupdf python-docx tqdm")
    sys.exit(1)


# ---------------------------
# 环境验证
# ---------------------------
def validate_environment():
    """验证环境变量和目录结构"""
    if not os.getenv("SILICON_API_KEY"):
        raise ValueError("❌ 未设置 SILICON_API_KEY 环境变量，请在 .env 文件中配置")

    if not os.path.exists(config.INPUT_DIR):
        raise FileNotFoundError(f"❌ 输入目录不存在：{config.INPUT_DIR}")

    # 创建临时图片目录
    os.makedirs(config.TEMP_IMAGE_DIR, exist_ok=True)

    logger.info("✅ 环境验证通过")


# ---------------------------
# 文本清洗模块
# ---------------------------
def clean_text_content(text: str) -> str:
    """
    清洗文本内容，移除JSON格式残留和其他脏数据

    Args:
        text: 原始文本

    Returns:
        清洗后的文本
    """
    if not text or not text.strip():
        return text

    original_length = len(text)

    # 1. 移除开头的 JSON 数组标记 ["
    if text.startswith('["'):
        text = text[2:]
        logger.debug("移除开头的 [\" 标记")

    # 2. 移除结尾的 JSON 数组标记 ", ["xxx", "yyy", ...]] 或 "]
    # 匹配模式：, ["Manual38_0", "vr_01", ...]]
    pattern_trailing_json = r',\s*\[[^\]]*\]\s*\]\s*$'
    if re.search(pattern_trailing_json, text):
        text = re.sub(pattern_trailing_json, '', text)
        logger.debug("移除结尾的 JSON 数组标记")

    # 匹配简单的 "] 结尾
    elif text.rstrip().endswith('"]'):
        text = text.rstrip()[:-2]
        logger.debug("移除结尾的 \"] 标记")

    # 3. 清理多余的空行（连续4个以上空行缩减为2个）
    text = re.sub(r'\n{4,}', '\n\n\n', text)

    # 4. 移除首尾空白
    text = text.strip()

    cleaned_length = len(text)
    if original_length != cleaned_length:
        logger.info(
            f"文本清洗：{original_length} -> {cleaned_length} 字符（减少 {original_length - cleaned_length} 字符）")

    return text


# ---------------------------
# 文档解析模块
# ---------------------------
def extract_text_from_word(path: str) -> str:
    """从Word文档提取文本"""
    try:
        doc = docx.Document(path)
        text = "\n".join([para.text for para in doc.paragraphs])

        # 清洗文本
        text = clean_text_content(text)

        logger.debug(f"Word文档提取完成，长度：{len(text)}字符")
        return text
    except Exception as e:
        logger.error(f"Word文档提取失败 {path}：{e}")
        raise


def extract_text_from_txt(path: str) -> str:
    """从TXT文件读取文本（带清洗）"""
    try:
        with open(path, 'r', encoding='utf-8') as f:
            text = f.read()

        # 清洗文本
        text = clean_text_content(text)

        logger.debug(f"TXT文件读取完成，长度：{len(text)}字符")
        return text
    except UnicodeDecodeError:
        # 尝试其他编码
        with open(path, 'r', encoding='gbk') as f:
            text = f.read()

        # 清洗文本
        text = clean_text_content(text)

        logger.debug(f"TXT文件读取完成（GBK编码），长度：{len(text)}字符")
        return text
    except Exception as e:
        logger.error(f"TXT文件读取失败 {path}：{e}")
        raise


def extract_text_and_images_from_pdf(pdf_path: str) -> Tuple[str, List[Tuple[str, str]]]:
    """从PDF提取文本和图片（保留页码信息和图片ID映射）

    Returns:
        (全文文本含页码标记, [(图片路径, 图片ID), ...])
    """
    try:
        doc = fitz.open(pdf_path)
        text_parts = []
        images_with_ids = []

        # 获取手册ID
        manual_id = get_manual_id(pdf_path)
        image_counter = 0  # 全局图片计数器

        for page_num, page in enumerate(doc):
            page_number = page_num + 1  # 页码从1开始

            # 提取文本并添加页码标记
            page_text = page.get_text()
            if page_text.strip():
                text_parts.append(f"[第{page_number}页]\n{page_text}")

            # 提取图片
            image_list = page.get_images(full=True)
            for img_index, img in enumerate(image_list):
                xref = img[0]
                try:
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    image_ext = base_image["ext"]

                    # 生成比赛要求的图片ID格式：ManualXX_YY
                    image_counter += 1
                    pic_id = f"{manual_id}_{image_counter:02d}"  # 例如：Manual38_01

                    # 保存图片到临时目录
                    img_filename = f"{pic_id}.{image_ext}"
                    img_path = os.path.join(config.TEMP_IMAGE_DIR, img_filename)

                    with open(img_path, "wb") as f:
                        f.write(image_bytes)
                    images_with_ids.append((img_path, pic_id))

                    # 在文本中插入图片位置标记
                    text_parts.append(f"<PIC>{pic_id}</PIC>")

                    logger.debug(f"提取图片：{pic_id} (页{page_number}, 图{img_index + 1})")

                except Exception as img_err:
                    logger.warning(f"PDF图片提取失败（页{page_number}，图{img_index + 1}）：{img_err}")

        doc.close()

        # 合并所有文本部分
        full_text = "\n".join(text_parts)

        logger.info(f"PDF提取完成：{len(full_text)}字符，{len(images_with_ids)}张图片")
        return full_text, images_with_ids

    except Exception as e:
        logger.error(f"PDF提取失败 {pdf_path}：{e}")
        raise


# ---------------------------
# 视觉模型模块
# ---------------------------
def describe_image(image_path: str, pic_id: str = "") -> str:
    """用视觉模型描述图片

    Args:
        image_path: 图片路径
        pic_id: 图片ID（如 Manual38_01）

    Returns:
        图片描述文本（包含图片ID）
    """
    try:
        # 读取图片并转base64
        with open(image_path, "rb") as f:
            image_data = base64.b64encode(f.read()).decode('utf-8')

        # 初始化视觉模型
        vision_llm = ChatOpenAI(
            api_key=os.getenv("SILICON_API_KEY"),
            base_url=config.SILICON_API_BASE,
            model=config.VISION_MODEL,
            request_timeout=config.VISION_TIMEOUT,
            max_retries=1
        )

        prompt = f"""请详细描述这张产品图片中的内容，包括：
1. 设备外观特征
2. 指示灯状态及颜色
3. 按钮标识及功能
4. 接口类型
5. 文字说明或标签

请用简洁的中文描述，控制在50字以内。
图片ID：{pic_id}"""

        response = vision_llm.invoke([
            HumanMessage(content=[
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {
                    "url": f"data:image/png;base64,{image_data}"
                }}
            ])
        ])

        description = response.content.strip()
        logger.debug(f"图片描述完成：{pic_id}")
        return f"[图片描述 {pic_id}] {description}"

    except Exception as e:
        logger.warning(f"图片描述失败 {image_path}：{e}")
        return f"[图片描述 {pic_id} 无法获取]"


# ---------------------------
# Prompt工程模块
# ---------------------------
def build_sample_based_prompt(chunk_text: str, source_name: str) -> str:
    """构建与5个标准答案样本对齐的提炼prompt

    Args:
        chunk_text: 待提炼的文本块
        source_name: 来源文件名

    Returns:
        完整的prompt字符串
    """
    return f"""你是产品文档智能提炼专家。你的任务是将以下文档片段，转换成可以直接用于客服回答的**标准化知识点**。

请严格按照官方给出的**5种标准答案样本风格**输出知识点，每个知识点一行，格式为：
问题关键词|标准答案要点|来源：{source_name}

## 五种标准风格及示例：

1. **技术说明类**（指示灯、按钮含义、状态标识）
   - 格式：直接列举状态/含义，用 <PIC>pic_ID</PIC> 分隔，无问候语
   - 示例：DCB107、DCB112 电池组充电中<PIC>电池组已充满<PIC>过热/过冷延迟<PIC>
   - 字数：≤60字

2. **配件咨询类**（表带、尺寸、规格、兼容性）
   - 格式：分段给出信息，<PIC> 标注图表/表格位置
   - 示例：表带尺寸\\n\\n表带尺寸如下所示。注意：单独销售的配件表带可能略有差异。\\n<PIC>\\n\\n环境条件\\n<PIC>
   - 字数：≤80字

3. **物流快递类**（配送范围、运费、时效）
   - 格式：以"您好"开头，给出具体细节（时间、费用、条件），主动提供帮助
   - 示例：您好，我们的商品支持送到大部分乡镇哦，具体能否送达，取决于您的收货地址，您可以告诉我详细的收货地址，我帮您查询。送到乡镇一般不需要额外加运费，和市区运费一致；物流时效会比市区稍慢，正常情况下，下单后48小时发货，乡镇地区3-5天可收到，偏远乡镇可能需要5-7天哦。
   - 字数：≤120字

4. **物流异常类**（待揽收、丢件、延误）
   - 格式：解释原因 + 解决方案 + 时间承诺，语气亲切负责
   - 示例：您好，物流显示待揽收，大概率是商品已打包完成，等待快递员上门取件哦，一般24小时内会完成揽收；若超过24小时仍未揽收，您可以联系我们客服，我们会催促快递方尽快上门。
   - 字数：≤120字

5. **售后维修类**（故障、维修、质量问题）
   - 格式：先道歉，承认问题责任，给出解决步骤和所需信息
   - 示例：您好，非常抱歉给您带来困扰！维修后短期内出现同样故障，且是上次维修不彻底导致的，属于我们的维修失误，支持免费重新维修，并延长维修质保期。请您提供维修单号、商品故障描述，我们立即安排专业维修人员处理。
   - 字数：≤120字

## 通用规则：
- 保留原文中的 <PIC>pic_ID</PIC> 标签，不要遗漏。
- 如果文档中有图片描述，请将其融入相应知识点。
- 每条知识点必须是**自包含的**，即使脱离上下文也能直接用于回答。
- 只用中文输出，每条知识点一行，不要其他解释。
- 若原文没有明确的问题场景，请根据内容推断可能的用户提问。
- **重要**：如果原文中包含页码标记（如[第5页]），请在知识点中保留该信息，便于追溯来源。

## 待提炼文档：
{chunk_text[:8000]}"""


# ---------------------------
# 精炼引擎模块
# ---------------------------
def refine_chunk(chunk: str, source: str, retry: int = 0) -> str:
    """调用精炼模型，提炼知识点

    Args:
        chunk: 文本块
        source: 来源文件名
        retry: 当前重试次数

    Returns:
        提炼后的知识点（多行）
    """
    try:
        llm = ChatOpenAI(
            api_key=os.getenv("SILICON_API_KEY"),
            base_url=config.SILICON_API_BASE,
            model=config.REFINE_MODEL,
            temperature=config.REFINE_TEMPERATURE,
            request_timeout=config.REFINE_TIMEOUT,
            max_retries=1
        )

        prompt = build_sample_based_prompt(chunk, source)
        response = llm.invoke([HumanMessage(content=prompt)])

        result = response.content.strip()
        logger.debug(f"精炼成功，生成 {len(result.split(chr(10)))} 条知识点")
        return result

    except Exception as e:
        if retry < config.MAX_RETRIES:
            logger.warning(f"精炼失败，{config.RETRY_WAIT}秒后重试（{retry + 1}/{config.MAX_RETRIES}）：{e}")
            time.sleep(config.RETRY_WAIT)
            return refine_chunk(chunk, source, retry + 1)
        else:
            logger.error(f"精炼彻底失败，使用降级策略：{e}")
            # 降级：返回原文摘要
            return f"原文摘要|{chunk[:200]}|来源：{source}"


# ---------------------------
# 进度管理模块
# ---------------------------
def load_progress() -> set:
    """加载已处理的文件列表"""
    processed = set()
    if os.path.exists(config.PROGRESS_FILE):
        with open(config.PROGRESS_FILE, 'r', encoding='utf-8') as f:
            processed = set(line.strip() for line in f if line.strip())
    return processed


def save_progress(filename: str):
    """保存处理进度"""
    with open(config.PROGRESS_FILE, 'a', encoding='utf-8') as f:
        f.write(filename + '\n')


def log_error(filename: str, error: str):
    """记录错误日志"""
    with open(config.ERROR_LOG, 'a', encoding='utf-8') as f:
        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} | {filename} | {error}\n")


# ---------------------------
# 输出模块
# ---------------------------
def save_to_txt(knowledge_lines: List[str], append: bool = True):
    """保存知识点到TXT文件

    Args:
        knowledge_lines: 知识点行列表
        append: 是否追加模式
    """
    mode = 'a' if append else 'w'
    with open(config.OUTPUT_TXT, mode, encoding='utf-8') as f:
        for line in knowledge_lines:
            if line.strip():
                f.write(line + '\n')
    logger.debug(f"TXT保存完成：{len(knowledge_lines)}条知识点")


def save_to_jsonl(knowledge_lines: List[str], append: bool = True):
    """保存知识点到JSONL文件（结构化格式）

    Args:
        knowledge_lines: 知识点行列表（格式：关键词|答案|来源）
        append: 是否追加模式
    """
    mode = 'a' if append else 'w'
    saved_count = 0

    with open(config.OUTPUT_JSONL, mode, encoding='utf-8') as f:
        for line in knowledge_lines:
            if not line.strip():
                continue

            parts = line.split('|', 2)
            if len(parts) == 3:
                record = {
                    "question_keyword": parts[0].strip(),
                    "answer": parts[1].strip(),
                    "source": parts[2].replace("来源：", "").strip()
                }
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
                saved_count += 1
            else:
                logger.warning(f"JSONL格式错误，跳过：{line[:50]}...")

    logger.debug(f"JSONL保存完成：{saved_count}条记录")


# ---------------------------
# 主处理流程
# ---------------------------
def process_single_file(filename: str) -> bool:
    """处理单个文档文件

    Args:
        filename: 文件名

    Returns:
        是否成功
    """
    filepath = os.path.join(config.INPUT_DIR, filename)
    logger.info(f"\n{'=' * 60}")
    logger.info(f"开始处理：{filename}")
    logger.info(f"{'=' * 60}")

    try:
        # 1. 提取文本和图片
        text = ""
        img_descs = []

        if filename.endswith('.pdf'):
            text, images_with_ids = extract_text_and_images_from_pdf(filepath)

            # 描述图片（images_with_ids 是 [(路径, ID), ...] 的列表）
            if images_with_ids:
                logger.info(f"  正在描述 {len(images_with_ids)} 张图片...")
                for i, (img_path, pic_id) in enumerate(images_with_ids):
                    logger.info(f"    图片 {i + 1}/{len(images_with_ids)} - {pic_id}")
                    desc = describe_image(img_path, pic_id)
                    img_descs.append(desc)
                    time.sleep(config.REQUEST_DELAY)

                # 将图片描述附加到文本末尾
                if img_descs:
                    text += "\n\n" + "\n\n".join(img_descs)

        elif filename.endswith('.docx'):
            text = extract_text_from_word(filepath)

        elif filename.endswith('.txt'):
            text = extract_text_from_txt(filepath)

        else:
            logger.warning(f"不支持的文件格式：{filename}")
            return False

        if not text.strip():
            logger.warning(f"文件内容为空：{filename}")
            return False

        logger.info(f"  文本提取完成：{len(text)}字符")

        # 2. 文本切分
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP,
            length_function=len
        )
        chunks = splitter.split_text(text)
        logger.info(f"  切分完成：共 {len(chunks)} 个块")

        # 3. 精炼每个块
        all_knowledge = []

        for i, chunk in enumerate(tqdm(chunks, desc=f"精炼 {filename}", unit="块")):
            logger.info(f"  处理块 {i + 1}/{len(chunks)}")

            try:
                knowledge = refine_chunk(chunk, filename)
                all_knowledge.append(knowledge)

                # 符合规范的延迟
                time.sleep(config.REQUEST_DELAY)

            except Exception as chunk_err:
                logger.error(f"  块 {i + 1} 处理失败：{chunk_err}")
                log_error(filename, f"块{i + 1}: {str(chunk_err)}")
                continue

        if not all_knowledge:
            logger.warning(f"未生成任何知识点：{filename}")
            return False

        # 4. 保存结果
        logger.info(f"  保存结果...")
        save_to_txt(all_knowledge, append=True)
        save_to_jsonl(all_knowledge, append=True)

        total_lines = sum(len(k.split('\n')) for k in all_knowledge)
        logger.info(f"  ✅ 完成！生成 {total_lines} 条知识点")

        # 5. 标记进度
        save_progress(filename)

        return True

    except Exception as e:
        error_msg = str(e)
        logger.error(f"  ❌ 处理失败：{error_msg}")
        log_error(filename, error_msg)
        return False


def main():
    """主函数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='离线文档预处理脚本 V2.0')
    parser.add_argument('--force', action='store_true',
                        help='强制重新处理所有文件（忽略进度记录）')
    parser.add_argument('--reset', action='store_true',
                        help='重置进度记录并重新处理所有文件')
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("离线文档预处理脚本 V2.0 - 启动")
    logger.info("=" * 60)

    try:
        # 验证环境
        validate_environment()

        # 获取待处理文件列表
        files = [f for f in os.listdir(config.INPUT_DIR)
                 if f.endswith(('.pdf', '.docx', '.txt'))]

        if not files:
            logger.error("❌ manuals目录中无文档文件")
            return

        logger.info(f"发现 {len(files)} 个文档文件")

        # 加载进度
        processed = set()

        if args.reset:
            # 重置模式：删除进度文件
            logger.info("🔄 重置模式：清除所有进度记录...")
            if os.path.exists(config.PROGRESS_FILE):
                os.remove(config.PROGRESS_FILE)
                logger.info(f"已删除进度文件：{config.PROGRESS_FILE}")

            if os.path.exists(config.ERROR_LOG):
                os.remove(config.ERROR_LOG)
                logger.info(f"已删除错误日志：{config.ERROR_LOG}")

            remaining_files = files
            logger.info("✅ 进度已重置，将重新处理所有文件")

        elif args.force:
            # 强制模式：忽略进度记录
            logger.info("⚡ 强制模式：忽略进度记录，重新处理所有文件")
            remaining_files = files
        else:
            # 正常模式：加载进度，跳过已处理
            processed = load_progress()
            remaining_files = [f for f in files if f not in processed]

            logger.info(f"已处理：{len(processed)} 个")
            logger.info(f"待处理：{len(remaining_files)} 个")

            if not remaining_files:
                logger.info("✅ 所有文件已处理完成！")
                logger.info("💡 提示：如需重新处理，使用 --force 或 --reset 参数")
                return

        # 处理文件
        success_count = 0
        fail_count = 0

        for idx, filename in enumerate(remaining_files, 1):
            logger.info(f"\n进度：{idx}/{len(remaining_files)}")

            success = process_single_file(filename)

            if success:
                success_count += 1
            else:
                fail_count += 1

        # 统计总结
        logger.info("\n" + "=" * 60)
        logger.info("预处理完成统计")
        logger.info("=" * 60)
        logger.info(f"总文件数：{len(files)}")
        logger.info(f"本次处理：{len(remaining_files)}")
        logger.info(f"成功处理：{success_count}")
        logger.info(f"失败数量：{fail_count}")
        logger.info(f"输出文件：")
        logger.info(f"  - TXT格式：{config.OUTPUT_TXT}")
        logger.info(f"  - JSONL格式：{config.OUTPUT_JSONL}")
        logger.info(f"  - 进度记录：{config.PROGRESS_FILE}")
        logger.info(f"  - 错误日志：{config.ERROR_LOG}")
        logger.info("=" * 60)

        if fail_count > 0:
            logger.warning(f"⚠️ 有 {fail_count} 个文件处理失败，请查看 {config.ERROR_LOG}")

    except Exception as e:
        logger.error(f"❌ 程序异常：{e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

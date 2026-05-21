# AI智能客服 API 使用指南

## 🏗️ 系统架构与模块设计

### 整体架构

```
┌─────────────────────────────────────────────────────────┐
│                    FastAPI REST API                      │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │ Bearer Token │  │ 统一异常处理  │  │ 请求验证中间件│  │
│  │   认证层     │  │   中间件层    │  │   数据校验    │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  │
└─────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────┐
│                   业务逻辑层 (Core)                       │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │ 客服模板匹配  │  │ 多模态处理    │  │ RAG检索引擎  │  │
│  │   引擎       │  │   引擎       │  │              │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  │
└─────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────┐
│                   AI模型层 (Models)                       │
│  ┌──────────────┐  ┌──────────────┐                     │
│  │ 文本模型      │  │ 视觉模型      │                     │
│  │ DeepSeek-V3.2 │  │ Qwen3-VL-32B│                     │
│  │ (超时20s)    │  │ (超时30s)    │                     │
│  └──────────────┘  └──────────────┘                     │
└─────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────┐
│                  数据存储层 (Storage)                     │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │ FAISS向量库   │  │ 会话历史存储  │  │ 图片元数据    │  │
│  │ (BGE嵌入)    │  │ (内存字典)    │  │ (文件索引)    │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  │
└─────────────────────────────────────────────────────────┘
```

### 核心模块说明

#### 1. API接口层 (`api_server.py`)

**功能**：提供符合竞赛规范的REST API接口

**主要组件**：
- **认证中间件**：Bearer Token验证（KAFU_API_TOKEN）
- **异常处理器**：统一JSON格式错误响应
- **数据模型**：Pydantic验证（ChatRequest/ChatResponse）
- **路由端点**：POST /chat、GET /health、GET /stats

**关键特性**：
- ✅ 支持0-3张图片列表输入
- ✅ 严格超时控制（文本20s/视觉30s）
- ✅ 标准JSON响应格式 `{"code": 0, "data": {...}}`

---

#### 2. 客服模板引擎 (`CUSTOMER_SERVICE_TEMPLATES`)

**功能**：快速响应常见客服问题，无需RAG检索

**支持的场景**：
| 场景 | 关键词 | 响应速度 |
|-----|--------|---------|
| 物流查询 | 物流、快递、包裹 | <0.1秒 |
| 发货时间 | 发货、多久发 | <0.1秒 |
| 退货政策 | 退货、换货 | <0.1秒 |
| 保修政策 | 保修、质保 | <0.1秒 |
| 维修服务 | 维修、坏了 | <0.1秒 |
| 发票问题 | 发票、开票 | <0.1秒 |
| 支付方式 | 支付、付款 | <0.1秒 |
| 退款流程 | 退款、退钱 | <0.1秒 |

**工作流程**：
```
用户输入 → 关键词匹配 → 返回预设模板 → 跳过RAG检索
```

---

#### 3. 多模态处理引擎

**功能**：支持纯文字、纯图片、文字+多图（最多3张）的混合输入

**处理流程**：
```
接收images列表
    ↓
循环分析每张图片（Qwen-VL-32B）
    ↓
生成图片描述文本
    ↓
拼接所有图片描述
    ↓
传递给文本模型
```

**关键技术**：
- Base64图片编码/解码
- 视觉模型并发调用（单图分析）
- 图片描述结构化输出

---

#### 4. RAG检索引擎

**功能**：从知识库中检索相关内容和配图

**工作流程**：
```
用户问题
    ↓
FAISS向量相似度搜索（top_k=3）
    ↓
获取相关文本块 + 图片元数据
    ↓
构建<PIC>占位符引用
    ↓
注入System Prompt
```

**核心技术**：
- **向量数据库**：FAISS（本地存储）
- **嵌入模型**：BAAI/bge-large-zh-v1.5
- **文本分块**：RecursiveCharacterTextSplitter（500字符/块）
- **多格式支持**：PDF（文字+图片）、Word（文字）、TXT（文字）

**`<PIC>`标签机制**：
```python
# System Prompt要求模型保留标签
"回答中如果涉及配图，请用 <PIC>图片ID</PIC> 的格式引用"

# 从答案中提取标签
pic_tags = re.findall(r'<PIC>(pic_\d+)</PIC>', answer)
ret_list = list(dict.fromkeys(pic_tags))  # 去重并保持顺序
```

---

#### 5. AI模型层

**文本模型**：
- **模型**：deepseek-ai/DeepSeek-V4-Flash
- **用途**：生成最终回答
- **超时**：20秒（赛题要求）
- **特性**：轻量化、快速响应

**视觉模型**：
- **模型**：Qwen/Qwen3-VL-32B-Instruct
- **用途**：图片内容理解
- **超时**：30秒（赛题要求）
- **特性**：多模态理解、图文关联

---

#### 6. 会话管理模块

**功能**：维护多轮对话历史

**存储结构**：
```python
session_histories = {
    "session_id_1": [
        HumanMessage(content="问题1"),
        AIMessage(content="回答1"),
        HumanMessage(content="问题2"),
        AIMessage(content="回答2")
    ],
    "session_id_2": [...]
}
```

**关键特性**：
- 使用LangChain消息对象（HumanMessage/AIMessage）
- 自动限制历史长度（最多12条消息）
- 支持并发会话隔离

---

#### 7. 知识库构建模块

**功能**：解析文档并构建向量数据库

**支持格式**：
| 格式 | 解析库 | 提取内容 |
|-----|--------|---------|
| PDF | PyMuPDF (fitz) | 文字 + 图片 |
| Word (.docx) | python-docx | 文字 |
| TXT | 内置open() | 文字（自动编码识别） |

**构建流程**：
```
扫描manuals文件夹
    ↓
解析文档（提取文字+图片）
    ↓
文本分块（500字符/块，50重叠）
    ↓
生成向量嵌入（BGE模型）
    ↓
存储到FAISS向量库
    ↓
保存图片元数据
```

---

### 数据流向图

```
用户请求
    ↓
┌─────────────────┐
│  API接口层       │ ← Bearer Token认证
└─────────────────┘
    ↓
┌─────────────────┐
│  模板匹配？      │ → 是 → 返回模板答案（快速路径）
└─────────────────┘
    ↓ 否
┌─────────────────┐
│  有图片？        │ → 是 → 视觉模型分析（最多3张）
└─────────────────┘
    ↓
┌─────────────────┐
│  RAG检索         │ → FAISS相似度搜索
└─────────────────┘
    ↓
┌─────────────────┐
│  构建Prompt      │ ← 图片描述 + RAG内容 + <PIC>标签
└─────────────────┘
    ↓
┌─────────────────┐
│  文本模型推理    │ → DeepSeek-V4-Flash
└─────────────────┘
    ↓
┌─────────────────┐
│  提取<PIC>标签   │ → 正则表达式提取ret列表
└─────────────────┘
    ↓
┌─────────────────┐
│  更新会话历史    │ → HumanMessage/AIMessage
└─────────────────┘
    ↓
标准JSON响应
```

---

## 🚀 快速开始

### 第1步：安装依赖

```bash
pip install fastapi uvicorn python-multipart
```

### 第2步：配置环境变量

在 `.env` 文件中添加（或使用 `.env.example` 作为模板）：
```env
SILICON_API_KEY=your_api_key_here
KAFU_API_TOKEN=your-secret-token-here  # 赛题规范变量名
```
> ⚠️ **重要提示**：为保障API密钥安全，本项目不提供真实的API Key。请评委老师自行配置以下环境变量：
> - `SILICON_API_KEY`: Silicon Flow平台的API密钥（用于调用DeepSeek和Qwen模型）
> - `KAFU_API_TOKEN`: API访问认证Token（可自定义任意字符串）
> 
> 获取方式：
> 1. 访问 [Silicon Flow官网](https://siliconflow.cn/) 注册账号
> 2. 在控制台生成API Key
> 3. 将Key填入`.env`文件或设置为环境变量

---


### 第3步：准备知识库

将 PDF/Word/TXT 文件放入 `manuals` 文件夹。

### 第4步：启动服务

```bash
cd C:\llama3\RAG
python api_server.py
```

服务将在 **http://localhost:8000** 启动。

---

## 📋 API 接口文档

### 1. 智能对话接口

**端点**: `POST /chat`

**认证**: Bearer Token

**请求体**:
```json
{
  "question": "如何更换电池？",
  "session_id": "optional-session-id",
  "images": ["base64_encoded_image_1", "base64_encoded_image_2"]
}
```

**字段说明**:
| 字段 | 类型 | 必填 | 说明 |
|-----|------|-----|------|
| question | string | ❌ | 用户问题（如果有图片可以不填） |
| session_id | string | ❌ | 会话ID（不提供则自动生成） |
| images | array | ❌ | Base64图片列表（最多3张） |

**注意**：`question` 和 `images` **至少需要提供一项**

**响应格式**:
```json
{
  "code": 0,
  "data": {
    "answer": "根据《产品手册》第15页...\n<PIC>pic_1</PIC>\n<PIC>pic_2</PIC>",
    "session_id": "abc-123-def-456",
    "ret": ["pic_1", "pic_2"]
  }
}
```

**字段说明**:
| 字段 | 类型 | 说明 |
|-----|------|------|
| code | int | 状态码（0=成功） |
| answer | string | AI回答（包含`<PIC>`占位符） |
| session_id | string | 会话ID |
| ret | array | 相关图片ID列表 |

---

### 2. 健康检查接口

**端点**: `GET /health`

**无需认证**

**响应**:
```json
{
  "status": "healthy",
  "kb_loaded": true,
  "sessions": 5
}
```

---

### 3. 统计信息接口

**端点**: `GET /stats`

**认证**: Bearer Token

**响应**:
```json
{
  "code": 0,
  "data": {
    "total_sessions": 5,
    "kb_status": "loaded",
    "image_metadata_count": 20
  }
}
```

---

## 💻 调用示例

### Python 示例

```python
import requests
import base64
import json

# 配置
API_URL = "http://localhost:8000/chat"
TOKEN = "your-secret-token-here"

headers = {
    "Authorization": f"Bearer {TOKEN}",
    "Content-Type": "application/json"
}

# 示例1: 纯文本问答
def text_chat():
    payload = {
        "question": "公考分为哪几部分？",
        "session_id": "session_001"
    }
    
    response = requests.post(API_URL, headers=headers, json=payload)
    result = response.json()
    
    print("回答:", result["data"]["answer"])
    print("会话ID:", result["data"]["session_id"])
    print("相关图片:", result["data"]["ret"])

# 示例2: 带图片的问答
def multimodal_chat():
    # 读取图片并转为Base64
    with open("test_image.jpg", "rb") as f:
        image_base64 = base64.b64encode(f.read()).decode('utf-8')
    
    payload = {
        "question": "这个按钮是做什么的？",
        "image": image_base64,
        "session_id": "session_002"
    }
    
    response = requests.post(API_URL, headers=headers, json=payload)
    result = response.json()
    
    print("回答:", result["data"]["answer"])
    print("相关图片ID:", result["data"]["ret"])

# 示例2.5: 仅图片问答（不提供问题）
def image_only_chat():
    # 读取图片并转为Base64
    with open("test_image.jpg", "rb") as f:
        image_base64 = base64.b64encode(f.read()).decode('utf-8')
    
    payload = {
        "image": image_base64,  # 只传图片，不传问题
        "session_id": "session_003"
    }
    
    response = requests.post(API_URL, headers=headers, json=payload)
    result = response.json()
    
    print("回答:", result["data"]["answer"])  # AI会自动描述图片内容

# 示例2.6: 多图片问答（最多3张）
def multi_image_chat():
    images_base64 = []
    
    # 读取多张图片
    for img_file in ["image1.jpg", "image2.jpg", "image3.jpg"]:
        with open(img_file, "rb") as f:
            images_base64.append(base64.b64encode(f.read()).decode('utf-8'))
    
    payload = {
        "question": "这些图片有什么区别？",
        "images": images_base64,  # 图片列表
        "session_id": "session_004"
    }
    
    response = requests.post(API_URL, headers=headers, json=payload)
    result = response.json()
    
    print("回答:", result["data"]["answer"])
    print("相关图片ID:", result["data"]["ret"])

# 示例3: 多轮对话
def multi_turn_chat():
    session_id = None
    
    questions = [
        "你好",
        "如何更换电池？",
        "保修期多久？"
    ]
    
    for question in questions:
        payload = {
            "question": question,
            "session_id": session_id  # 第一次为None，后续使用返回的session_id
        }
        
        response = requests.post(API_URL, headers=headers, json=payload)
        result = response.json()
        
        print(f"问: {question}")
        print(f"答: {result['data']['answer']}\n")
        
        # 更新session_id
        session_id = result["data"]["session_id"]

if __name__ == "__main__":
    text_chat()
    # multimodal_chat()
    # multi_turn_chat()
```

---

### cURL 示例

```bash
# 纯文本问答
curl -X POST http://localhost:8000/chat \
  -H "Authorization: Bearer your-secret-token-here" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "如何更换电池？",
    "session_id": "session_001"
  }'

# 带图片的问答
curl -X POST http://localhost:8000/chat \
  -H "Authorization: Bearer your-secret-token-here" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "这个指示灯闪烁代表什么？",
    "image": "iVBORw0KGgoAAAANSUhEUgAA...",
    "session_id": "session_002"
  }'
```

---

### JavaScript 示例

```javascript
const API_URL = 'http://localhost:8000/chat';
const TOKEN = 'your-secret-token-here';

async function chat(question, sessionId = null, imageBase64 = null) {
  const payload = {
    question: question,
    session_id: sessionId,
    image: imageBase64
  };
  
  const response = await fetch(API_URL, {
    method: 'POST',
    headers: {
      'Authorization': `Bearer ${TOKEN}`,
      'Content-Type': 'application/json'
    },
    body: JSON.stringify(payload)
  });
  
  const result = await response.json();
  return result;
}

// 使用示例
(async () => {
  const result = await chat('如何更换电池？');
  console.log('回答:', result.data.answer);
  console.log('相关图片:', result.data.ret);
})();
```

---

## 🔑 客服模板库说明

### 支持的场景

| 场景 | 关键词 | 示例问题 |
|-----|--------|---------|
| 物流查询 | 物流、快递、包裹 | "我的包裹到哪了？" |
| 发货时间 | 发货、多久发 | "什么时候发货？" |
| 退货政策 | 退货、换货 | "可以退货吗？" |
| 保修政策 | 保修、质保 | "保修期多久？" |
| 维修服务 | 维修、坏了 | "设备坏了怎么办？" |
| 发票问题 | 发票、开票 | "能开发票吗？" |
| 支付方式 | 支付、付款 | "支持哪些支付方式？" |
| 退款流程 | 退款、退钱 | "怎么申请退款？" |

### 工作原理

1. 用户提问 → 关键词匹配
2. 匹配成功 → 返回预设模板答案
3. 匹配失败 → 进入 RAG 检索流程

**优势**：
- ⚡ 响应速度快（无需检索）
- ✅ 答案准确一致
- 🎯 适合常见问题

---

## 🖼️ 图片处理说明

### `<PIC>` 占位符格式

当回答中引用相关配图时，使用以下格式：

```
根据《产品手册》第15页，更换电池的步骤如下：

步骤1: 打开后盖...
<PIC>pic_1</PIC>

步骤2: 取出旧电池...
<PIC>pic_2</PIC>
```

### 图片ID列表

响应中的 `ret` 字段包含所有相关图片的ID：

```json
{
  "ret": ["pic_1", "pic_2", "pic_3"]
}
```

### 获取实际图片

图片存储在：
```
C:\llama3\RAG\knowledge_base\extracted_images\
```

文件名格式：`page{页码}_img{序号}.png`

---

## 🛠️ 高级配置

### 修改端口

编辑 `api_server.py`：
```python
class Config:
    API_PORT = 8000  # 改为其他端口
```

### 修改Token

**方法1**: 环境变量
```env
API_TOKEN=my-new-secret-token
```

**方法2**: 代码中修改
```python
class Config:
    API_TOKEN = "my-new-secret-token"
```

### 调整RAG参数

```python
class Config:
    CHUNK_SIZE = 500      # 文本分块大小
    CHUNK_OVERLAP = 50    # 分块重叠
    MAX_IMAGE_SIZE_MB = 5 # 最大图片大小
```

---

## 📊 性能优化建议

### 1. 启用缓存

对于高频问题，可以添加Redis缓存：

```python
import redis

redis_client = redis.Redis(host='localhost', port=6379, db=0)

def get_cached_answer(question):
    cached = redis_client.get(question)
    if cached:
        return json.loads(cached)
    return None

def cache_answer(question, answer):
    redis_client.setex(question, 3600, json.dumps(answer))  # 缓存1小时
```

### 2. 异步处理

FastAPI 天然支持异步，可以处理并发请求。

### 3. 负载均衡

使用 Nginx 反向代理：

```nginx
upstream api_servers {
    server 127.0.0.1:8000;
    server 127.0.0.1:8001;
    server 127.0.0.1:8002;
}

server {
    listen 80;
    
    location / {
        proxy_pass http://api_servers;
    }
}
```

---

## 🔍 调试技巧

### 查看日志

启动时添加详细日志：
```bash
python api_server.py --log-level debug
```

### 测试接口

使用 Swagger UI：
```
http://localhost:8000/docs
```

### 监控性能

添加响应时间监控：
```python
import time

@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response
```

---

## ❓ 常见问题

### Q1: 401 Unauthorized

**原因**: Token 无效或缺失

**解决**:
```python
headers = {
    "Authorization": "Bearer your-correct-token"
}
```

### Q2: 回答中没有 `<PIC>` 占位符

**原因**: 
1. 知识库中没有相关图片
2. RAG 未检索到相关内容

**解决**: 检查 `manuals` 文件夹中是否有带图片的PDF

### Q3: 响应速度慢

**原因**: 
1. RAG 检索耗时
2. 模型推理慢

**解决**:
- 减少 `CHUNK_SIZE`
- 使用更快的模型
- 启用缓存

### Q4: 会话历史不连续

**原因**: `session_id` 不一致

**解决**: 确保每次请求使用相同的 `session_id`

---

## 📞 技术支持

如遇问题：
1. 查看控制台日志
2. 检查 `.env` 配置
3. 确认知识库已加载
4. 访问 Swagger UI 测试接口

---



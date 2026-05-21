"""
批量问答处理脚本
读取CSV文件中的问题，调用API获取回答，生成JSON结果文件
"""
import csv
import json
import requests
import time
from typing import List, Dict
import os

# 配置
API_URL = "http://localhost:8000/chat"
API_TOKEN = "Wyx200349"  # 从.env读取或手动配置
INPUT_CSV = r"C:\llama3\手册\question_public.csv"  # 输入CSV文件名
OUTPUT_JSON = "answers.json"  # 输出JSON文件名
REQUEST_DELAY = 2  # 请求间隔（秒），避免频繁请求
MAX_RETRIES = 2  # 最大重试次数
RETRY_DELAY = 2  # 重试间隔（秒）
REQUEST_TIMEOUT = 35  # 请求超时时间（秒），比服务器端超时多15秒余量



def read_questions_from_csv(csv_file: str) -> List[Dict]:
    """从CSV文件读取问题列表"""
    questions = []

    if not os.path.exists(csv_file):
        print(f"❌ 文件不存在: {csv_file}")
        return questions

    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)

        # 自动检测列名（支持多种格式）
        headers = reader.fieldnames
        print(f"📋 CSV文件列名: {headers}")

        # 尝试找到问题列
        question_col = None
        for col in headers:
            if 'question' in col.lower() or '问题' in col or 'query' in col.lower():
                question_col = col
                break

        if not question_col:
            # 如果没有找到，使用第一列
            question_col = headers[0]
            print(f"⚠️  未找到明确的问题列，使用第一列: {question_col}")
        else:
            print(f"✅ 检测到问题列: {question_col}")

        # 读取所有问题
        for idx, row in enumerate(reader, 1):
            question = row.get(question_col, '').strip()
            if question:
                questions.append({
                    "id": idx,
                    "question": question
                })

    print(f"📊 共读取 {len(questions)} 个问题")
    return questions


def call_api(question: str, session_id: str = None) -> Dict:
    """调用API获取回答"""
    headers = {
        "Authorization": f"Bearer {API_TOKEN}",
        "Content-Type": "application/json"
    }

    payload = {
        "question": question,
        "session_id": session_id
    }

    try:
        response = requests.post(API_URL, headers=headers, json=payload, timeout=30)

        if response.status_code == 200:
            result = response.json()
            return {
                "success": True,
                "data": result.get("data", {}),
                "error": None
            }
        else:
            return {
                "success": False,
                "data": None,
                "error": f"HTTP {response.status_code}: {response.text}"
            }

    except Exception as e:
        return {
            "success": False,
            "data": None,
            "error": str(e)
        }


def process_batch_questions(questions: List[Dict]) -> List[Dict]:
    """批量处理问题"""
    results = []
    session_id = None  # 保持会话连续性

    total = len(questions)

    for idx, item in enumerate(questions, 1):
        question_id = item["id"]
        question = item["question"]

        print(f"\n[{idx}/{total}] 处理问题 #{question_id}: {question[:50]}...")

        # 调用API
        result = call_api(question, session_id)

        if result["success"]:
            answer_data = result["data"]

            # 标准格式：只保留4个字段
            result_item = {
                "id": question_id,
                "question": question,
                "answer": answer_data.get("answer", ""),
                "ret": answer_data.get("ret", [])
            }
            print(f"  ✅ 成功")
        else:
            # 失败时也保持相同格式
            result_item = {
                "id": question_id,
                "question": question,
                "answer": "",
                "ret": []
            }
            print(f"  ❌ 失败: {result['error']}")

        results.append(result_item)

        # 延迟，避免频繁请求
        if idx < total:
            time.sleep(REQUEST_DELAY)

    return results


def save_results_to_json(results: List[Dict], output_file: str):
    """保存结果到JSON文件（标准格式）"""
    # 直接保存结果列表
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    success_count = sum(1 for r in results if r.get("answer"))
    failed_count = len(results) - success_count

    print(f"\n💾 结果已保存到: {output_file}")
    print(f"   总计: {len(results)} 个问题")
    print(f"   成功: {success_count} 个")
    print(f"   失败: {failed_count} 个")


def main():
    """主函数"""
    print("="*60)
    print("🚀 批量问答处理脚本")
    print("="*60)

    # 步骤1: 读取问题
    print(f"\n📖 步骤1: 读取问题文件 {INPUT_CSV}")
    questions = read_questions_from_csv(INPUT_CSV)

    if not questions:
        print("❌ 没有读取到问题，退出")
        return

    # 步骤2: 批量处理
    print(f"\n⚙️  步骤2: 批量处理 {len(questions)} 个问题")
    results = process_batch_questions(questions)

    # 步骤3: 保存结果
    print(f"\n💾 步骤3: 保存结果到 {OUTPUT_JSON}")
    save_results_to_json(results, OUTPUT_JSON)

    print("\n" + "="*60)
    print("✅ 处理完成！")
    print("="*60)


if __name__ == "__main__":
    main()

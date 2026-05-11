# -*- coding: utf-8 -*-
"""
RAG问答系统Web界面 - 稳定版本
"""
from flask import Flask, render_template, request, jsonify, session
import os
import sys
import io
import traceback
import uuid
import re

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from embeddings import get_embedder, get_llm
from vectorstore import CosmeticVectorStore

app = Flask(__name__)
app.secret_key = os.urandom(24)
app.config['JSON_AS_ASCII'] = False

# Force Flask to accept form data
from werkzeug.formparser import FormDataParser
app.config['MAX_FORM_MEMORY_SIZE'] = 16 * 1024 * 1024

qa_system = None

class ConversationMemory:
    def __init__(self, max_history=10):
        self.history = []
        self.max_history = max_history

    def add(self, question, answer, context=None):
        self.history.append({"question": question, "answer": answer, "context": context})
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]

    def get_history_text(self):
        if not self.history:
            return ""
        text = "\n历史对话:\n"
        for i, item in enumerate(self.history, 1):
            text += f"【轮次{i}】\n用户: {item['question']}\n助手: {item['answer']}\n"
        return text

    def get_conversation_summary(self):
        if not self.history:
            return ""
        summary = "之前的对话:\n"
        for item in self.history:
            summary += f"Q: {item['question']}\n"
            summary += f"A: {item['answer'][:100]}...\n"
        return summary

    def has_history(self):
        return len(self.history) > 0

    def clear(self):
        self.history = []

session_memories = {}

def get_session_memory():
    if 'session_id' not in session:
        session['session_id'] = str(uuid.uuid4())
    sid = session['session_id']
    if sid not in session_memories:
        session_memories[sid] = ConversationMemory(max_history=10)
    return session_memories[sid]

class CosmeticQAWeb:
    def __init__(self):
        self.vector_store = None
        self.embedder = None
        self.llm = None

    def initialize(self, data_path=None, use_mock=False, save_path=None):
        print("=" * 50)
        print("初始化化妆品知识库问答系统...")
        print("=" * 50)

        if data_path is None:
            data_path = config.DATA_FILE
        if save_path is None:
            save_path = "C:/Users/niko/Desktop/cosmetic_qa_data/vectorstore"

        print("\n[1/5] 初始化向量化模型...")
        self.embedder = get_embedder(use_mock=use_mock)
        print("向量化模型初始化完成")

        print("\n[2/5] 初始化LLM推理...")
        self.llm = get_llm(use_mock=use_mock)
        print("LLM推理初始化完成")

        print("\n[3/5] 检查向量存储...")
        if os.path.exists(save_path):
            print("找到已保存的向量存储，尝试加载...")
            try:
                self.vector_store = CosmeticVectorStore(embedding_dim=config.EMBEDDING_DIM)
                self.vector_store.load(save_path, self.embedder)
                print("向量存储加载成功!")
                return True, "向量存储加载成功"
            except Exception as e:
                print(f"加载失败: {e}，将重新创建...")

        print("\n[4/5] 创建向量存储...")
        self.vector_store = CosmeticVectorStore(embedding_dim=config.EMBEDDING_DIM)
        data = self.vector_store.load_csv_data(data_path)
        print(f"成功加载 {len(data)} 条成分数据")

        documents = self.vector_store.prepare_documents(data)
        print(f"成功处理 {len(documents)} 条文档")

        self.vector_store.create_vector_store(documents, self.embedder)
        self.vector_store.save(save_path)

        print("\n[5/5] 系统初始化完成!")
        print("=" * 50)
        return True, "初始化完成"

    def query_with_steps(self, question, top_k=5, memory=None):
        from datetime import datetime

        result = {
            "status": "success",
            "original_question": question,
            "rewritten_question": None,
            "has_history": memory and memory.has_history(),
            "steps": [],
            "results": [],
            "llm_input": "",
            "llm_output": "",
            "answer": ""
        }

        try:
            result["steps"].append({"step": 1, "name": "Query输入", "description": "用户输入的自然语言问题", "data": question})

            query_embedding = self.embedder.embed_query(question)
            result["steps"].append({
                "step": 2,
                "name": "Text Embedding",
                "description": "将Query转换为1536维向量",
                "data": f"向量维度: {len(query_embedding)}",
                "embedding_sample": query_embedding[:10]
            })

            start_time = datetime.now()
            results = self.vector_store.similarity_search(question, top_k=top_k)
            search_time = (datetime.now() - start_time).total_seconds()
            result["steps"].append({
                "step": 3,
                "name": "Similarity Search",
                "description": "在向量库中检索Top-K相似文档",
                "data": f"检索模式: L2距离 | Top-K: {top_k}",
                "metrics": {"search_time": f"{search_time*1000:.2f}ms", "total_docs": len(results)}
            })
            result["results"] = results

            # 构建Context
            context = "参考信息:\n\n"
            for i, r in enumerate(results, 1):
                context += f"【{i}】{r['ingredient_name']}\n"
                context += f"  功效分类: {r['efficacy_category']}\n"
                context += f"  适用肤质: {r['skin_type']}\n"
                context += f"  简要说明: {r['description']}\n"
                if r['precautions']:
                    context += f"  注意事项: {r['precautions']}\n"
                context += "\n"
            result["llm_input"] = context

            result["steps"].append({
                "step": 4,
                "name": "Context构建",
                "description": "将检索结果格式化为LLM输入",
                "data": f"参考文档数: {len(results)}"
            })

            # LLM推理
            llm_start = datetime.now()
            llm_answer = self.llm.generate(prompt=question, context=context)
            llm_time = (datetime.now() - llm_start).total_seconds()
            result["llm_output"] = llm_answer
            result["answer"] = llm_answer

            result["steps"].append({
                "step": 5,
                "name": "LLM推理",
                "description": "通义千问基于Context生成回答",
                "data": f"模型: qwen-turbo | Top-K: {top_k}",
                "metrics": {"llm_time": f"{llm_time*1000:.2f}ms", "model": "qwen-turbo"}
            })

            # 保存到对话历史
            if memory:
                context_summary = "\n".join([r['ingredient_name'] for r in results[:3]])
                memory.add(question, llm_answer, context_summary)

        except Exception as e:
            result["status"] = "error"
            result["error"] = str(e)
            traceback.print_exc()

        return result

# 稳定版HTML模板 - 使用原生表单提交
HTML_TEMPLATE = '''<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>化妆品知识库问答系统</title>
<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
body { font-family: "Microsoft YaHei", "PingFang SC", sans-serif; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); min-height: 100vh; }
.container { max-width: 900px; margin: 0 auto; height: 100vh; display: flex; flex-direction: column; }
.header { padding: 15px 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; display: flex; justify-content: space-between; align-items: center; }
.header h1 { font-size: 1.3em; }
.clear-btn { background: rgba(255,255,255,0.2); border: none; color: white; padding: 8px 16px; border-radius: 20px; cursor: pointer; font-size: 13px; }
.clear-btn:hover { background: rgba(255,255,255,0.3); }
.chat-area { flex: 1; overflow-y: auto; padding: 20px; background: #f5f5f5; }
.message { margin-bottom: 20px; display: flex; flex-direction: column; }
.message.user { align-items: flex-end; }
.message.assistant { align-items: flex-start; }
.message-bubble { max-width: 80%; padding: 12px 16px; border-radius: 18px; font-size: 15px; line-height: 1.5; word-wrap: break-word; }
.message.user .message-bubble { background: #667eea; color: white; border-bottom-right-radius: 4px; }
.message.assistant .message-bubble { background: white; color: #333; border-bottom-left-radius: 4px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }
.message-time { font-size: 11px; color: #999; margin-top: 5px; }
.message.user .message-time { text-align: right; }
.input-area { padding: 15px 20px; background: white; border-top: 1px solid #e9ecef; display: flex; gap: 10px; }
.query-input { flex: 1; padding: 12px 16px; font-size: 15px; border: 2px solid #ddd; border-radius: 25px; outline: none; }
.query-input:focus { border-color: #667eea; }
.submit-btn { padding: 12px 24px; font-size: 15px; background: #667eea; color: white; border: none; border-radius: 25px; cursor: pointer; }
.submit-btn:hover { background: #5a6fd6; }
.submit-btn:disabled { opacity: 0.6; cursor: not-allowed; }
.empty-state { text-align: center; padding: 60px 20px; color: #999; }
</style>
</head>
<body>
<div class="container">
<div class="header">
<h1>化妆品知识库问答</h1>
<button class="clear-btn" id="clearBtn">清空对话</button>
</div>
<div class="chat-area" id="chatArea">
<div class="empty-state" id="emptyState">
<div>有什么可以帮您的？</div>
</div>
<div id="messages"></div>
</div>
<div class="input-area">
<form id="chatForm" method="POST" action="/api/query">
<input type="text" class="query-input" id="queryInput" name="question" placeholder="请输入您的问题...">
<button type="submit" class="submit-btn" id="submitBtn">发送</button>
</form>
</div>
</div>
<script>
var chatArea = document.getElementById("chatArea");
var messagesDiv = document.getElementById("messages");
var emptyState = document.getElementById("emptyState");
var chatForm = document.getElementById("chatForm");
var queryInput = document.getElementById("queryInput");
var submitBtn = document.getElementById("submitBtn");

function formatTime() {
    var now = new Date();
    return (now.getHours() < 10 ? "0" + now.getHours() : now.getHours()) + ":" + (now.getMinutes() < 10 ? "0" + now.getMinutes() : now.getMinutes());
}

function escapeHtml(text) {
    return text.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

function addMessage(content, isUser) {
    emptyState.style.display = "none";
    var div = document.createElement("div");
    div.className = "message " + (isUser ? "user" : "assistant");
    var text = escapeHtml(content).replace(/\n/g, "<br>");
    div.innerHTML = "<div class=\"message-bubble\">" + text + "</div><div class=\"message-time\">" + formatTime() + "</div>";
    messagesDiv.appendChild(div);
    chatArea.scrollTop = chatArea.scrollHeight;
}

chatForm.onsubmit = function(e) {
    e.preventDefault();
    var question = queryInput.value.trim();
    if (!question) return;
    addMessage(question, true);
    queryInput.value = "";
    submitBtn.disabled = true;

    var formData = new FormData();
    formData.append("question", question);

    fetch("/api/query", {
        method: "POST",
        body: formData
    }).then(function(response) { return response.text(); })
    .then(function(html) {
        var parser = new DOMParser();
        var doc = parser.parseFromString(html, "text/html");
        var result = doc.querySelector("#answerContent");
        if (result) {
            addMessage(result.textContent, false);
        } else {
            addMessage("收到响应", false);
        }
        submitBtn.disabled = false;
        queryInput.focus();
    })
    .catch(function(err) {
        addMessage("请求失败: " + err.message, false);
        submitBtn.disabled = false;
    });
};

document.getElementById("clearBtn").onclick = function() {
    fetch("/api/clear", {method: "POST"}).then(function() {
        messagesDiv.innerHTML = "";
        emptyState.style.display = "block";
    });
};
</script>
</body>
</html>'''

@app.route('/')
def index():
    return render_template('index.html')

# API - 支持 JSON 和 FormData
@app.route('/api/query', methods=['POST'])
def query():
    global qa_system
    question = ''
    print(f"Request is_json: {request.is_json}, data: {request.data}")

    # 尝试从JSON获取
    if request.is_json:
        try:
            data = request.get_json(force=True)
            print(f"JSON data: {data}")
            if data:
                question = data.get('question', '')
        except Exception as e:
            print(f"JSON parse error: {e}")

    # 如果没有JSON，从form获取
    if not question:
        question = request.form.get('question', '')

    print(f"Final question: {question}")

    if not question:
        return jsonify({"status": "error", "error": "问题不能为空"})
    if qa_system is None:
        qa_system = CosmeticQAWeb()
        _, msg = qa_system.initialize()
        print(msg)
    memory = get_session_memory()
    result = qa_system.query_with_steps(question, memory=memory)
    return jsonify(result)

@app.route('/api/clear', methods=['POST'])
def clear_session():
    memory = get_session_memory()
    memory.clear()
    return jsonify({"status": "success", "message": "会话已清除"})

@app.route('/api/clear', methods=['GET'])
def clear_session_get():
    memory = get_session_memory()
    memory.clear()
    return jsonify({"status": "success", "message": "会话已清除"})

def main():
    global qa_system
    qa_system = CosmeticQAWeb()
    status, msg = qa_system.initialize()
    print("系统状态: " + msg)
    print("\n启动Web服务: http://127.0.0.1:5003")
    app.run(host='127.0.0.1', port=5003, debug=False)

if __name__ == "__main__":
    main()
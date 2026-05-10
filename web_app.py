# -*- coding: utf-8 -*-
"""
RAG问答系统Web界面 - 展示RAG流程
"""
from flask import Flask, render_template_string, request, jsonify
import os
import sys
import io
import traceback

# 设置标准输出编码
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from embeddings import get_embedder, get_llm
from vectorstore import CosmeticVectorStore

app = Flask(__name__)

# 全局变量
qa_system = None


class CosmeticQAWeb:
    """化妆品问答系统Web版"""

    def __init__(self):
        self.vector_store = None
        self.embedder = None
        self.llm = None

    def initialize(self, data_path: str = None, use_mock: bool = False, save_path: str = None):
        """初始化系统"""
        print("=" * 50)
        print("初始化化妆品知识库问答系统...")
        print("=" * 50)

        if data_path is None:
            data_path = config.DATA_FILE

        # 保存路径
        if save_path is None:
            save_path = "C:/Users/niko/Desktop/cosmetic_qa_data/vectorstore"

        print("\n[1/5] 初始化向量化模型...")
        self.embedder = get_embedder(use_mock=use_mock)
        print("向量化模型初始化完成")

        print("\n[2/5] 初始化LLM推理...")
        self.llm = get_llm(use_mock=use_mock)
        print("LLM推理初始化完成")

        # 尝试加载已保存的向量存储
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

        # 重新创建向量存储
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

    def query_with_steps(self, question: str, top_k: int = 5):
        """带详细步骤的问答 - 展示RAG流程"""
        result = {
            "status": "success",
            "question": question,
            "steps": [],
            "results": [],
            "llm_input": "",
            "llm_output": "",
            "answer": ""
        }

        try:
            # Step 1: 用户Query输入
            result["steps"].append({
                "step": 1,
                "name": "Query输入",
                "description": "用户输入的自然语言问题",
                "data": question
            })

            # Step 2: Text Embedding (向量化)
            query_embedding = self.embedder.embed_query(question)
            result["steps"].append({
                "step": 2,
                "name": "Text Embedding",
                "description": "将Query转换为1536维向量",
                "data": f"向量维度: {len(query_embedding)}",
                "embedding_sample": query_embedding[:10]  # 展示前10维
            })

            # Step 3: Similarity Search (向量检索)
            from datetime import datetime
            start_time = datetime.now()

            results = self.vector_store.similarity_search(question, top_k=top_k)

            search_time = (datetime.now() - start_time).total_seconds()
            result["steps"].append({
                "step": 3,
                "name": "Similarity Search",
                "description": "在向量库中检索Top-K相似文档",
                "data": f"检索模式: L2距离 | Top-K: {top_k}",
                "metrics": {
                    "search_time": f"{search_time*1000:.2f}ms",
                    "total_docs": len(results)
                }
            })
            result["results"] = results

            # Step 4: 构建Context (检索结果格式化)
            context = self._format_context(results)
            result["llm_input"] = context

            result["steps"].append({
                "step": 4,
                "name": "Context构建",
                "description": "将检索结果格式化为LLM输入",
                "data": f"参考文档数: {len(results)}"
            })

            # Step 5: LLM推理生成
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
                "metrics": {
                    "llm_time": f"{llm_time*1000:.2f}ms",
                    "model": "qwen-turbo"
                }
            })

        except Exception as e:
            result["status"] = "error"
            result["error"] = str(e)
            traceback.print_exc()

        return result

    def _format_context(self, results: list) -> str:
        """将检索结果格式化为LLM输入的Context"""
        if not results:
            return "无相关检索结果"

        context = "参考信息:\n\n"
        for i, r in enumerate(results, 1):
            context += f"【{i}】{r['ingredient_name']}\n"
            context += f"  功效分类: {r['efficacy_category']}\n"
            context += f"  适用肤质: {r['skin_type']}\n"
            context += f"  简要说明: {r['description']}\n"
            if r['precautions']:
                context += f"  注意事项: {r['precautions']}\n"
            context += "\n"

        return context

    def _build_answer(self, question: str, result: dict) -> str:
        """构建回答"""
        question_lower = question.lower()
        results = result["results"]

        if not results:
            return "未找到相关内容"

        if any(keyword in question_lower for keyword in ["推荐", "适合", "用什么", "哪种"]):
            answer = "根据您的需求，为您推荐以下成分:\n\n"
            for i, r in enumerate(results, 1):
                answer += f"{i}. {r['ingredient_name']}\n"
                answer += f"   功效: {r['efficacy_category']}\n"
                answer += f"   适用: {r['skin_type']}\n"
                answer += f"   说明: {r['description']}\n"
                if r['precautions']:
                    answer += f"   注意: {r['precautions']}\n"
                answer += "\n"

        elif any(keyword in question_lower for keyword in ["合规", "安全", "有没有", "是否"]):
            answer = "关于成分安全性:\n\n"
            for r in results:
                answer += f"• {r['ingredient_name']}: {r['description']}\n"
                if r['precautions']:
                    answer += f"  注意事项: {r['precautions']}\n"

        elif any(keyword in question_lower for keyword in ["功效", "作用", "有什么用"]):
            answer = "相关成分的功效说明:\n\n"
            for r in results:
                answer += f"• {r['ingredient_name']}: {r['efficacy_category']}\n"
                answer += f"  说明: {r['description']}\n"

        else:
            answer = "为您找到以下相关信息:\n\n"
            for i, r in enumerate(results, 1):
                answer += f"{i}. {r['ingredient_name']}\n"
                answer += f"   功效: {r['efficacy_category']}\n"
                answer += f"   适用肤质: {r['skin_type']}\n"
                answer += f"   说明: {r['description']}\n"
                answer += "\n"

        return answer


# HTML模板
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>化妆品知识库问答系统 - RAG流程展示</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        body {
            font-family: "Microsoft YaHei", "PingFang SC", sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
        }
        .header {
            text-align: center;
            color: white;
            margin-bottom: 30px;
        }
        .header h1 {
            font-size: 2.5em;
            margin-bottom: 10px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }
        .header p {
            font-size: 1.1em;
            opacity: 0.9;
        }
        .card {
            background: white;
            border-radius: 16px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.2);
            overflow: hidden;
        }
        .input-section {
            padding: 30px;
            background: #f8f9fa;
            border-bottom: 2px solid #e9ecef;
        }
        .input-section h2 {
            color: #333;
            font-size: 1.3em;
            margin-bottom: 20px;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        .input-section h2::before {
            content: "💬";
            font-size: 1.2em;
        }
        .query-form {
            display: flex;
            gap: 15px;
        }
        .query-input {
            flex: 1;
            padding: 15px 20px;
            font-size: 16px;
            border: 2px solid #ddd;
            border-radius: 12px;
            outline: none;
            transition: all 0.3s;
        }
        .query-input:focus {
            border-color: #667eea;
            box-shadow: 0 0 0 4px rgba(102, 126, 234, 0.1);
        }
        .submit-btn {
            padding: 15px 40px;
            font-size: 16px;
            font-weight: bold;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 12px;
            cursor: pointer;
            transition: all 0.3s;
        }
        .submit-btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 20px rgba(102, 126, 234, 0.4);
        }
        .submit-btn:disabled {
            opacity: 0.6;
            cursor: not-allowed;
            transform: none;
        }
        .tips {
            margin-top: 15px;
            color: #666;
            font-size: 14px;
        }
        .tips span {
            display: inline-block;
            background: #e9ecef;
            padding: 5px 12px;
            border-radius: 20px;
            margin-right: 10px;
            margin-top: 5px;
        }

        /* RAG流程展示 */
        .rag-flow {
            padding: 30px;
        }
        .rag-flow h2 {
            color: #333;
            font-size: 1.3em;
            margin-bottom: 25px;
            text-align: center;
        }
        .steps-container {
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            gap: 10px;
            flex-wrap: wrap;
        }
        .step {
            flex: 1;
            min-width: 150px;
            background: #f8f9fa;
            border-radius: 12px;
            padding: 15px;
            position: relative;
            opacity: 0.5;
            transition: all 0.5s;
        }
        .step.active {
            opacity: 1;
            transform: translateY(-5px);
            box-shadow: 0 10px 30px rgba(102, 126, 234, 0.2);
        }
        .step.completed {
            opacity: 1;
            background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
        }
        .step-number {
            position: absolute;
            top: -12px;
            left: 20px;
            width: 30px;
            height: 30px;
            background: #667eea;
            color: white;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: bold;
            font-size: 14px;
        }
        .step.completed .step-number {
            background: #28a745;
        }
        .step-title {
            font-size: 16px;
            font-weight: bold;
            color: #333;
            margin-bottom: 10px;
            margin-top: 5px;
        }
        .step-desc {
            font-size: 13px;
            color: #666;
            margin-bottom: 10px;
        }
        .step-data {
            background: #2d3748;
            color: #68d391;
            padding: 10px;
            border-radius: 8px;
            font-family: "Consolas", monospace;
            font-size: 12px;
            word-break: break-all;
        }
        .arrow {
            font-size: 24px;
            color: #667eea;
            align-self: center;
        }

        /* 结果展示 */
        .results-section {
            padding: 30px;
            background: #f8f9fa;
            display: none;
        }
        .results-section.show {
            display: block;
        }
        .results-section h2 {
            color: #333;
            font-size: 1.3em;
            margin-bottom: 20px;
        }
        .result-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
            gap: 20px;
        }
        .result-card {
            background: white;
            border-radius: 12px;
            padding: 20px;
            border-left: 4px solid #667eea;
            transition: all 0.3s;
        }
        .result-card:hover {
            transform: translateX(5px);
            box-shadow: 0 5px 20px rgba(0,0,0,0.1);
        }
        .result-card .rank {
            display: inline-block;
            width: 28px;
            height: 28px;
            background: #667eea;
            color: white;
            border-radius: 50%;
            text-align: center;
            line-height: 28px;
            font-weight: bold;
            margin-right: 10px;
        }
        .result-card .ingredient {
            font-size: 18px;
            font-weight: bold;
            color: #333;
            display: flex;
            align-items: center;
        }
        .result-card .meta {
            margin-top: 10px;
            padding: 10px;
            background: #f8f9fa;
            border-radius: 8px;
        }
        .result-card .meta-label {
            color: #666;
            font-size: 13px;
        }
        .result-card .meta-value {
            color: #333;
            font-weight: 500;
        }
        .similarity {
            display: inline-block;
            background: #28a745;
            color: white;
            padding: 3px 10px;
            border-radius: 20px;
            font-size: 12px;
            margin-left: 10px;
        }

        /* 回答展示 */
        .answer-section {
            padding: 30px;
            display: none;
        }
        .answer-section.show {
            display: block;
        }
        .answer-section h2 {
            color: #333;
            font-size: 1.3em;
            margin-bottom: 20px;
        }
        .answer-content {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 25px;
            border-radius: 12px;
            white-space: pre-wrap;
            line-height: 1.8;
            font-size: 15px;
        }

        /* 加载动画 */
        .loading {
            display: none;
            text-align: center;
            padding: 40px;
        }
        .loading.show {
            display: block;
        }
        .spinner {
            width: 50px;
            height: 50px;
            border: 4px solid #f3f3f3;
            border-top: 4px solid #667eea;
            border-radius: 50%;
            animation: spin 1s linear infinite;
            margin: 0 auto;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }

        /* 错误提示 */
        .error {
            background: #f8d7da;
            color: #721c24;
            padding: 15px 20px;
            border-radius: 8px;
            margin: 20px;
            display: none;
        }
        .error.show {
            display: block;
        }

        .metrics {
            display: flex;
            gap: 20px;
            margin-top: 10px;
        }
        .metric-item {
            background: #fff;
            padding: 8px 15px;
            border-radius: 8px;
            font-size: 13px;
        }
        .metric-label {
            color: #666;
        }
        .metric-value {
            font-weight: bold;
            color: #667eea;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>💄 化妆品知识库问答系统</h1>
            <p>基于RAG的智能问答 | 展示检索增强生成流程</p>
        </div>

        <div class="card">
            <!-- 问题输入 -->
            <div class="input-section">
                <h2>请输入您的问题</h2>
                <form class="query-form" id="queryForm">
                    <input type="text" class="query-input" id="queryInput" placeholder="例如：我想美白，推荐什么成分？" autocomplete="off">
                    <button type="submit" class="submit-btn" id="submitBtn">提问</button>
                </form>
                <div class="tips">
                    <span>我想要美白，推荐什么成分？</span>
                    <span>敏感肌可以用什么成分？</span>
                    <span>烟酰胺安全吗？</span>
                    <span>抗衰老有哪些成分？</span>
                </div>
            </div>

            <!-- 加载中 -->
            <div class="loading" id="loading">
                <div class="spinner"></div>
                <p style="margin-top: 15px; color: #666;">正在检索，请稍候...</p>
            </div>

            <!-- 错误提示 -->
            <div class="error" id="error"></div>

            <!-- RAG流程 -->
            <div class="rag-flow" id="ragFlow">
                <h2>RAG流程展示</h2>
                <div class="steps-container" id="stepsContainer">
                    <div class="step" id="step1">
                        <div class="step-number">1</div>
                        <div class="step-title">Query输入</div>
                        <div class="step-desc">用户自然语言问题</div>
                        <div class="step-data" id="step1Data">等待输入...</div>
                    </div>
                    <div class="arrow">→</div>
                    <div class="step" id="step2">
                        <div class="step-number">2</div>
                        <div class="step-title">Text Embedding</div>
                        <div class="step-desc">向量化处理(1536维)</div>
                        <div class="step-data" id="step2Data">等待处理...</div>
                    </div>
                    <div class="arrow">→</div>
                    <div class="step" id="step3">
                        <div class="step-number">3</div>
                        <div class="step-title">Similarity Search</div>
                        <div class="step-desc">向量库L2距离检索</div>
                        <div class="step-data" id="step3Data">等待检索...</div>
                    </div>
                    <div class="arrow">→</div>
                    <div class="step" id="step4">
                        <div class="step-number">4</div>
                        <div class="step-title">Context构建</div>
                        <div class="step-desc">格式化检索结果</div>
                        <div class="step-data" id="step4Data">等待构建...</div>
                    </div>
                    <div class="arrow">→</div>
                    <div class="step" id="step5">
                        <div class="step-number">5</div>
                        <div class="step-title">LLM推理</div>
                        <div class="step-desc">通义千问生成回答</div>
                        <div class="step-data" id="step5Data">等待生成...</div>
                    </div>
                </div>
            </div>

            <!-- 检索结果 -->
            <div class="results-section" id="resultsSection">
                <h2>📋 检索结果 (Top-K)</h2>
                <div class="result-grid" id="resultGrid"></div>
            </div>

            <!-- 最终回答 -->
            <div class="answer-section" id="answerSection">
                <h2>💡 最终回答 (LLM生成)</h2>
                <div class="answer-content" id="answerContent"></div>
            </div>
        </div>
    </div>

    <script>
        const form = document.getElementById('queryForm');
        const input = document.getElementById('queryInput');
        const submitBtn = document.getElementById('submitBtn');
        const loading = document.getElementById('loading');
        const error = document.getElementById('error');
        const ragFlow = document.getElementById('ragFlow');
        const resultsSection = document.getElementById('resultsSection');
        const answerSection = document.getElementById('answerSection');

        // 重置UI
        function resetUI() {
            loading.classList.remove('show');
            error.classList.remove('show');
            resultsSection.classList.remove('show');
            answerSection.classList.remove('show');
            ragFlow.classList.add('show');

            // 重置步骤 (5步RAG流程)
            for (let i = 1; i <= 5; i++) {
                const step = document.getElementById('step' + i);
                step.classList.remove('active', 'completed');
                document.getElementById('step' + i + 'Data').textContent = '等待...';
            }
        }

        // 更新步骤
        function updateStep(stepNum, data, metrics = null) {
            const step = document.getElementById('step' + stepNum);
            step.classList.add('active');

            const dataEl = document.getElementById('step' + stepNum + 'Data');
            if (typeof data === 'object') {
                dataEl.innerHTML = '';
                for (const [key, value] of Object.entries(data)) {
                    dataEl.innerHTML += `<div>${key}: ${value}</div>`;
                }
            } else {
                dataEl.textContent = data;
            }

            // 完成后标记
            setTimeout(() => {
                step.classList.add('completed');
                step.classList.remove('active');
            }, 500);
        }

        // 处理表单提交
        form.addEventListener('submit', async (e) => {
            e.preventDefault();

            const question = input.value.trim();
            if (!question) {
                return;
            }

            resetUI();
            loading.classList.add('show');
            submitBtn.disabled = true;

            try {
                const response = await fetch('/api/query', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({question: question})
                });

                const result = await response.json();

                loading.classList.remove('show');

                if (result.status === 'error') {
                    error.textContent = '错误: ' + result.error;
                    error.classList.add('show');
                    return;
                }

                // 更新步骤
                if (result.steps) {
                    result.steps.forEach((step, idx) => {
                        const stepNum = step.step;
                        let data = step.data;
                        if (step.embedding_sample) {
                            data = step.embedding_sample.map(v => v.toFixed(4)).join(', ') + '...';
                        }
                        if (step.metrics) {
                            updateStep(stepNum, step.metrics);
                        } else {
                            updateStep(stepNum, data);
                        }
                    });
                }

                // 显示结果
                if (result.results && result.results.length > 0) {
                    resultsSection.classList.add('show');
                    const resultGrid = document.getElementById('resultGrid');
                    resultGrid.innerHTML = result.results.map((r, i) => `
                        <div class="result-card">
                            <div class="ingredient">
                                <span class="rank">${i + 1}</span>
                                ${r.ingredient_name}
                            </div>
                            <div class="meta">
                                <div><span class="meta-label">功效分类:</span> <span class="meta-value">${r.efficacy_category}</span></div>
                                <div><span class="meta-label">适用肤质:</span> <span class="meta-value">${r.skin_type}</span></div>
                                <div><span class="meta-label">说明:</span> ${r.description}</div>
                                ${r.precautions ? `<div><span class="meta-label">注意:</span> ${r.precautions}</div>` : ''}
                            </div>
                        </div>
                    `).join('');

                    // 显示回答
                    answerSection.classList.add('show');
                    document.getElementById('answerContent').textContent = result.answer;
                }

            } catch (err) {
                loading.classList.remove('show');
                error.textContent = '请求错误: ' + err.message;
                error.classList.add('show');
            } finally {
                submitBtn.disabled = false;
            }
        });
    </script>
</body>
</html>
'''


@app.route('/')
def index():
    """首页"""
    return render_template_string(HTML_TEMPLATE)


@app.route('/api/query', methods=['POST'])
def query():
    """问答API"""
    global qa_system

    data = request.get_json()
    question = data.get('question', '')

    if not question:
        return jsonify({"status": "error", "error": "问题不能为空"})

    # 初始化系统（如未初始化）
    if qa_system is None:
        qa_system = CosmeticQAWeb()
        _, msg = qa_system.initialize()
        print(msg)

    # 执行问答
    result = qa_system.query_with_steps(question)

    return jsonify(result)


def main():
    """主函数"""
    global qa_system

    # 初始化系统
    qa_system = CosmeticQAWeb()
    status, msg = qa_system.initialize()
    print(f"系统状态: {msg}")

    # 启动Web服务
    print("\n启动Web服务: http://127.0.0.1:5000")
    app.run(host='127.0.0.1', port=5000, debug=False)


if __name__ == "__main__":
    main()
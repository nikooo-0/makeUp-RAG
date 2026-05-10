# -*- coding: utf-8 -*-
"""
向量Embedding模块 - 使用DashScope text-embedding-v1模型
"""
import os
import dashscope
from dashscope import TextEmbedding
from dashscope import Generation
from typing import List
from langchain.embeddings.base import Embeddings
import config

# 设置API Key
dashscope.api_key = os.getenv("DASHSCOPE_API_KEY", "sk-8a4d28a1c1584406bdbab6f6de8c0533")


class DashScopeEmbeddings(Embeddings):
    """DashScope向量嵌入模型"""

    def __init__(self, model: str = "text-embedding-v1"):
        self.model = model

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """为文档列表生成向量"""
        # DashScope API单次最多支持25条文本
        batch_size = 25
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            response = TextEmbedding.call(
                model=self.model,
                input=batch
            )
            if response.status_code == 200:
                embeddings = [item['embedding'] for item in response.output['embeddings']]
                all_embeddings.extend(embeddings)
            else:
                raise Exception(f"Embedding调用失败: {response.message}")

        return all_embeddings

    def embed_query(self, text: str) -> List[float]:
        """为单个查询生成向量"""
        response = TextEmbedding.call(
            model=self.model,
            input=text
        )
        if response.status_code == 200:
            return response.output['embeddings'][0]['embedding']
        else:
            raise Exception(f"Embedding调用失败: {response.message}")


class DashScopeLLM:
    """DashScope通义千问LLM推理"""

    def __init__(self, model: str = "qwen-turbo"):
        self.model = model

    def generate(self, prompt: str, context: str = None, history: List[tuple] = None) -> str:
        """生成回答

        Args:
            prompt: 用户问题
            context: 检索到的上下文内容
            history: 对话历史 [(user, assistant), ...]

        Returns:
            LLM生成的回复
        """
        # 构建消息列表
        messages = []

        # 系统提示词
        system_prompt = """你是一个专业的化妆品成分顾问，熟悉各类化妆品的功效、适用肤质、安全性等信息。
请根据提供的上下文信息回答用户的问题。如果上下文信息不足以回答，请基于你的知识库给出合理建议。
回答要专业、清晰、简洁。"""

        messages.append({
            "role": "system",
            "content": system_prompt
        })

        # 添加历史对话（如果有）
        if history:
            for user_msg, assistant_msg in history:
                messages.append({"role": "user", "content": user_msg})
                messages.append({"role": "assistant", "content": assistant_msg})

        # 添加上下文
        if context:
            context_prompt = f"""以下是检索到的相关成分信息，请参考这些信息回答用户问题：

{context}

---"""
            messages.append({
                "role": "system",
                "content": context_prompt
            })

        # 添加用户问题
        messages.append({
            "role": "user",
            "content": prompt
        })

        # 调用API
        response = Generation.call(
            model=self.model,
            messages=messages,
            result_format="message"
        )

        if response.status_code == 200:
            return response.output.choices[0].message.content
        else:
            raise Exception(f"LLM调用失败: {response.message}")

    def chat(self, messages: List[dict]) -> str:
        """对话接口

        Args:
            messages: 消息列表 [{"role": "user/assistant/system", "content": "..."}, ...]

        Returns:
            LLM生成的回复
        """
        response = Generation.call(
            model=self.model,
            messages=messages,
            result_format="message"
        )

        if response.status_code == 200:
            return response.output.choices[0].message.content
        else:
            raise Exception(f"LLM调用失败: {response.message}")


def get_embedder(use_mock: bool = False):
    """获取向量化模型实例

    Args:
        use_mock: True使用模拟模型(本地测试), False使用DashScope(生产环境)
    """
    if use_mock:
        print("使用模拟向量化模型（仅用于本地测试）")
        from embeddings_mock import MockEmbeddings
        return MockEmbeddings(dim=config.EMBEDDING_DIM)
    else:
        print("使用DashScope向量化模型")
        return DashScopeEmbeddings()


def get_llm(use_mock: bool = False):
    """获取LLM推理实例

    Args:
        use_mock: True使用模拟模型, False使用通义千问
    """
    if use_mock:
        print("使用模拟LLM（仅用于本地测试）")
        return MockLLM()
    else:
        print("使用DashScope通义千问LLM")
        return DashScopeLLM()
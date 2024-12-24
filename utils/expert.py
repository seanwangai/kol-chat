from utils.quota import check_quota, use_quota, get_quota_display  # 使用新的函数名
from openai import OpenAI
from openai import APIError, APIConnectionError, RateLimitError, APITimeoutError
from openai import AsyncOpenAI  # 改用异步客户端
import logging
import time
import tiktoken
import asyncio
from concurrent.futures import ThreadPoolExecutor
import streamlit as st
import backoff  # 添加到导入列表
from datetime import datetime, timedelta
import sys
import os
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type
)
import random

# 添加项目根目录到 Python 路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 在文件开头添加更详细的日志格式设置
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# X-AI API 配置
client = AsyncOpenAI(  # 改用异步客户端
    api_key=st.secrets.get("XAI_API_KEY", ""),
    base_url=st.secrets.get("XAI_API_BASE", "https://api.x.ai/v1")
)

# 创建线程池
executor = ThreadPoolExecutor(max_workers=10)

# 获取 token 计数器
encoding = tiktoken.get_encoding("cl100k_base")  # GPT-4 使用的编码器

MAX_TOKENS = 131072  # Grok 最大 token 限制
SYSTEM_PROMPT_TEMPLATE = """你是著名文案專家

{knowledge}

"""


def truncate_text(text, max_tokens):
    """截断文本以确保不超过最大 token 限制"""
    tokens = encoding.encode(text)
    if len(tokens) <= max_tokens:
        return text

    # 计算需要保留的中间部分的 token 数量
    middle_tokens = max_tokens

    # 计算开始和结束的位置
    total_tokens = len(tokens)
    remove_tokens = total_tokens - middle_tokens

    # 前面保留更多内容（70%），后面少一些（30%）
    remove_front = int(remove_tokens * 0.3)
    remove_back = remove_tokens - remove_front

    # 保留中间部分的 tokens
    start_idx = remove_front
    end_idx = total_tokens - remove_back

    # 记录截断信息
    logger.info(f"文本被截断：总tokens={total_tokens}, "
                f"保留tokens={middle_tokens}, "
                f"前面删除={remove_front}, "
                f"后面删除={remove_back}")

    # 返回截断后的文本
    return (
        f"...[前面已省略 {remove_front} tokens]...\n\n" +
        encoding.decode(tokens[start_idx:end_idx]) +
        f"\n\n...[后面已省略 {remove_back} tokens]..."
    )


logger = logging.getLogger(__name__)


# 添加请求限制管理
class RateLimiter:
    def __init__(self, requests_per_second=1):
        self.requests_per_second = requests_per_second
        self.last_request_time = None
        self._lock = None

    @property
    def lock(self):
        if self._lock is None:
            self._lock = asyncio.Lock()
        return self._lock

    async def acquire(self):
        async with self.lock:
            now = datetime.now()
            if self.last_request_time is not None:
                time_since_last = (
                    now - self.last_request_time).total_seconds()
                if time_since_last < 1/self.requests_per_second:
                    wait_time = 1/self.requests_per_second - time_since_last
                    await asyncio.sleep(wait_time)
            self.last_request_time = datetime.now()


# 创建全局限速器实例
rate_limiter = RateLimiter(requests_per_second=1)


class Expert:
    def __init__(self, name):
        self.name = name
        self.background = self._load_background()  # 初始化時就讀取背景資料
        self.chat_history = []

    def _load_background(self):
        try:
            with open(f"data/{self.name}/data.txt", "r", encoding="utf-8") as f:
                background = f.read().strip()
                # 記錄載入的背景資料
                logger.info({
                    "action": "load_expert_background",
                    "expert": self.name,
                    "background_length": len(background),
                    "background_preview": background[:200] + "..." if len(background) > 200 else background
                })
                return background
        except Exception as e:
            logger.error(f"Error loading background for {self.name}: {e}")
            return ""

    def get_system_prompt(self):
        return f"""你現在扮演的是{self.name}。以下是你的背景資料：

{self.background}

請依據以上背景來回答問題。請用真誠、專業的態度來回答。
"""


class ExpertAgent:
    def __init__(self, name, knowledge_base, avatar=None):
        self.name = name
        self.original_knowledge = knowledge_base
        self.avatar = avatar or "🤖"
        self.chat_history = []
        self.max_history = 5
        self.history_tokens = 0

        # 創建 Expert 實例��管理背景資料
        self.expert = Expert(name)

        # 計算基本 token
        self.base_tokens = len(encoding.encode(
            self.expert.get_system_prompt()))
        self.tokens_per_turn = 2000

    def count_tokens(self, text):
        """计算文本的 token 数量"""
        return len(encoding.encode(text))

    def adjust_knowledge_base(self):
        """根据对话历史动态调整知识库大小"""
        # 计算可用于知识库的 tokens
        available_tokens = (MAX_TOKENS - self.base_tokens -
                            self.history_tokens - self.tokens_per_turn)

        # 确保至少保留定比例的知识库内容
        min_knowledge_tokens = min(
            80000, available_tokens)  # 提高最小保留量到80k tokens
        max_knowledge_tokens = max(min_knowledge_tokens, available_tokens)

        # 截断知识库内容
        self.knowledge_base = truncate_text(
            self.original_knowledge, max_knowledge_tokens)

        # 记录调整信息
        logger.info(f"知识库调整：历史tokens={self.history_tokens}, "
                    f"可用tokens={available_tokens}, "
                    f"分配给知识库tokens={max_knowledge_tokens}")

    def get_system_prompt(self):
        """获取当前的系统提示词"""
        # 使��� Expert 類的系統提示
        return self.expert.get_system_prompt()

    def update_chat_history(self, question, answer):
        """新对话历史"""
        # 计算新对话的 tokens
        new_qa_tokens = self.count_tokens(f"Q: {question}\nA: {answer}")

        # 如果需要移除旧对话
        while (self.history_tokens + new_qa_tokens > MAX_TOKENS * 0.3 and  # 历史最多占用30%
               self.chat_history):
            # 移除最早的对话并减少 token 计数
            old_q, old_a = self.chat_history.pop(0)
            removed_tokens = self.count_tokens(f"Q: {old_q}\nA: {old_a}")
            self.history_tokens -= removed_tokens
            logger.info(f"移除旧对话，释放 {removed_tokens} tokens")

        # 添加新对话
        self.chat_history.append((question, answer))
        self.history_tokens += new_qa_tokens

        logger.info(f"添加新对话，使用 {new_qa_tokens} tokens，"
                    f"当前历史总计 {self.history_tokens} tokens")

        self.adjust_knowledge_base()  # 重新调整知识库大小

    # 修改装饰器
    @retry(
        retry=retry_if_exception_type(
            (APIConnectionError, APITimeoutError, RateLimitError)),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        stop=stop_after_attempt(3)
    )
    async def get_response(self, prompt):
        """获取专家回应"""
        try:
            logger.info(f"开始处理家 {self.name} 的回应")
            current_model = getattr(
                st.session_state, 'current_model', 'grok-beta')

            # 每次對話都會帶入完整的系統提示
            messages = [
                {
                    "role": "system",
                    "content": self.expert.get_system_prompt()  # 包含專家背景
                }
            ]

            # 然後加入歷史對話
            for old_q, old_a in self.chat_history:
                messages.append({"role": "user", "content": old_q})
                messages.append({"role": "assistant", "content": old_a})

            # 最後加入當前問題
            messages.append({"role": "user", "content": prompt})

            # 記錄請求內容
            logger.info({
                "action": "send_to_ai_api",
                "expert": self.name,
                "model": current_model,
                "request_data": {
                    "messages": messages,
                    "temperature": 0.7,
                    "max_tokens": MAX_TOKENS,
                    "system_prompt": messages[0]["content"],  # 完整記錄系統提示
                    "history_length": len(self.chat_history),
                    "history_tokens": self.history_tokens
                },
                "timestamp": datetime.now().isoformat()
            })

            try:
                await rate_limiter.acquire()
                response = await client.chat.completions.create(
                    model="grok-beta",
                    messages=messages,
                    temperature=0.7
                )
                answer = response.choices[0].message.content
            except Exception as e:
                logger.error(f"Grok API 调用失败: {str(e)}")
                raise

            # 記錄回應內容
            logger.info({
                "action": "receive_from_ai_api",
                "expert": self.name,
                "model": current_model,
                "response_data": {
                    "content_length": len(answer),
                    "content_preview": answer[:200] + "..." if len(answer) > 200 else answer
                },
                "timestamp": datetime.now().isoformat()
            })

            self.update_chat_history(prompt, answer)
            return answer

        except Exception as e:
            logger.error({
                "action": "api_call_error",
                "expert": self.name,
                "model": current_model,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            })
            raise


async def get_responses_async(experts, prompt):
    start_time = time.time()
    logger.info(f"开始并发处理所有专家回应，时间: {start_time}")

    # 獲取當前事件循環
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    async def get_expert_response(expert):
        try:
            response = await expert.get_response(prompt)
            return expert, response, time.time()
        except Exception as e:
            logger.error(f"专家 {expert.name} 处理失败: {str(e)}")
            return expert, f"抱歉，生成回应时出现错误: {str(e)}", time.time()

    # 創建所有任務
    tasks = []
    for expert in experts:
        try:
            # 確保使用同一個事件循環
            task = loop.create_task(get_expert_response(expert))
            tasks.append(task)
        except Exception as e:
            logger.error(f"创建任务失败: {str(e)}")
            continue

    if not tasks:
        logger.error("没有成功创建任何任务")
        return

    try:
        # 使用 gather 而不是 as_completed
        responses = await asyncio.gather(*tasks, return_exceptions=True)

        # 處理每個回應
        for expert, response, finish_time in responses:
            if isinstance(response, Exception):
                logger.error(f"专家 {expert.name} 处理失败: {str(response)}")
                yield expert, f"抱歉，生成回应时出现错误: {str(response)}"
            else:
                logger.info(
                    f"专家 {expert.name} 响应完成，耗时: {finish_time - start_time:.2f}秒")
                yield expert, response

        # 生成總結
        try:
            # 過濾出成功的回應
            valid_responses = [
                (e, r) for e, r, _ in responses if not isinstance(r, Exception)]
            if valid_responses:
                experts_for_summary, responses_for_summary = zip(
                    *valid_responses)
                summary = await generate_summary(prompt, responses_for_summary, experts_for_summary)
                yield st.session_state.titans, summary
            else:
                logger.error("没有成功的回应可以生成总结")
                yield st.session_state.titans, "抱歉，由于所有专家回应都失败，无法生成总结。"
        except Exception as e:
            logger.error(f"生成总结时出错: {str(e)}")
            yield st.session_state.titans, "抱歉，生成总结时出现错误。"

    except Exception as e:
        logger.error(f"处理响应过程中出错: {str(e)}")
        raise


async def generate_summary(prompt, responses, experts):
    """生成总结"""
    logger.info("开始生成总结...")

    # 動態構建專家回應列表
    expert_responses = []
    for expert, response in zip(experts, responses):
        expert_responses.append(f"{expert.name}：{response}")
        logger.info(f"整合 {expert.name} 的回應到總結中")

    # 創建一個專門的整合專家
    summary_expert = Expert("文案整合專家")

    # 構建消息列表，確保系統提示在最前面
    messages = [
        {
            "role": "system",
            "content": summary_expert.get_system_prompt()
        },
        {
            "role": "user",
            "content": f"""結合各文章的優點，改寫出一篇最終文案：

{chr(10).join(expert_responses)}
"""
        }
    ]

    logger.info({
        "action": "generate_summary",
        "system_prompt": messages[0]["content"],
        "user_prompt": messages[1]["content"][:200] + "..."
    })

    try:
        # 使用异步 API 调用
        summary_response = await client.chat.completions.create(
            model="grok-beta",
            messages=messages,
            temperature=0.7
        )
        summary = summary_response.choices[0].message.content
        return summary
    except Exception as e:
        error_msg = "生成总结时出错"
        logger.error(error_msg)
        logger.exception(e)
        return "抱歉，无法生成总结。"

__all__ = ['ExpertAgent', 'get_responses_async', 'generate_summary']

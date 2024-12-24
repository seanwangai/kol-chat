import requests
import logging
import streamlit as st
import datetime
import genai

logger = logging.getLogger(__name__)


class GeminiHandler:
    def __init__(self):
        self.model = genai.GenerativeModel('gemini-pro')
        self.api_key = st.secrets["GOOGLE_API_KEY"]

    def get_response(self, expert, user_input):
        """使用 chat 方式與 Gemini 互動"""
        try:
            # 每次創建新的對話
            chat = self.model.start_chat(history=[])

            # 首先發送系統提示
            system_prompt = expert.get_system_prompt()

            # 記錄系統提示
            logger.info({
                "action": "send_to_gemini_api",
                "expert": expert.name,
                "system_prompt": system_prompt,  # 完整記錄系統提示
                "user_input": user_input
            })

            # 每次對話都發送系統提示
            chat.send_message(system_prompt)

            # 然後發送用戶輸入
            response = chat.send_message(user_input)

            return response.text

        except Exception as e:
            logger.error(f"Gemini API 錯誤: {str(e)}")
            raise

    def generate_gemini_response(self, expert, prompt, model_name="gemini-1.0-pro", max_tokens=1000):
        """使用 REST API 方式與 Gemini 互動"""
        try:
            # 每次都獲取並發送系統提示
            system_prompt = expert.get_system_prompt()

            data = {
                "contents": [{
                    "parts": [
                        {"text": system_prompt},  # 先發送系統提示
                        {"text": prompt}  # 再發送用戶輸入
                    ]
                }]
            }

            logger.info({
                "action": "send_to_gemini_api",
                "expert": expert.name,
                "model": model_name,
                "request_data": {
                    "system_prompt": system_prompt,
                    "user_prompt": prompt
                }
            })

            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent"
            headers = {
                "Content-Type": "application/json",
                "x-goog-api-key": st.secrets["GOOGLE_API_KEY"]
            }

            response = requests.post(url, headers=headers, json=data)
            response.raise_for_status()
            result = response.json()

            if 'candidates' in result:
                response_text = result['candidates'][0]['content']['parts'][0]['text']

                # 記錄回應
                logger.info({
                    "action": "receive_from_gemini_api",
                    "expert": expert.name,
                    "response_preview": response_text[:200] + "..." if len(response_text) > 200 else response_text
                })

                return response_text
            else:
                raise Exception(f"API 錯誤: {result}")
        except Exception as e:
            logger.error(f"Gemini API 錯誤: {str(e)}")
            raise

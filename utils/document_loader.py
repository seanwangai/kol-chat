import os
import PyPDF2
import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup
from .expert import ExpertAgent
import logging
import base64
import requests
from io import BytesIO
import streamlit as st
import datetime

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 检查是否在 Streamlit Cloud 环境运行
IS_CLOUD = st.secrets.get("DEPLOY_ENV") == "cloud"


def download_file(url):
    """从 Dropbox 下载文件"""
    try:
        # 构建正确的 Dropbox 下载链接
        base_url = url.split('?')[0]  # 移除所有参数
        if '/file/' not in base_url:
            base_url = base_url.replace('/scl/fo/', '/scl/fo/file/')
        direct_url = f"{base_url}?dl=1"

        # 記錄請求
        logger.info({
            "action": "download_request",
            "url": direct_url,
            "timestamp": datetime.datetime.now().isoformat()
        })

        response = requests.get(direct_url)
        response.raise_for_status()

        # 記錄回應
        logger.info({
            "action": "download_response",
            "status_code": response.status_code,
            "content_length": len(response.content),
            "timestamp": datetime.datetime.now().isoformat()
        })

        return BytesIO(response.content)
    except Exception as e:
        logger.error({
            "action": "download_error",
            "url": url,
            "error": str(e),
            "timestamp": datetime.datetime.now().isoformat()
        })
        return None


def get_expert_folders():
    """获取专家文件夹列表"""
    try:
        logger.info({
            "action": "get_expert_folders_start",
            "timestamp": datetime.datetime.now().isoformat()
        })
        experts = ["Warren Buffett", "Charlie Munger", "Ray Dalio"]
        logger.info({
            "action": "get_expert_folders_complete",
            "experts": experts,
            "timestamp": datetime.datetime.now().isoformat()
        })
        return experts
    except Exception as e:
        logger.error({
            "action": "get_expert_folders_error",
            "error": str(e),
            "timestamp": datetime.datetime.now().isoformat()
        })
        return None


def read_pdf(file_path):
    """读取 PDF 文件内容"""
    try:
        logger.info({
            "action": "read_pdf_start",
            "file": file_path,
            "timestamp": datetime.datetime.now().isoformat()
        })

        if IS_CLOUD:
            # file_path 已经是 BytesIO 对象
            if not file_path:
                return ''
            reader = PyPDF2.PdfReader(file_path)
        else:
            with open(file_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)

        text = ''
        for page in reader.pages:
            text += page.extract_text() + '\n'

        logger.info({
            "action": "read_pdf_complete",
            "file": file_path,
            "content_length": len(text),
            "content_preview": text[:200] + "..." if len(text) > 200 else text,
            "timestamp": datetime.datetime.now().isoformat()
        })
        return text
    except Exception as e:
        logger.error({
            "action": "read_pdf_error",
            "file": file_path,
            "error": str(e),
            "timestamp": datetime.datetime.now().isoformat()
        })
        return ''


def read_epub(file_path):
    """读取 EPUB 文件内容"""
    try:
        logger.info({
            "action": "read_epub_start",
            "file": file_path,
            "timestamp": datetime.datetime.now().isoformat()
        })

        book = epub.read_epub(file_path)
        text = ''
        for item in book.get_items():
            if item.get_type() == ebooklib.ITEM_DOCUMENT:
                soup = BeautifulSoup(item.get_content(), 'html.parser')
                text += soup.get_text() + '\n'

        logger.info({
            "action": "read_epub_complete",
            "file": file_path,
            "content_length": len(text),
            "content_preview": text[:200] + "..." if len(text) > 200 else text,
            "timestamp": datetime.datetime.now().isoformat()
        })
        return text
    except Exception as e:
        logger.error({
            "action": "read_epub_error",
            "file": file_path,
            "error": str(e),
            "timestamp": datetime.datetime.now().isoformat()
        })
        return ''


def read_txt(file_path):
    """读取 TXT 文件内容"""
    try:
        logger.info({
            "action": "read_txt_start",
            "file": file_path,
            "timestamp": datetime.datetime.now().isoformat()
        })

        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()

        logger.info({
            "action": "read_txt_complete",
            "file": file_path,
            "content_length": len(content),
            "content_preview": content[:200] + "..." if len(content) > 200 else content,
            "timestamp": datetime.datetime.now().isoformat()
        })
        return content
    except Exception as e:
        logger.error({
            "action": "read_txt_error",
            "file": file_path,
            "error": str(e),
            "timestamp": datetime.datetime.now().isoformat()
        })
        return ''


def load_document(file_path):
    """加载单个文档"""
    logger.info({
        "action": "load_document_start",
        "file": file_path,
        "timestamp": datetime.datetime.now().isoformat()
    })

    file_extension = os.path.splitext(file_path)[1].lower()
    if file_extension == '.pdf':
        with open(file_path, 'rb') as f:
            return read_pdf(BytesIO(f.read()))
    elif file_extension == '.epub':
        with open(file_path, 'rb') as f:
            return read_epub(BytesIO(f.read()))
    elif file_extension == '.txt':
        return read_txt(file_path)
    else:
        logger.warning({
            "action": "load_document_unsupported",
            "file": file_path,
            "file_extension": file_extension,
            "timestamp": datetime.datetime.now().isoformat()
        })
        return ''


def load_image_as_base64(image_path):
    """加载图片并转换为 base64"""
    try:
        logger.info({
            "action": "load_image_start",
            "image_path": image_path,
            "timestamp": datetime.datetime.now().isoformat()
        })

        with open(image_path, "rb") as image_file:
            encoded = base64.b64encode(image_file.read()).decode()
            result = f"data:image/png;base64,{encoded}"

        logger.info({
            "action": "load_image_complete",
            "image_path": image_path,
            "encoded_length": len(encoded),
            "timestamp": datetime.datetime.now().isoformat()
        })
        return result
    except Exception as e:
        logger.error({
            "action": "load_image_error",
            "image_path": image_path,
            "error": str(e),
            "timestamp": datetime.datetime.now().isoformat()
        })
        return None


def load_experts():
    """
    从data目录加载专家数据
    """
    logger.info({
        "action": "load_experts_start",
        "timestamp": datetime.datetime.now().isoformat()
    })

    experts = []
    try:
        data_dir = "./data"
        if os.path.exists(data_dir):
            expert_folders = [f for f in os.listdir(
                data_dir) if os.path.isdir(os.path.join(data_dir, f))]

            logger.info({
                "action": "found_expert_folders",
                "folders": expert_folders,
                "timestamp": datetime.datetime.now().isoformat()
            })

            for folder in expert_folders:
                expert_path = os.path.join(data_dir, folder)
                data_file = os.path.join(expert_path, "data.txt")

                logger.info({
                    "action": "process_expert_start",
                    "expert": folder,
                    "expert_path": expert_path,
                    "timestamp": datetime.datetime.now().isoformat()
                })

                # 記錄發送給 API 的數據
                if os.path.exists(data_file):
                    content = read_txt(data_file)
                    logger.info({
                        "action": "load_expert_data",
                        "expert": folder,
                        "file": data_file,
                        "content_length": len(content),
                        "content_preview": content[:200] + "..." if len(content) > 200 else content,
                        "timestamp": datetime.datetime.now().isoformat()
                    })

                # 尝试加载头像
                avatar_path = os.path.join(expert_path, "head.png")
                if os.path.exists(avatar_path):
                    avatar = load_image_as_base64(avatar_path)
                else:
                    avatar = f"data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg'/>"
                    logger.info({
                        "action": "using_default_avatar",
                        "expert": folder,
                        "timestamp": datetime.datetime.now().isoformat()
                    })

                try:
                    logger.info({
                        "action": "create_expert_agent_start",
                        "expert": folder,
                        "timestamp": datetime.datetime.now().isoformat()
                    })

                    expert = ExpertAgent(
                        name=folder,
                        knowledge_base=expert_path,
                        avatar=avatar
                    )
                    experts.append(expert)

                    logger.info({
                        "action": "create_expert_agent_complete",
                        "expert": folder,
                        "timestamp": datetime.datetime.now().isoformat()
                    })
                except Exception as e:
                    logger.error({
                        "action": "create_expert_agent_error",
                        "expert": folder,
                        "error": str(e),
                        "timestamp": datetime.datetime.now().isoformat()
                    })
                    continue

            logger.info({
                "action": "load_experts_complete",
                "expert_count": len(experts),
                "experts": [expert.name for expert in experts],
                "timestamp": datetime.datetime.now().isoformat()
            })

    except Exception as e:
        logger.error({
            "action": "load_experts_error",
            "error": str(e),
            "timestamp": datetime.datetime.now().isoformat()
        })

    return experts


def get_file_type(file_path):
    """获取文件类型"""
    logger.info({
        "action": "get_file_type_start",
        "file_path": file_path,
        "timestamp": datetime.datetime.now().isoformat()
    })

    extension = os.path.splitext(file_path)[1].lower()
    file_type = 'unknown'
    if extension in ['.txt', '.md']:
        file_type = 'text'
    elif extension in ['.pdf']:
        file_type = 'pdf'
    elif extension in ['.doc', '.docx']:
        file_type = 'word'

    logger.info({
        "action": "get_file_type_complete",
        "file_path": file_path,
        "extension": extension,
        "file_type": file_type,
        "timestamp": datetime.datetime.now().isoformat()
    })
    return file_type

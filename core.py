"""共享的模型接口、JSON 解析与本地运行目录。"""
from __future__ import annotations

import json
import ssl
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import certifi
except ImportError:
    certifi = None

RUNS_DIR = Path(__file__).with_name("runs")
_SSL_CONTEXT = ssl.create_default_context(cafile=certifi.where() if certifi else None)


@dataclass
class RunLog:
    step: str
    status: str
    detail: str
    duration_ms: int


class LLMClient:
    """调用兼容 Chat Completions 的服务；模拟客户端由上层流程提供。"""
    def __init__(self, mode: str, base_url: str, model: str, api_key: str):
        self.mode = mode
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key

    def complete(self, system: str, user: str) -> str:
        if self.mode == "模拟模式（不调用真实模型）":
            return ""
        if not self.api_key:
            raise ValueError("真实模型模式需要 API Key。请设置 OPENAI_API_KEY 环境变量后重新启动。")
        url = self.base_url
        if not url.endswith("/chat/completions"):
            url += "/chat/completions"
        payload = json.dumps({
            "model": self.model,
            "messages": [{"role":"system","content":system},{"role":"user","content":user}],
        }).encode("utf-8")
        request = urllib.request.Request(url,data=payload,headers={
            "Content-Type":"application/json","Authorization":f"Bearer {self.api_key}"},method="POST")
        try:
            with urllib.request.urlopen(request,timeout=90,context=_SSL_CONTEXT) as response:
                data=json.loads(response.read().decode("utf-8"))
            return data["choices"][0]["message"]["content"]
        except urllib.error.HTTPError as exc:
            body=exc.read().decode("utf-8",errors="replace")[:500]
            raise RuntimeError(f"模型接口返回 HTTP {exc.code}: {body}") from exc
        except (urllib.error.URLError,TimeoutError) as exc:
            raise RuntimeError(f"无法连接模型接口：{exc}") from exc
        except (KeyError,IndexError,json.JSONDecodeError) as exc:
            raise RuntimeError(f"模型接口返回格式无法识别：{exc}") from exc


def parse_json(text: str) -> dict[str, Any]:
    text=text.strip()
    if text.startswith("```"):
        text=text.strip("`")
        if text.startswith("json"):
            text=text[4:]
    start,end=text.find("{"),text.rfind("}")
    if start<0 or end<start:
        raise ValueError("模型没有返回有效 JSON。")
    return json.loads(text[start:end+1])

"""简历文本提取与预处理（招聘筛选流水线第一阶段）。

职责：
    - 从上传的 PDF / TXT（可选 DOCX）文件提取原始文本。
    - 保留原始文本不裁剪、不补全。
    - 检查空文件、无法解析文件，给出明确错误。
    - 为每份简历分配稳定的 resume_id。

本模块不调用 LLM、不访问网络、不产生任何确定性评分。文本提取是纯函数式
处理，PDF 解析依赖可选第三方库 pypdf（未安装时对 PDF 报清晰错误，TXT 始终可用）。

稳定性约束：相同输入文件内容 -> 相同 resume_id。resume_id 由内容摘要决定，
与文件名、上传顺序无关，保证同一份简历多次上传得到同一标识。
"""

from __future__ import annotations

import hashlib
import io
from dataclasses import dataclass
from pathlib import Path

# 支持的简历文件后缀（小写）。
_SUPPORTED_EXTENSIONS = frozenset({".pdf", ".txt", ".text", ".md", ".docx"})


@dataclass(frozen=True)
class ExtractedResume:
    """一份简历的文本提取结果。

    resume_id    稳定标识，由原文内容摘要决定（与文件名/顺序无关）。
    source_name  原始文件名（仅用于展示与日志，不进入评分）。
    raw_text     提取的原始文本（保留原样，可能含换行）。
    format       文件格式（pdf/txt/docx 等）。
    """
    resume_id: str
    source_name: str
    raw_text: str
    format: str


@dataclass
class ExtractionError(Exception):
    """某份简历提取失败时的错误记录。"""
    source_name: str
    error: str
    format: str = ""
    # 稳定标识在失败时按文件名摘要生成，便于关联日志。
    resume_id: str = ""

    def __post_init__(self) -> None:
        if not self.resume_id:
            self.resume_id = _stable_id_from_name(self.source_name)
        # Exception 的消息字段
        super().__init__(self.error)


def _stable_id_from_name(name: str) -> str:
    return "resume-" + hashlib.sha1(name.encode("utf-8")).hexdigest()[:12]


def stable_resume_id(raw_text: str, source_name: str) -> str:
    """由原文内容摘要生成稳定 resume_id。

    同一份简历（内容一致）无论文件名、上传顺序如何，都得到相同 resume_id；
    不同简历得到不同 resume_id。空文本时退回文件名摘要，保证仍有标识。
    """
    if raw_text.strip():
        digest = hashlib.sha1(raw_text.encode("utf-8")).hexdigest()[:16]
    else:
        digest = hashlib.sha1(source_name.encode("utf-8")).hexdigest()[:12]
    return "resume-" + digest


def _normalize_filename(name: str) -> str:
    return (name or "resume").strip() or "resume"


def detect_format(name: str) -> str:
    """根据文件名后缀判断格式，返回小写后缀（不含点），无法识别时抛错。"""
    suffix = Path(name).suffix.lower()
    if suffix not in _SUPPORTED_EXTENSIONS:
        raise ValueError(f"不支持的文件格式：{suffix or '无后缀'}。支持 {sorted(_SUPPORTED_EXTENSIONS)}。")
    return suffix.lstrip(".")


def extract_text_from_bytes(data: bytes, filename: str) -> str:
    """从文件字节提取文本（PDF / TXT / MD / DOCX）。

    返回提取的原始文本；空文件、无法解析的文件抛 ValueError，给出明确原因。
    不会凭空补全任何内容。
    """
    fmt = detect_format(filename)

    if fmt == "txt" or fmt == "text" or fmt == "md":
        if not data.strip():
            raise ValueError("文件内容为空，无法提取简历文本。")
        try:
            return data.decode("utf-8")
        except UnicodeDecodeError:
            try:
                return data.decode("gb18030")
            except UnicodeDecodeError as exc:
                raise ValueError("文件编码无法识别（非 UTF-8 / GB18030）。") from exc

    if fmt == "pdf":
        if not data.strip():
            raise ValueError("PDF 文件为空，无法提取文本。")
        return _extract_pdf(data, filename)

    if fmt == "docx":
        return _extract_docx(data, filename)

    raise ValueError(f"暂不支持的格式：{fmt}。")


def _extract_pdf(data: bytes, filename: str) -> str:
    """解析 PDF 文本；pypdf 不可用时给出清晰错误。"""
    try:
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover - 依赖缺失路径
        raise ValueError(
            "缺少 PDF 解析依赖 pypdf。请安装：pip install pypdf。"
        ) from exc

    try:
        reader = PdfReader(io.BytesIO(data))
    except Exception as exc:  # noqa: BLE001 - 解析失败需明确报错
        raise ValueError(f"PDF 文件无法解析：{exc}") from exc

    pages: list[str] = []
    for page in reader.pages:
        try:
            text = page.extract_text() or ""
        except Exception:  # noqa: BLE001 - 单页失败不中断整份
            text = ""
        pages.append(text)

    joined = "\n".join(pages)
    if not joined.strip():
        raise ValueError("PDF 未提取到文本（可能是扫描件或图片型 PDF）。")
    return joined


def _extract_docx(data: bytes, filename: str) -> str:
    """解析 DOCX 文本；python-docx 不可用时给出清晰错误。"""
    try:
        from docx import Document
    except ImportError as exc:  # pragma: no cover - 依赖缺失路径
        raise ValueError(
            "缺少 DOCX 解析依赖 python-docx。请安装：pip install python-docx。"
        ) from exc

    try:
        document = Document(io.BytesIO(data))
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"DOCX 文件无法解析：{exc}") from exc

    parts = [p.text for p in document.paragraphs]
    for table in document.tables:
        for row in table.rows:
            parts.append("\t".join(cell.text for cell in row.cells))
    joined = "\n".join(parts)
    if not joined.strip():
        raise ValueError("DOCX 未提取到文本。")
    return joined


def extract_resume(data: bytes, filename: str) -> ExtractedResume:
    """提取单份简历，返回 ExtractedResume 或抛 ExtractionError。"""
    fmt = detect_format(filename)
    raw_text = extract_text_from_bytes(data, filename)
    if not raw_text.strip():
        raise ExtractionError(
            source_name=_normalize_filename(filename),
            error="简历文本为空，无法继续处理。",
            format=fmt,
        )
    return ExtractedResume(
        resume_id=stable_resume_id(raw_text, _normalize_filename(filename)),
        source_name=_normalize_filename(filename),
        raw_text=raw_text,
        format=fmt,
    )


def extract_all_resumes(
    files: list[tuple[bytes, str]],
) -> tuple[list[ExtractedResume], list[ExtractionError]]:
    """批量提取简历，返回 (成功列表, 失败列表)。

    单份失败不影响其它简历；失败原因记录在 ExtractionError，不抛异常中断整批。
    """
    resumes: list[ExtractedResume] = []
    errors: list[ExtractionError] = []
    for data, filename in files:
        try:
            resumes.append(extract_resume(data, filename))
        except ExtractionError as exc:
            errors.append(exc)
        except ValueError as exc:
            errors.append(ExtractionError(
                source_name=_normalize_filename(filename),
                error=str(exc),
                format=_guess_format(filename),
            ))
    return resumes, errors


def _guess_format(name: str) -> str:
    try:
        return detect_format(name)
    except ValueError:
        return ""

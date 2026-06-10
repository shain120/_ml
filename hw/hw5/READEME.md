# HW5: Agent0 — 安全 LLM Agent

透過 Ollama API 呼叫 LLM，具備路徑安全檢查機制的 AI 代理人。

## 檔案
- `agent.py` — 主程式（CLI 互動式 Agent）

## 功能
- Ollama API 串接（支援自訂 model / review model）
- XML 工具呼叫（`run_command`）
- 路徑安全檢查（限制 workspace 內操作）
- Reviewer LLM 審查機制
- 記憶管理（對話歷史 + 關鍵資訊提取）
- 危險指令阻擋（rm -rf /, mkfs, dd 等）

## 環境變數
- `AGENT0_MODEL` — 主要模型（預設 minimax-m2.5:cloud）
- `AGENT0_REVIEW_MODEL` — 審查模型（預設 llama3.1:8b）

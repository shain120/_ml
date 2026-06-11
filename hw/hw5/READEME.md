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

第一層：路徑檢查
第二層：危險指令阻擋
第三層：人類審核與第二個 LLM review

第一層是路徑檢查。
程式會分析 command 裡面的路徑，判斷它是不是在 agent0.py 所在的資料夾裡面。
如果是內部檔案，就可以執行。
如果是外部路徑，例如 ../、/etc/passwd、~/Desktop，就會被標記成需要人工同意。

第二層是危險指令阻擋。
像是：

rm -rf /
mkfs
dd of=/dev/...
shutdown
reboot

這些可能破壞系統的指令會直接擋掉。

第三層是 reviewer LLM。
也就是除了主要 LLM 之外，再用另一個 LLM 當安全審查員。
它會判斷 command 應該 allow、ask 還是 deny。
只要偵測到外部路徑，就一定要問使用者是否同意。

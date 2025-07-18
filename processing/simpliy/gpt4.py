import pandas as pd
import openai
import time

# —— 配置 —— #
openai.api_base = 'xxx'  # 代理网址
openai.api_key = 'xxx'  # 替换为你的 API 密钥

DATA_PATH      = "devign_processed.csv"
OUTPUT_PATH    = "devign_simplified.csv"
SAVE_EVERY     = 10
SLEEP_SECS     = 1
TOKEN_THRESH   = 512
PRE_PRUNE_RATE = 0.7   # 预剪枝保留 70% 最重要 token
MAX_TOKENS     = 2000

CONTROL_KEYWORDS = {"if", "for", "while", "switch", "case", "default", "do", "return", "goto"}
SYMBOLS = set("(){}[];,<>+-=*/%&|^!~")

def count_tokens(code: str) -> int:
    return len(code.split())

11111
def categorize_tokens(code: str):
    sig_end = code.find('{')
    sig_tokens = set(code[:sig_end].split()) if sig_end != -1 else set()
    tokens = code.split()
    scored = []
    for i, tok in enumerate(tokens):
        if tok in sig_tokens:
            score = 1
        elif any(tok.startswith(kw) for kw in CONTROL_KEYWORDS) or ('(' in tok and ')' in tok):
            score = 3
        elif all(ch in SYMBOLS for ch in tok):
            score = 5
        else:
            score = 2
        scored.append((i, tok, score))
    return scored

def pre_prune(code: str, keep_rate: float) -> str:
    scored = categorize_tokens(code)
    keep_n = max(1, int(len(scored) * keep_rate))
    scored_sorted = sorted(scored, key=lambda x: (x[2], x[0]))
    to_keep = set(idx for idx, _, _ in scored_sorted[:keep_n])
    tokens = code.split()
    pruned = [tokens[i] for i in range(len(tokens)) if i in to_keep]
    return " ".join(pruned)

def simplify_whole_code(code_snippet: str) -> str:
    prompt = f"""
You are a vulnerability analysis expert. Aggressively shorten the following C/C++ function initialization code.
- Remove all comments and blank lines.
- Combine consecutive variable declarations into a single line.
- Merge simple field assignments if possible.
- Keep only the absolutely necessary lines to achieve the same behavior.
- Preserve control flow (if/else, switch), input validation, error checks, and memory management.

Code to simplify:
```c/c++
{code_snippet}
```"""
    try:
        resp = openai.Completion.create(
            engine="gpt-4o-mini",
            prompt=prompt,
            max_tokens=MAX_TOKENS,
            temperature=0.3,
            top_p=1.0
        )
        return resp.choices[0].text.strip()
    except Exception as e:
        print(f"[ERROR] LLM 调用失败: {e}")
        return code_snippet

def main():
    df = pd.read_csv(DATA_PATH)
    total = len(df)
    print(f"Total rows: {total}")

    out = pd.DataFrame(columns=['func', 'label'])
    for i, row in df.iterrows():
        orig = row.get('func', '') or ""
        label = row.get('target', '')

        if count_tokens(orig) > TOKEN_THRESH:
            print(f"\n[{i+1}/{total}] 原始 token: {count_tokens(orig)}，开始预剪枝…")
            pruned = pre_prune(orig, PRE_PRUNE_RATE)
            print(f"   预剪枝后 token: {count_tokens(pruned)}，样例如下：")
            print(pruned[:200] + ("..." if len(pruned) > 200 else ""))
            print("开始调用 LLM 进行深度简化…")
            simplified = simplify_whole_code(pruned)
            print("----- Simplified Code Start -----")
            print(simplified)
            print("------ Simplified Code End ------\n")
        else:
            simplified = orig
            print(f"[{i+1}/{total}] token(<={TOKEN_THRESH}) 未触发简化，保留原样。")

        out.loc[len(out)] = [simplified, label]
        time.sleep(SLEEP_SECS)

        if (i+1) % SAVE_EVERY == 0:
            out.to_csv(OUTPUT_PATH, index=False)
            print(f"已保存 {len(out)} 条记录到 {OUTPUT_PATH}\n")

    out.to_csv(OUTPUT_PATH, index=False)
    print(f"Final save: {len(out)} rows to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()

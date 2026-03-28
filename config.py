# ============================================================
#  配置文件 - 请根据实际情况修改
# ============================================================

# PushPlus Token（登录 https://www.pushplus.plus 获取）
PUSHPLUS_TOKEN = "70a87015756f483ab09f70a5ebe5d6ff"

# 选股参数
STOCK_CONFIG = {
    "min_price": 3.0,
    "max_price": 100.0,
    "min_turnover_rate": 1.0,
    "max_turnover_rate": 15.0,
    "min_amount": 5000_0000,
    "min_score": 55,
    "max_results": 20,
    "history_days": 120,

    # ========== 多线程配置 ==========
    "max_workers": 10,          # 并发线程数（建议 5~15）
    "request_delay": 0.02,      # 每次请求间隔(秒)，防封控制
}

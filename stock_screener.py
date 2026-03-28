"""
============================================================
  📈 智能选股系统 v3.0 — 多线程版
  基于多维度技术指标分析，筛选次日上涨概率较大的A股股票
  通过 PushPlus 推送选股结果到微信
============================================================
"""

import akshare as ak
import pandas as pd
import numpy as np
import requests
import datetime
import time
import warnings
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

warnings.filterwarnings('ignore')

from config import PUSHPLUS_TOKEN, STOCK_CONFIG


# ============================================================
#  第一部分：技术指标计算（纯计算，天然线程安全）
# ============================================================

class TechnicalIndicators:
    """技术指标计算器"""

    @staticmethod
    def MA(series: pd.Series, window: int) -> pd.Series:
        return series.rolling(window=window, min_periods=1).mean()

    @staticmethod
    def EMA(series: pd.Series, span: int) -> pd.Series:
        return series.ewm(span=span, adjust=False).mean()

    @staticmethod
    def MACD(close: pd.Series, fast=12, slow=26, signal=9):
        ema_fast = close.ewm(span=fast, adjust=False).mean()
        ema_slow = close.ewm(span=slow, adjust=False).mean()
        dif = ema_fast - ema_slow
        dea = dif.ewm(span=signal, adjust=False).mean()
        macd_hist = 2 * (dif - dea)
        return dif, dea, macd_hist

    @staticmethod
    def KDJ(high: pd.Series, low: pd.Series, close: pd.Series, n=9):
        lowest = low.rolling(window=n, min_periods=1).min()
        highest = high.rolling(window=n, min_periods=1).max()
        rsv = (close - lowest) / (highest - lowest + 1e-10) * 100
        k = rsv.ewm(com=2, adjust=False).mean()
        d = k.ewm(com=2, adjust=False).mean()
        j = 3 * k - 2 * d
        return k, d, j

    @staticmethod
    def RSI(close: pd.Series, period=14) -> pd.Series:
        delta = close.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)
        avg_gain = gain.rolling(window=period, min_periods=1).mean()
        avg_loss = loss.rolling(window=period, min_periods=1).mean()
        rs = avg_gain / (avg_loss + 1e-10)
        return 100 - (100 / (1 + rs))

    @staticmethod
    def BOLL(close: pd.Series, window=20, num_std=2):
        mid = close.rolling(window=window, min_periods=1).mean()
        std = close.rolling(window=window, min_periods=1).std()
        upper = mid + num_std * std
        lower = mid - num_std * std
        return upper, mid, lower

    @staticmethod
    def VOL_RATIO(volume: pd.Series, n=5) -> float:
        avg_vol = volume.iloc[-n - 1:-1].mean()
        if avg_vol == 0:
            return 0
        return volume.iloc[-1] / avg_vol


# ============================================================
#  第二部分：线程安全的进度追踪器
# ============================================================

class ProgressTracker:
    """线程安全的进度条"""

    def __init__(self, total: int):
        self.total = total
        self.done = 0
        self.found = 0
        self.lock = threading.Lock()
        self.start_time = time.time()

    def update(self, code: str, name: str, hit: bool = False):
        with self.lock:
            self.done += 1
            if hit:
                self.found += 1

            pct = self.done / self.total * 100
            bar_len = 30
            filled = int(pct / 100 * bar_len)
            bar = '█' * filled + '░' * (bar_len - filled)
            elapsed = time.time() - self.start_time

            # 估算剩余时间
            if self.done > 0:
                eta = elapsed / self.done * (self.total - self.done)
                eta_str = f"{eta:.0f}s"
            else:
                eta_str = "..."

            sys.stdout.write(
                f'\r   [{bar}] {pct:5.1f}% '
                f'({self.done}/{self.total}) '
                f'命中:{self.found} '
                f'ETA:{eta_str} '
                f'│ {code} {name}      '
            )
            sys.stdout.flush()

    def finish(self):
        elapsed = time.time() - self.start_time
        print(f'\n\n   ✅ 全部分析完成！ 耗时 {elapsed:.1f}s，命中 {self.found} 只')


# ============================================================
#  第三部分：选股策略引擎（多线程版）
# ============================================================

class StockScreener:
    """选股引擎"""

    def __init__(self, config: dict):
        self.config = config
        self.ti = TechnicalIndicators()
        self.selected_stocks = []
        self._results = []           # 线程共享结果列表
        self._results_lock = threading.Lock()  # 结果写入锁

    def fetch_realtime_data(self) -> pd.DataFrame:
        print("📥 正在获取A股实时行情数据...")
        df = ak.stock_zh_a_spot_em()
        print(f"   ✅ 共获取 {len(df)} 只股票行情")
        return df

    def pre_filter(self, df: pd.DataFrame) -> pd.DataFrame:
        cfg = self.config
        mask = (
            (df['最新价'] >= cfg['min_price']) &
            (df['最新价'] <= cfg['max_price']) &
            (df['换手率'] >= cfg['min_turnover_rate']) &
            (df['换手率'] <= cfg['max_turnover_rate']) &
            (df['成交额'] >= cfg['min_amount']) &
            (~df['名称'].str.contains('ST|退|N|C', na=False)) &
            (df['涨跌幅'] > -5) &
            (df['涨跌幅'] < 7)
        )
        filtered = df[mask].copy()
        print(f"🔽 粗筛后剩余: {len(filtered)} 只股票")
        return filtered

    def fetch_history(self, code: str) -> pd.DataFrame:
        try:
            days = self.config['history_days']
            end_date = datetime.datetime.now().strftime('%Y%m%d')
            start_date = (datetime.datetime.now() - datetime.timedelta(days=days)).strftime('%Y%m%d')
            df = ak.stock_zh_a_hist(
                symbol=code, period="daily",
                start_date=start_date, end_date=end_date,
                adjust="qfq"
            )
            return df
        except Exception:
            return None

    def compute_all_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        close = df['收盘']
        high = df['最高']
        low = df['最低']
        volume = df['成交量']

        df['MA5'] = self.ti.MA(close, 5)
        df['MA10'] = self.ti.MA(close, 10)
        df['MA20'] = self.ti.MA(close, 20)
        df['MA60'] = self.ti.MA(close, 60)

        df['DIF'], df['DEA'], df['MACD_HIST'] = self.ti.MACD(close)
        df['K'], df['D'], df['J'] = self.ti.KDJ(high, low, close)
        df['RSI6'] = self.ti.RSI(close, 6)
        df['RSI14'] = self.ti.RSI(close, 14)
        df['BOLL_UP'], df['BOLL_MID'], df['BOLL_DN'] = self.ti.BOLL(close)
        df['VOL_MA5'] = self.ti.MA(volume, 5)
        df['VOL_MA10'] = self.ti.MA(volume, 10)
        return df

    def score_stock(self, df: pd.DataFrame) -> tuple:
        if len(df) < 60:
            return 0, []

        latest = df.iloc[-1]
        prev = df.iloc[-2]
        prev2 = df.iloc[-3]

        score = 0
        reasons = []

        # --------- 均线系统 (25分) ---------
        if latest['MA5'] > latest['MA10'] and prev['MA5'] <= prev['MA10']:
            score += 15
            reasons.append("⭐MA5上穿MA10金叉")
        if latest['MA5'] > latest['MA10'] > latest['MA20']:
            score += 10
            reasons.append("📈均线多头排列")

        # --------- MACD (20分) ---------
        if latest['DIF'] > latest['DEA'] and prev['DIF'] <= prev['DEA']:
            score += 15
            reasons.append("⭐MACD金叉")
        elif latest['MACD_HIST'] > 0 and prev['MACD_HIST'] <= 0:
            score += 10
            reasons.append("🔴MACD柱转红")
        if latest['DIF'] > latest['DEA'] and latest['DIF'] > 0 and prev['DIF'] <= prev['DEA']:
            score += 5
            reasons.append("💪零轴上方金叉")

        # --------- KDJ (15分) ---------
        if latest['K'] > latest['D'] and prev['K'] <= prev['D']:
            if latest['J'] < 80:
                score += 15
                reasons.append("⭐KDJ金叉(未超买)")
            else:
                score += 5
                reasons.append("KDJ金叉(偏高位)")
        if prev['J'] < 20 and latest['J'] > prev['J']:
            score += 10
            reasons.append("🔵KDJ超卖反弹")

        # --------- RSI (10分) ---------
        if 30 <= latest['RSI14'] <= 55:
            score += 5
            reasons.append("RSI适中区间")
        if prev['RSI6'] < 20 and latest['RSI6'] > prev['RSI6']:
            score += 5
            reasons.append("RSI6超卖回升")

        # --------- 量价关系 (15分) ---------
        vol_ratio = self.ti.VOL_RATIO(df['成交量'], 5)
        if 1.5 <= vol_ratio <= 3.0:
            score += 10
            reasons.append(f"📊温和放量(量比{vol_ratio:.1f})")
        elif 1.2 <= vol_ratio < 1.5:
            score += 5
            reasons.append("成交量小幅放大")

        vol_prev_avg = df['成交量'].iloc[-6:-1].mean()
        vol_before = df['成交量'].iloc[-11:-6].mean()
        if vol_before > 0 and vol_prev_avg / vol_before < 0.7 and vol_ratio > 1.3:
            score += 5
            reasons.append("缩量后放量")

        # --------- 布林带 (10分) ---------
        boll_width = (latest['BOLL_UP'] - latest['BOLL_DN']) / (latest['BOLL_MID'] + 1e-10)
        if latest['收盘'] <= latest['BOLL_MID'] and latest['收盘'] >= latest['BOLL_DN']:
            if latest['收盘'] > prev['收盘']:
                score += 10
                reasons.append("💡布林下轨反弹")
        elif latest['收盘'] > latest['BOLL_MID'] and boll_width < 0.15:
            score += 5
            reasons.append("布林收口向上")

        # --------- 价格位置 (5分) ---------
        if latest['收盘'] > latest['MA20'] and prev['收盘'] <= prev['MA20']:
            score += 5
            reasons.append("突破20日均线")
        elif latest['收盘'] > latest['MA20']:
            score += 2

        # --------- K线形态 (加分) ---------
        if (prev2['收盘'] < prev2['开盘'] and
                prev['收盘'] < prev['开盘'] and
                latest['收盘'] > latest['开盘']):
            score += 8
            reasons.append("🔄两阴一阳止跌")

        return score, reasons

    # ============================================================
    #  ⭐ 核心：单只股票分析任务（每个线程执行的单元）
    # ============================================================
    def _analyze_one(self, row: pd.Series, tracker: ProgressTracker):
        """
        分析单只股票（线程任务函数）
        - 获取历史K线 → 计算指标 → 评分 → 写入共享结果
        """
        code = str(row['代码'])
        name = row['名称']

        try:
            # 控制请求频率
            time.sleep(self.config.get('request_delay', 0.02))

            hist = self.fetch_history(code)
            if hist is None or len(hist) < 60:
                tracker.update(code, name)
                return

            hist = self.compute_all_indicators(hist)
            score, reasons = self.score_stock(hist)

            hit = False
            if score >= self.config['min_score']:
                hit = True
                latest = hist.iloc[-1]
                result = {
                    '代码': code,
                    '名称': name,
                    '最新价': float(latest['收盘']),
                    '涨跌幅': float(row['涨跌幅']),
                    '换手率': float(row['换手率']),
                    '成交额(亿)': round(float(row['成交额']) / 1e8, 2),
                    '量比': round(self.ti.VOL_RATIO(hist['成交量'], 5), 2),
                    '得分': score,
                    '选股理由': reasons
                }
                # 🔒 线程安全地写入结果
                with self._results_lock:
                    self._results.append(result)

            tracker.update(code, name, hit)

        except Exception as e:
            tracker.update(code, name)

    # ============================================================
    #  ⭐ 主流程：多线程调度
    # ============================================================
    def run(self) -> list:
        print("\n" + "=" * 60)
        print("     📈  智 能 选 股 系 统  v3.0  (多线程版)")
        print("=" * 60)

        # 1. 获取实时数据
        realtime = self.fetch_realtime_data()

        # 2. 粗筛
        candidates = self.pre_filter(realtime)
        total = len(candidates)
        max_workers = self.config.get('max_workers', 10)

        print(f"\n🔍 开始多线程深度技术分析")
        print(f"   📋 待分析: {total} 只  |  🧵 线程数: {max_workers}")
        print("-" * 60)

        # 3. 初始化
        self._results = []
        tracker = ProgressTracker(total)

        # ============================================================
        #  ⭐⭐⭐ 多线程核心代码 ⭐⭐⭐
        # ============================================================
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for _, row in candidates.iterrows():
                future = executor.submit(self._analyze_one, row, tracker)
                futures.append(future)

            # 等待所有任务完成（异常也不会中断其他线程）
            for future in as_completed(futures):
                try:
                    future.result()  # 触发异常抛出（如果有）
                except Exception:
                    pass

        tracker.finish()

        # 4. 按得分排序
        self._results.sort(key=lambda x: x['得分'], reverse=True)
        self.selected_stocks = self._results[:self.config['max_results']]

        return self.selected_stocks


# ============================================================
#  第四部分：结果格式化与推送（与之前相同）
# ============================================================

class ResultFormatter:
    """结果格式化器"""

    @staticmethod
    def to_console(stocks: list):
        if not stocks:
            print("\n⚠️  今日未筛选出符合条件的股票")
            return

        print("\n" + "=" * 95)
        print(f" {'序号':^4} │ {'代码':^8} │ {'名称':^8} │ {'最新价':^8} │ "
              f"{'涨跌幅':^8} │ {'换手率':^6} │ {'得分':^4} │ 选股理由")
        print("─" * 95)

        for i, s in enumerate(stocks, 1):
            reasons_str = '、'.join(s['选股理由'][:3])
            change = f"+{s['涨跌幅']:.2f}%" if s['涨跌幅'] > 0 else f"{s['涨跌幅']:.2f}%"
            print(f" {i:>4} │ {s['代码']:^8} │ {s['名称']:^8} │ "
                  f"{s['最新价']:>8.2f} │ {change:>8} │ {s['换手率']:>5.2f}% │ "
                  f"{s['得分']:>4} │ {reasons_str}")

        print("=" * 95)

    @staticmethod
    def to_html(stocks: list) -> str:
        now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        today = datetime.datetime.now().strftime('%Y年%m月%d日')

        if not stocks:
            return f"""
            <div style="font-family:'Microsoft YaHei',sans-serif;padding:20px;">
                <h2>📈 {today} 选股报告</h2>
                <p style="color:#999;">{now}</p>
                <p>⚠️ 今日未筛选出符合条件的股票，建议观望。</p>
            </div>"""

        # 生成每只股票的卡片
        cards_html = ""
        for i, s in enumerate(stocks, 1):
            # 涨跌幅颜色
            if s['涨跌幅'] > 0:
                change_text = f"+{s['涨跌幅']:.2f}%"
                change_color = "#e74c3c"
            elif s['涨跌幅'] < 0:
                change_text = f"{s['涨跌幅']:.2f}%"
                change_color = "#27ae60"
            else:
                change_text = "0.00%"
                change_color = "#999"

            # 得分标签
            if s['得分'] >= 80:
                score_tag = f'<span style="background:#e74c3c;color:#fff;padding:2px 8px;border-radius:4px;font-size:12px;">🔥 {s["得分"]}分</span>'
            elif s['得分'] >= 65:
                score_tag = f'<span style="background:#e67e22;color:#fff;padding:2px 8px;border-radius:4px;font-size:12px;">⭐ {s["得分"]}分</span>'
            else:
                score_tag = f'<span style="background:#2196F3;color:#fff;padding:2px 8px;border-radius:4px;font-size:12px;">✅ {s["得分"]}分</span>'

            # 理由拼接
            reasons_str = " | ".join(s['选股理由'])

            cards_html += f"""
            <div style="background:#fff;border-radius:8px;padding:15px;margin-bottom:10px;
                        border-left:4px solid {'#e74c3c' if s['得分']>=70 else '#2196F3'};">
                <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;">
                    <div>
                        <span style="font-size:16px;font-weight:bold;color:#333;">{s['名称']}</span>
                        <span style="color:#999;font-size:13px;margin-left:6px;">{s['代码']}</span>
                    </div>
                    <div>{score_tag}</div>
                </div>
                <div style="display:flex;gap:20px;font-size:13px;color:#666;margin-bottom:8px;">
                    <span>现价 <b style="color:#333;">¥{s['最新价']:.2f}</b></span>
                    <span>涨跌 <b style="color:{change_color};">{change_text}</b></span>
                    <span>换手 <b style="color:#333;">{s['换手率']:.2f}%</b></span>
                </div>
                <div style="font-size:12px;color:#888;line-height:1.6;">
                    💡 {reasons_str}
                </div>
            </div>"""

        html = f"""
        <div style="font-family:'Microsoft YaHei',sans-serif;max-width:600px;margin:0 auto;padding:15px;">
            <h2 style="text-align:center;margin:0 0 5px 0;">📈 智能选股报告</h2>
            <p style="text-align:center;color:#999;font-size:13px;margin:0 0 15px 0;">
                {today} · 共 {len(stocks)} 只 · {now}
            </p>
            {cards_html}
            <p style="font-size:11px;color:#bbb;text-align:center;margin-top:15px;">
                ⚠️ 仅供参考，不构成投资建议，股市有风险
            </p>
        </div>"""
        return html


class PushPlusNotifier:
    """PushPlus 推送器"""

    API_URL = "http://www.pushplus.plus/send"

    def __init__(self, token: str):
        self.token = token

    def send(self, title: str, content: str, template: str = "html") -> bool:
        if self.token == "your_pushplus_token_here":
            print("\n⚠️  请先在 config.py 中配置你的 PushPlus Token！")
            print("   👉 访问 https://www.pushplus.plus 免费获取")
            return False

        payload = {
            "token": self.token,
            "title": title,
            "content": content,
            "template": template
        }

        try:
            print("\n📤 正在推送到手机...")
            resp = requests.post(self.API_URL, json=payload, timeout=30)
            result = resp.json()

            if result.get('code') == 200:
                print("   ✅ 推送成功！请查看微信消息。")
                return True
            else:
                print(f"   ❌ 推送失败: {result.get('msg', '未知错误')}")
                return False
        except requests.exceptions.Timeout:
            print("   ❌ 推送超时，请检查网络连接")
            return False
        except Exception as e:
            print(f"   ❌ 推送异常: {e}")
            return False


# ============================================================
#  第五部分：主函数入口
# ============================================================

def main():
    start_time = time.time()

    # 1. 选股
    screener = StockScreener(STOCK_CONFIG)
    selected = screener.run()

    # 2. 控制台输出
    formatter = ResultFormatter()
    formatter.to_console(selected)

    # 3. 生成HTML + 推送
    html_report = formatter.to_html(selected)

    today_str = datetime.datetime.now().strftime('%m/%d')
    count = len(selected)
    title = f"📈 {today_str} 选股报告 | 发现{count}只潜力股"

    notifier = PushPlusNotifier(PUSHPLUS_TOKEN)
    notifier.send(title, html_report)

    # 4. 本地备份
    backup_file = f"report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
    with open(backup_file, 'w', encoding='utf-8') as f:
        f.write(html_report)
    print(f"\n💾 报告已保存至: {backup_file}")

    elapsed = time.time() - start_time
    print(f"⏱️  总耗时: {elapsed:.1f}秒")
    print("\n" + "=" * 60)
    print("         程序运行结束，祝投资顺利！🎉")
    print("=" * 60)


if __name__ == "__main__":
    main()

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
import io

# 配置页面
st.set_page_config(
    page_title="股票回测工具（增强版）",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义 CSS
st.markdown("""
    <style>
    .main {
        padding: 0;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 5px;
        border-left: 4px solid #1a1a1a;
    }
    h1 {
        color: #1a1a1a;
        margin-bottom: 5px;
    }
    h2 {
        color: #666;
        font-size: 14px;
        text-transform: uppercase;
        font-weight: 600;
        margin-top: 20px;
        margin-bottom: 15px;
        padding-bottom: 10px;
        border-bottom: 1px solid #e5e5e5;
    }
    </style>
    """, unsafe_allow_html=True)

# 常量
FEE_RATE = 0.0001
UNIT_COUNT = 5
STRATEGIES = {
    'custom': '自定义表达式（原始）',
    'ma5_20': '5日均线上穿20日均线（MA5/MA20）',
    'macd': 'MACD 金叉/死叉',
    'rsi': 'RSI 超买超卖 (14)',
    'macd_rsi': 'MACD + RSI 组合',
    'bollinger_kdj': '布林带超跌反弹（Bollinger + KDJ）',
    'kdj_volume': 'KDJ 钝化+量能博弈',
    'wr': '威廉指标 WR（14）',
    'dmi': 'DMI 趋势确认突破',
}

# ============================================================================
# 指标计算函数
# ============================================================================

def ema(prices, period):
    """计算指数移动平均线"""
    result = []
    alpha = 2 / (period + 1)
    for i, price in enumerate(prices):
        if i == 0:
            result.append(price)
        else:
            result.append(alpha * price + (1 - alpha) * result[i - 1])
    return result

def compute_indicators(df):
    """计算所有技术指标"""
    df = df.copy()
    
    # MA5, MA20
    df['ma5'] = df['close'].rolling(5).mean()
    df['ma20'] = df['close'].rolling(20).mean()
    
    # MACD
    closes = df['close'].values
    ema12 = ema(closes.tolist(), 12)
    ema26 = ema(closes.tolist(), 26)
    macd_line = np.array(ema12) - np.array(ema26)
    signal_line = ema(macd_line.tolist(), 9)
    df['macdLine'] = macd_line
    df['signalLine'] = signal_line
    
    # RSI (14)
    delta = df['close'].diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).rolling(14).mean()
    avg_loss = pd.Series(loss).rolling(14).mean()
    rs = avg_gain / avg_loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # Bollinger Bands (20, 2)
    bb_middle = df['close'].rolling(20).mean()
    bb_std = df['close'].rolling(20).std()
    df['bb_middle'] = bb_middle
    df['bb_upper'] = bb_middle + 2 * bb_std
    df['bb_lower'] = bb_middle - 2 * bb_std
    df['bb_dev'] = (df['close'] - bb_middle) / bb_std
    
    # KDJ (9, 3, 3)
    high_9 = df['high'].rolling(9).max()
    low_9 = df['low'].rolling(9).min()
    rsv = ((df['close'] - low_9) / (high_9 - low_9)) * 100
    
    k = [50]
    d = [50]
    for i in range(1, len(df)):
        if pd.notna(rsv.iloc[i]):
            k_val = (2/3) * k[-1] + (1/3) * rsv.iloc[i]
            k.append(k_val)
            d_val = (2/3) * d[-1] + (1/3) * k_val
            d.append(d_val)
        else:
            k.append(np.nan)
            d.append(np.nan)
    
    df['k'] = k
    df['d'] = d
    df['j'] = 3 * pd.Series(k) - 2 * pd.Series(d)
    
    # Williams %R (14)
    high_14 = df['high'].rolling(14).max()
    low_14 = df['low'].rolling(14).min()
    df['wr'] = ((high_14 - df['close']) / (high_14 - low_14)) * -100
    
    # DMI (14)
    high_diff = df['high'].diff()
    low_diff = -df['low'].diff()
    
    plus_dm = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0)
    minus_dm = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0)
    
    tr = np.maximum(
        df['high'] - df['low'],
        np.maximum(
            np.abs(df['high'] - df['close'].shift()),
            np.abs(df['low'] - df['close'].shift())
        )
    )
    
    def wilder_smooth(values, period):
        smoothed = [np.nan] * len(values)
        first_sum = np.nansum(values[:period])
        smoothed[period-1] = first_sum
        for i in range(period, len(values)):
            smoothed[i] = smoothed[i-1] - (smoothed[i-1] / period) + values[i]
        return smoothed
    
    atr = wilder_smooth(tr, 14)
    plus_di_smooth = wilder_smooth(plus_dm, 14)
    minus_di_smooth = wilder_smooth(minus_dm, 14)
    
    df['plusDI'] = (np.array(plus_di_smooth) / np.array(atr)) * 100
    df['minusDI'] = (np.array(minus_di_smooth) / np.array(atr)) * 100
    
    dx = np.abs(df['plusDI'] - df['minusDI']) / (df['plusDI'] + df['minusDI']) * 100
    df['adx'] = pd.Series(wilder_smooth(dx, 14))
    
    return df

def evaluate_condition(condition, row):
    """评估自定义条件"""
    try:
        open = row['open']
        close = row['close']
        high = row['high']
        low = row['low']
        volume = row['volume']
        ma5 = row.get('ma5', 0) or 0
        ma20 = row.get('ma20', 0) or 0
        macdLine = row.get('macdLine', 0) or 0
        signalLine = row.get('signalLine', 0) or 0
        rsi = row.get('rsi', 0) or 0
        k = row.get('k', 0) or 0
        d = row.get('d', 0) or 0
        j = row.get('j', 0) or 0
        bb_upper = row.get('bb_upper', 0) or 0
        bb_middle = row.get('bb_middle', 0) or 0
        bb_lower = row.get('bb_lower', 0) or 0
        wr = row.get('wr', 0) or 0
        plusDI = row.get('plusDI', 0) or 0
        minusDI = row.get('minusDI', 0) or 0
        adx = row.get('adx', 0) or 0
        
        return eval(condition)
    except Exception as e:
        st.error(f"表达式错误: {e}")
        return False

def strategy_decision(strategy, df, idx, state, buy_cond="", sell_cond=""):
    """根据策略做出交易决定"""
    if idx >= len(df):
        return 0, 0, None
    
    day = df.iloc[idx]
    prev = df.iloc[idx-1] if idx > 0 else None
    remaining_units = UNIT_COUNT - state['unitsHeld']
    
    if strategy == 'custom':
        buy = 1 if evaluate_condition(buy_cond, day) else 0
        sell = state['unitsHeld'] if evaluate_condition(sell_cond, day) else 0
        return buy, sell, 'custom'
    
    if strategy == 'ma5_20':
        if prev is None or pd.isna(day['ma5']) or pd.isna(day['ma20']):
            return 0, 0, None
        if prev['ma5'] <= prev['ma20'] < day['ma5']:
            return 1, 0, 'ma5_20_buy'
        if prev['ma5'] >= prev['ma20'] > day['ma5']:
            return 0, state['unitsHeld'], 'ma5_20_sell'
        return 0, 0, None
    
    if strategy == 'macd':
        if prev is None or pd.isna(day['macdLine']) or pd.isna(day['signalLine']):
            return 0, 0, None
        if prev['macdLine'] <= prev['signalLine'] < day['macdLine']:
            return 1, 0, 'macd_buy'
        if prev['macdLine'] >= prev['signalLine'] > day['macdLine']:
            return 0, state['unitsHeld'], 'macd_sell'
        return 0, 0, None
    
    if strategy == 'rsi':
        if pd.isna(day['rsi']):
            return 0, 0, None
        if day['rsi'] < 30:
            return 1, 0, 'rsi_buy'
        if day['rsi'] > 70:
            return 0, state['unitsHeld'], 'rsi_sell'
        return 0, 0, None
    
    if strategy == 'macd_rsi':
        macd_buy, _, _ = strategy_decision('macd', df, idx, state)
        rsi_buy = not pd.isna(day['rsi']) and day['rsi'] < 40
        if macd_buy > 0 and rsi_buy:
            return 1, 0, 'macd_rsi'
        macd_sell, _, _ = strategy_decision('macd', df, idx, state)
        rsi_sell = not pd.isna(day['rsi']) and day['rsi'] > 60
        if macd_sell > 0 or rsi_sell:
            return 0, state['unitsHeld'], 'macd_rsi_sell'
        return 0, 0, None
    
    if strategy == 'bollinger_kdj':
        buy_count = 0
        if (not pd.isna(day['bb_lower']) and day['close'] < day['bb_lower'] and
            not pd.isna(day['j']) and day['j'] < -10):
            buy_count = 1
            if (prev is not None and not pd.isna(prev['bb_lower']) and prev['close'] < prev['bb_lower'] and
                not pd.isna(day['bb_dev']) and not pd.isna(prev['bb_dev']) and
                day['bb_dev'] < prev['bb_dev']):
                buy_count = 2
        
        sell_count = 0
        if (not pd.isna(day['bb_middle']) and day['close'] >= day['bb_middle'] and
            prev is not None and (pd.isna(day['volume']) or pd.isna(prev['volume']) or
            day['volume'] <= prev['volume'] * 1.2)):
            bottom_count = sum(1 for u in state['unitRecords'] if u.get('tag') == 'bottom')
            sell_count = bottom_count
        
        buy_count = max(0, min(buy_count, remaining_units))
        sell_count = max(0, min(sell_count, state['unitsHeld']))
        return buy_count, sell_count, 'bollinger_kdj'
    
    if strategy == 'kdj_volume':
        buy_count = 0
        lookback = 60
        start = max(0, idx - lookback + 1)
        slice_data = df.iloc[start:idx+1]
        max_vol = slice_data['volume'].max()
        vol_threshold = max_vol / 3
        
        last_3 = df.iloc[max(0, idx-2):idx+1]
        k_flat = False
        if len(last_3) == 3 and last_3['k'].notna().all():
            diffs = [last_3.iloc[1]['k'] - last_3.iloc[0]['k'],
                     last_3.iloc[2]['k'] - last_3.iloc[1]['k']]
            k_flat = all(d >= -2 for d in diffs)
        
        if (not pd.isna(day['k']) and day['k'] < 20 and k_flat and
            day['volume'] <= vol_threshold and prev is not None and
            day['volume'] > prev['volume'] * 1.5 and day['close'] > prev['close']):
            buy_count = 2 if remaining_units >= 2 else (1 if remaining_units >= 1 else 0)
        
        sell_count = 0
        if (not pd.isna(day['k']) and day['k'] >= 80 and prev is not None and
            not pd.isna(prev['j']) and not pd.isna(day['j']) and day['j'] < prev['j']):
            sell_count = state['unitsHeld']
        
        return buy_count, sell_count, 'kdj_volume'
    
    if strategy == 'wr':
        buy_count = 0
        sell_count = 0
        
        if prev is not None and not pd.isna(prev['wr']) and not pd.isna(day['wr']):
            if prev['wr'] < -80 and day['wr'] >= -80:
                buy_count = 1
            if prev['wr'] < -20 and day['wr'] >= -20:
                buy_count = max(buy_count, min(remaining_units, 2))
            if prev['wr'] > -20 and day['wr'] < -20 and day['wr'] < prev['wr']:
                sell_count = state['unitsHeld']
        
        return buy_count, sell_count, 'wr'
    
    if strategy == 'dmi':
        buy_count = 0
        sell_count = 0
        
        if (prev is not None and not pd.isna(day['plusDI']) and not pd.isna(day['minusDI']) and
            not pd.isna(day['adx']) and not pd.isna(prev['adx'])):
            if (prev['plusDI'] <= prev['minusDI'] and day['plusDI'] > day['minusDI'] and
                day['adx'] > 25 and day['adx'] > prev['adx']):
                buy_count = 2 if remaining_units >= 2 else (1 if remaining_units >= 1 else 0)
            if prev['minusDI'] <= prev['plusDI'] and day['minusDI'] > day['plusDI']:
                sell_count = state['unitsHeld']
            if day['adx'] > 50 and day['adx'] < prev['adx']:
                sell_count = state['unitsHeld']
        
        return buy_count, sell_count, 'dmi'
    
    return 0, 0, None

def run_backtest(data, initial_capital, start_date, end_date, strategy, buy_cond="", sell_cond=""):
    """运行回测"""
    data['date'] = pd.to_datetime(data['date'])
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    
    # 按股票代码计算指标
    data_by_code = {}
    for code in data['code'].unique():
        df = data[data['code'] == code].sort_values('date').reset_index(drop=True)
        df = compute_indicators(df)
        data_by_code[code] = df
    
    # 回测状态
    cash = initial_capital
    unit_value = initial_capital / UNIT_COUNT
    units_held = 0
    unit_records = []
    holding_code = None
    trade_count = 0
    trade_log = []
    
    asset_dates = []
    asset_values = []
    return_values = []
    
    # 获取时间范围内的所有日期
    all_dates = sorted(data[(data['date'] >= start) & (data['date'] <= end)]['date'].unique())
    
    for current_date in all_dates:
        current_date_str = current_date.strftime('%Y-%m-%d')
        
        # 计算市场价值
        market_value = 0
        if units_held > 0 and holding_code:
            df = data_by_code[holding_code]
            price_data = df[df['date'] == current_date]
            if not price_data.empty:
                price = price_data.iloc[0]['close']
                market_value = sum(u['shares'] * price for u in unit_records)
            else:
                last_price = df[df['date'] <= current_date].iloc[-1]['close']
                market_value = sum(u['shares'] * last_price for u in unit_records)
        
        total_asset = cash + market_value
        asset_dates.append(current_date_str)
        asset_values.append(total_asset)
        return_values.append(((total_asset - initial_capital) / initial_capital * 100))
        
        # 交易逻辑
        if units_held == 0:
            for code in data_by_code.keys():
                df = data_by_code[code]
                code_data = df[df['date'] == current_date]
                if code_data.empty:
                    continue
                
                idx = df[df['date'] == current_date].index[0]
                state = {
                    'unitsHeld': units_held,
                    'unitRecords': unit_records,
                    'unitValue': unit_value,
                    'cash': cash
                }
                
                buy_units, sell_units, reason = strategy_decision(strategy, df, idx, state, buy_cond, sell_cond)
                
                if buy_units > 0:
                    to_buy = min(buy_units, UNIT_COUNT - units_held)
                    for _ in range(to_buy):
                        price = code_data.iloc[0]['close']
                        max_shares = int(unit_value / (price * (1 + FEE_RATE)))
                        
                        if max_shares <= 0:
                            break
                        
                        cost = max_shares * price
                        fee = cost * FEE_RATE
                        
                        if cash >= cost + fee:
                            cash -= (cost + fee)
                            unit_records.append({
                                'shares': max_shares,
                                'price': price,
                                'date': current_date_str,
                                'tag': 'bottom' if (reason == 'bollinger_kdj' and price < code_data.iloc[0].get('bb_lower', float('inf'))) else 'normal'
                            })
                            units_held += 1
                            trade_count += 1
                            trade_log.append({
                                'date': current_date_str,
                                'code': code,
                                'action': '买入',
                                'price': price,
                                'units': 1,
                                'shares': max_shares,
                                'fee': fee,
                                'cash': cash
                            })
                            holding_code = code
                        else:
                            break
                    break
        else:
            df = data_by_code[holding_code]
            code_data = df[df['date'] == current_date]
            
            if not code_data.empty:
                idx = df[df['date'] == current_date].index[0]
                state = {
                    'unitsHeld': units_held,
                    'unitRecords': unit_records,
                    'unitValue': unit_value,
                    'cash': cash
                }
                
                buy_units, sell_units, reason = strategy_decision(strategy, df, idx, state, buy_cond, sell_cond)
                
                # 先卖出
                if sell_units > 0:
                    sell_count = min(sell_units, units_held)
                    sell_indices = []
                    
                    for i in range(len(unit_records) - 1, -1, -1):
                        if unit_records[i].get('tag') == 'bottom' and sell_count > 0:
                            sell_indices.append(i)
                            sell_count -= 1
                    
                    for i in range(len(unit_records) - 1, -1, -1):
                        if i not in sell_indices and sell_count > 0:
                            sell_indices.append(i)
                            sell_count -= 1
                    
                    for idx_sell in sell_indices:
                        rec = unit_records[idx_sell]
                        price = code_data.iloc[0]['close']
                        revenue = rec['shares'] * price
                        fee = revenue * FEE_RATE
                        cash += revenue - fee
                        trade_count += 1
                        trade_log.append({
                            'date': current_date_str,
                            'code': holding_code,
                            'action': '卖出',
                            'price': price,
                            'units': 1,
                            'shares': 0,
                            'fee': fee,
                            'cash': cash
                        })
                    
                    unit_records = [u for i, u in enumerate(unit_records) if i not in sell_indices]
                    units_held = len(unit_records)
                    if units_held == 0:
                        holding_code = None
                
                # 再买入
                if buy_units > 0:
                    to_buy = min(buy_units, UNIT_COUNT - units_held)
                    for _ in range(to_buy):
                        price = code_data.iloc[0]['close']
                        max_shares = int(unit_value / (price * (1 + FEE_RATE)))
                        
                        if max_shares <= 0:
                            break
                        
                        cost = max_shares * price
                        fee = cost * FEE_RATE
                        
                        if cash >= cost + fee:
                            cash -= (cost + fee)
                            unit_records.append({
                                'shares': max_shares,
                                'price': price,
                                'date': current_date_str,
                                'tag': 'bottom' if (reason == 'bollinger_kdj' and price < code_data.iloc[0].get('bb_lower', float('inf'))) else 'normal'
                            })
                            units_held += 1
                            trade_count += 1
                            trade_log.append({
                                'date': current_date_str,
                                'code': holding_code,
                                'action': '加仓',
                                'price': price,
                                'units': 1,
                                'shares': max_shares,
                                'fee': fee,
                                'cash': cash
                            })
                        else:
                            break
    
    # 平仓
    if units_held > 0 and holding_code:
        df = data_by_code[holding_code]
        last_date = df[df['date'] <= end].iloc[-1]
        for rec in unit_records:
            revenue = rec['shares'] * last_date['close']
            fee = revenue * FEE_RATE
            cash += revenue - fee
            trade_count += 1
            trade_log.append({
                'date': last_date['date'].strftime('%Y-%m-%d'),
                'code': holding_code,
                'action': '平仓',
                'price': last_date['close'],
                'units': 1,
                'shares': 0,
                'fee': fee,
                'cash': cash
            })
    
    final_capital = cash
    total_return = ((final_capital - initial_capital) / initial_capital * 100)
    
    return {
        'final_capital': final_capital,
        'total_return': total_return,
        'trade_count': trade_count,
        'trade_log': pd.DataFrame(trade_log),
        'asset_dates': asset_dates,
        'asset_values': asset_values,
        'return_values': return_values
    }

# ============================================================================
# Streamlit UI
# ============================================================================

st.title("📈 股票回测工具（增强版）")
st.markdown("支持分仓（5份）、多种策略、绘制资产/收益曲线")

# 侧边栏
with st.sidebar:
    st.header("ℹ️ 应用信息")
    st.markdown("""
    - **版本**: 1.0.0
    - **策略数量**: 9 种
    - **分仓**: 5 份独立管理
    - **手续费**: 万分之一
    - **注意**: 所有计算在浏览器本地进行，数据不会上传
    """)

# 数据导入
st.header("📁 数据导入")
col1, col2 = st.columns([3, 1])
with col1:
    uploaded_file = st.file_uploader(
        "上传 CSV 文件",
        type=['csv'],
        help="格式: 日期, 股票代码, 开盘价, 收盘价, 最高价, 最低价, 成交量"
    )

# 处理文件上传
stock_data = None
if uploaded_file is not None:
    try:
        stock_data = pd.read_csv(uploaded_file)
        stock_data.columns = ['date', 'code', 'open', 'close', 'high', 'low', 'volume']
        stock_data['date'] = pd.to_datetime(stock_data['date'], format='mixed')
        stock_data['date'] = stock_data['date'].dt.strftime('%Y-%m-%d')
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("数据条数", len(stock_data))
        with col2:
            st.metric("股票数量", stock_data['code'].nunique())
        with col3:
            st.metric("开始日期", stock_data['date'].min())
        with col4:
            st.metric("结束日期", stock_data['date'].max())
        
        st.subheader("数据预览")
        st.dataframe(stock_data.head(10), use_container_width=True)
        
    except Exception as e:
        st.error(f"文件解析失败: {e}")

# 参数设置
if stock_data is not None:
    st.header("⚙️ 参数设置")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        initial_capital = st.number_input(
            "初始资金（¥）",
            value=100000,
            step=1000,
            min_value=10000
        )
    
    with col2:
        start_date = st.date_input(
            "起始日期",
            value=pd.to_datetime(stock_data['date'].min()),
            min_value=pd.to_datetime(stock_data['date'].min()),
            max_value=pd.to_datetime(stock_data['date'].max())
        )
    
    with col3:
        end_date = st.date_input(
            "终止日期",
            value=pd.to_datetime(stock_data['date'].max()),
            min_value=pd.to_datetime(stock_data['date'].min()),
            max_value=pd.to_datetime(stock_data['date'].max())
        )
    
    with col4:
        strategy = st.selectbox(
            "策略选择",
            options=list(STRATEGIES.keys()),
            format_func=lambda x: STRATEGIES[x]
        )
    
    # 自定义条件
    buy_condition = ""
    sell_condition = ""
    if strategy == 'custom':
        col1, col2 = st.columns(2)
        with col1:
            buy_condition = st.text_input(
                "买入条件（自定义）",
                value="close > open",
                help="可用变量: open, close, high, low, volume, ma5, ma20, macdLine, signalLine, rsi, k, d, j, bb_upper, bb_middle, bb_lower, wr, plusDI, minusDI, adx"
            )
        with col2:
            sell_condition = st.text_input(
                "卖出条件（自定义）",
                value="close < open",
                help="可用变量同上"
            )
    
    # 运行回测按钮
    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        run_btn = st.button("🚀 运行回测", use_container_width=True, type="primary")
    
    if run_btn:
        with st.spinner("回测中，请稍候..."):
            try:
                results = run_backtest(
                    stock_data,
                    initial_capital,
                    start_date.strftime('%Y-%m-%d'),
                    end_date.strftime('%Y-%m-%d'),
                    strategy,
                    buy_condition,
                    sell_condition
                )
                
                # 显示结果
                st.header("📊 回测结果")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("最终资金", f"¥{results['final_capital']:.2f}")
                with col2:
                    color = "normal" if results['total_return'] >= 0 else "inverse"
                    st.metric(
                        "总收益率",
                        f"{results['total_return']:.2f}%",
                        delta=f"{results['total_return']:.2f}%",
                        delta_color=color
                    )
                with col3:
                    st.metric("交易次数", results['trade_count'])
                
                # 图表
                st.subheader("资产曲线")
                col1, col2 = st.columns(2)
                
                with col1:
                    fig_asset = go.Figure()
                    fig_asset.add_trace(go.Scatter(
                        x=results['asset_dates'],
                        y=results['asset_values'],
                        mode='lines',
                        name='总资产',
                        line=dict(color='#1a1a1a', width=2),
                        fill='tozeroy',
                        fillcolor='rgba(26, 26, 26, 0.1)'
                    ))
                    fig_asset.update_layout(
                        title='总资产（现金 + 持仓市值）',
                        xaxis_title='日期',
                        yaxis_title='资金（¥）',
                        hovermode='x unified',
                        height=400,
                        template='plotly_white'
                    )
                    st.plotly_chart(fig_asset, use_container_width=True)
                
                with col2:
                    fig_return = go.Figure()
                    return_color = '#22a06b' if results['total_return'] >= 0 else '#ca3521'
                    fig_return.add_trace(go.Scatter(
                        x=results['asset_dates'],
                        y=results['return_values'],
                        mode='lines',
                        name='累计收益率',
                        line=dict(color=return_color, width=2),
                        fill='tozeroy',
                        fillcolor=return_color + '20'
                    ))
                    fig_return.update_layout(
                        title='累计收益率（%）',
                        xaxis_title='日期',
                        yaxis_title='收益率（%）',
                        hovermode='x unified',
                        height=400,
                        template='plotly_white'
                    )
                    st.plotly_chart(fig_return, use_container_width=True)
                
                # 交易日志
                st.subheader("📋 交易日志")
                if len(results['trade_log']) == 0:
                    st.info("无交易记录")
                else:
                    trade_log = results['trade_log'].copy()
                    trade_log['price'] = trade_log['price'].apply(lambda x: f"¥{x:.2f}")
                    trade_log['fee'] = trade_log['fee'].apply(lambda x: f"¥{x:.4f}")
                    trade_log['cash'] = trade_log['cash'].apply(lambda x: f"¥{x:.2f}")
                    
                    st.dataframe(trade_log, use_container_width=True, hide_index=True)
                    
                    # 下载选项
                    csv = results['trade_log'].to_csv(index=False)
                    st.download_button(
                        label="📥 下载交易日志 (CSV)",
                        data=csv,
                        file_name="trade_log.csv",
                        mime="text/csv"
                    )
                
            except Exception as e:
                st.error(f"回测失败: {e}")
                st.error("请检查数据格式或参数是否正确")
else:
    st.info("👈 请在左侧上传 CSV 文件开始回测")

# 页脚
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #888; font-size: 12px;">
    <p>💡 所有计算在浏览器本地进行，数据不会上传到任何服务器</p>
    <p>⭐ 如果觉得有用，请在 <a href="https://github.com/dannn547/stock-backtest-tool" target="_blank">GitHub</a> 上给个 Star</p>
</div>
""", unsafe_allow_html=True)
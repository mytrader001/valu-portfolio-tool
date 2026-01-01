import streamlit as st
import quantstats as qs
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import platform
import numpy as np
from datetime import datetime

st.set_page_config(page_title="ポートフォリオ分析ツール", layout="wide")
st.title("📊 ポートフォリオ分析ツール")

system = platform.system()
if system == "Windows":
    plt.rcParams['font.family'] = 'MS Gothic'
elif system == "Darwin":
    plt.rcParams['font.family'] = 'AppleGothic'
else:
    sns.set(font='DejaVu Sans') 

if 'weights_val' not in st.session_state:
    st.session_state.weights_val = "0.25, 0.25, 0.25, 0.25"

def set_equal_weights():
    current_text = st.session_state.tickers_input
    if current_text:
        t_list = [t for t in current_text.split(',') if t.strip()]
        count = len(t_list)
        if count > 0:
            base_val = round(1.0 / count, 2)
            weights = [base_val] * (count - 1)
            current_sum = sum(weights)
            last_val = round(1.0 - current_sum, 2)
            weights.append(last_val)
            new_weights = ", ".join([f"{w:.2f}" for w in weights])
            st.session_state.weights_val = new_weights

st.sidebar.header("ポートフォリオ設定")

default_tickers = "GOOGL, AAPL, META, AMZN" 
tickers_input = st.sidebar.text_area(
    "銘柄コード (カンマ区切り)", 
    default_tickers, 
    key="tickers_input", 
    help="米国株はティッカー、日本株はコード (例: 7203 または 130A) で入力してください。" 
)

st.sidebar.button("均等配分を設定", on_click=set_equal_weights, help="入力された銘柄数で均等に割ります（端数は最後の銘柄で調整されます）")

weights_input = st.sidebar.text_input(
    "ウェイト変更 (カンマ区切り)", 
    key="weights_val", 
    help="各銘柄の割合。合計が1になるようにしてください。"
)

rebalance_option = st.sidebar.radio(
    "リバランス設定",
    ("なし (Buy & Hold)", "毎日リバランス (ウェイト理論値)"),
    help="ポートフォリオ理論においては、ウェイトを常に一定に保つことが前提であり、理論的には連続的（毎日）なリバランスが最適となります。"
)

benchmark_options = {
    "S&P 500 (SPY)": "SPY",
    "全米株式 (VTI)": "VTI",
    "TOPIX (1306.T)": "1306.T",
    "日経225 (^N225)": "^N225"
}
benchmark_label = st.sidebar.selectbox("ベンチマーク", list(benchmark_options.keys()))
benchmark_symbol = benchmark_options[benchmark_label]

start_date = st.sidebar.date_input("開始日", datetime(2000, 1, 1))
end_date = st.sidebar.date_input("終了日", datetime.now())

run_btn = st.sidebar.button("分析実行")

def get_data(tickers, start, end):
    try:
        data = yf.download(tickers, start=start, end=end, progress=False, auto_adjust=True)['Close']
        if isinstance(data, pd.Series):
            data = data.to_frame()
            data.columns = tickers
        return data
    except Exception as e:
        st.error(f"データ取得エラー: {e}")
        return None

if run_btn:
    raw_tickers = [t.strip() for t in tickers_input.split(',')]
    ticker_list = []
    
    for t in raw_tickers:
        if not t: continue 
        if t.upper().endswith(".T"):
            ticker_list.append(t)
        elif t[0].isdigit():
            ticker_list.append(f"{t}.T")
        else:
            ticker_list.append(t)

    try:
        weight_list = [float(w.strip()) for w in weights_input.split(',')]
    except ValueError:
        st.error("ウェイトは数値で入力してください。")
        st.stop()

    if len(ticker_list) != len(weight_list):
        st.error(f"銘柄数({len(ticker_list)})とウェイト数({len(weight_list)})が一致しません。")
        st.stop()

    total_weight = sum(weight_list)
    if abs(total_weight - 1.0) > 0.0001:
        st.error(f"ウェイトの合計が1.0になっていません（現在の合計: {total_weight:.2f}）。合計が1.0になるように数値を調整してください。")
        st.stop() 

    with st.spinner('データを取得・計算中...'):
        df_prices = get_data(ticker_list, start_date, end_date)
        bm_prices = get_data([benchmark_symbol], start_date, end_date)

        if df_prices is None or bm_prices is None or df_prices.empty:
            st.error("データの取得に失敗しました。銘柄コードを確認してください。")
            st.stop()
            
        returns = df_prices.pct_change().dropna()
        bm_returns = bm_prices.pct_change().dropna()
        
        if isinstance(bm_returns, pd.DataFrame):
            if bm_returns.shape[1] >= 1:
                bm_returns = bm_returns.iloc[:, 0]

        common_idx = returns.index.intersection(bm_returns.index)
        returns = returns.loc[common_idx]
        bm_returns = bm_returns.loc[common_idx]
        
        if rebalance_option == "毎日リバランス (ウェイト理論値)":
            portfolio_returns = returns.dot(weight_list)
            chart_label_text = "Portfolio (Daily Rebalance)"
            caption_text = "※ポートフォリオ理論に基づき、毎日ウェイトを一定に保つ計算結果です。ベータ値とシャープレシオを算出します"
            is_rebalance = True
        else:
            cumulative_returns = (1 + returns).cumprod()
            portfolio_value = cumulative_returns.dot(weight_list)
            portfolio_returns = portfolio_value.pct_change()
            portfolio_returns.iloc[0] = portfolio_value.iloc[0] - 1.0
            
            chart_label_text = "Portfolio (Buy & Hold)"
            caption_text = "※リバランスを行わない（Buy & Hold）前提での計算結果です。"
            is_rebalance = False
        
        portfolio_returns.name = "Portfolio"

        st.subheader("1. 主要指標") 
        st.caption(caption_text)
        
        total_return = qs.stats.comp(portfolio_returns)
        cagr = qs.stats.cagr(portfolio_returns)
        volatility = qs.stats.volatility(portfolio_returns)
        max_dd = qs.stats.max_drawdown(portfolio_returns)

        if is_rebalance:
            rf = 0.0
            sharpe = qs.stats.sharpe(portfolio_returns, rf=rf)
            beta = qs.stats.greeks(portfolio_returns, bm_returns)['beta']

            col1, col2, col3 = st.columns(3)
            col1.metric("トータルリターン", f"{total_return:.2%}")
            col2.metric("CAGR (年平均成長率)", f"{cagr:.2%}")
            col3.metric("ボラティリティ (リスク)", f"{volatility:.2%}")

            col4, col5, col6 = st.columns(3)
            col4.metric("ベータ値", f"{beta:.2f}")
            col5.metric("シャープレシオ", f"{sharpe:.2f}")
            col6.metric("最大ドローダウン", f"{max_dd:.2%}")
        else:
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("トータルリターン", f"{total_return:.2%}")
            col2.metric("CAGR (年平均成長率)", f"{cagr:.2%}")
            col3.metric("ボラティリティ (リスク)", f"{volatility:.2%}")
            col4.metric("最大ドローダウン", f"{max_dd:.2%}")

        st.markdown("---")

        st.subheader("2. 累積リターン比較") 
        fig1, ax1 = plt.subplots(figsize=(10, 5))
        
        cum_port = (1 + portfolio_returns).cumprod() - 1
        cum_bm = (1 + bm_returns).cumprod() - 1
        
        ax1.plot(cum_port.index, cum_port.values, label=chart_label_text, linewidth=2.5, color='#1f77b4')
        ax1.plot(cum_bm.index, cum_bm.values, label=f"Benchmark ({benchmark_label})", linewidth=1.5, alpha=0.8, linestyle='--', color='black')
        
        ax1.set_title("ポートフォリオ vs ベンチマーク")
        ax1.legend()
        ax1.grid(True, linestyle='--', alpha=0.6)
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:.0%}".format(x)))
        st.pyplot(fig1)

        st.subheader("3. 個別銘柄の累積リターン比較") 
        fig_ind, ax_ind = plt.subplots(figsize=(10, 5))

        cum_assets = (1 + returns).cumprod() - 1
        colors = plt.cm.tab10(np.linspace(0, 1, len(returns.columns)))
        
        for i, col in enumerate(returns.columns):
            ax_ind.plot(cum_assets.index, cum_assets[col].values, label=col, linewidth=1.5, alpha=0.7, color=colors[i])
            
        ax_ind.plot(cum_bm.index, cum_bm.values, label=f"Benchmark ({benchmark_label})", linewidth=2.0, linestyle='--', color='black')

        ax_ind.set_title("個別銘柄 vs ベンチマーク")
        ax_ind.legend(loc='upper left', bbox_to_anchor=(1, 1))
        ax_ind.grid(True, linestyle='--', alpha=0.6)
        ax_ind.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:.0%}".format(x)))
        plt.tight_layout()
        st.pyplot(fig_ind)

        st.subheader("4. リスク推移") 
        fig2, ax2 = plt.subplots(figsize=(10, 4))
        
        window_vol = 20
        roll_vol_port = portfolio_returns.rolling(window_vol).std() * np.sqrt(252)
        roll_vol_bm = bm_returns.rolling(window_vol).std() * np.sqrt(252)
        
        ax2.plot(roll_vol_port.index, roll_vol_port.values, label="Portfolio Volatility", linewidth=1.5)
        ax2.plot(roll_vol_bm.index, roll_vol_bm.values, label="Benchmark Volatility", linewidth=1.5, alpha=0.7, linestyle='--')
        
        ax2.set_title("20日移動ボラティリティ (年率換算)")
        ax2.legend()
        ax2.grid(True, linestyle='--', alpha=0.6)
        st.pyplot(fig2)

        next_chart_num = 5

        if is_rebalance:
            st.subheader(f"{next_chart_num}. ベータ値の推移") 
            fig3, ax3 = plt.subplots(figsize=(10, 4))
            
            window_beta = 60
            cov = portfolio_returns.rolling(window_beta).cov(bm_returns)
            var = bm_returns.rolling(window_beta).var()
            rolling_beta_series = cov / var
            
            ax3.plot(rolling_beta_series.index, rolling_beta_series.values, label="Beta", color='orange')
            ax3.axhline(1.0, color='gray', linestyle='--', linewidth=0.8, label="Beta = 1.0")
            
            ax3.set_title("60日移動ベータ")
            ax3.legend()
            ax3.grid(True, linestyle='--', alpha=0.6)
            st.pyplot(fig3)
            
            next_chart_num += 1

        st.subheader(f"{next_chart_num}. 年次リターン") 
        next_chart_num += 1
        
        try:
            resample_rule = 'YE'
            p_yearly = portfolio_returns.resample(resample_rule).apply(lambda x: (1 + x).prod() - 1)
        except ValueError:
            resample_rule = 'Y'
            p_yearly = portfolio_returns.resample(resample_rule).apply(lambda x: (1 + x).prod() - 1)
            
        b_yearly = bm_returns.resample(resample_rule).apply(lambda x: (1 + x).prod() - 1)

        yearly_df = pd.DataFrame({
            'Portfolio': p_yearly,
            'Benchmark': b_yearly
        })
        yearly_df.index = yearly_df.index.strftime('%Y')

        fig4, ax4 = plt.subplots(figsize=(10, 5))
        yearly_df.plot(kind='bar', ax=ax4, width=0.8)
        
        ax4.set_title("年次リターン比較")
        ax4.grid(True, axis='y', linestyle='--', alpha=0.6)
        ax4.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:.0%}".format(x)))
        plt.xticks(rotation=45)
        st.pyplot(fig4)

        st.subheader(f"{next_chart_num}. 月次リターン") 
        next_chart_num += 1
        try:
            monthly_table = qs.stats.monthly_returns(portfolio_returns)
            st.dataframe(monthly_table.style.format("{:.2%}").background_gradient(cmap='RdYlGn', axis=None, vmin=-0.1, vmax=0.1))
        except AttributeError:
            m_ret = portfolio_returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
            m_ret_df = pd.DataFrame({'Year': m_ret.index.year, 'Month': m_ret.index.month, 'Return': m_ret.values})
            monthly_table = m_ret_df.pivot(index='Year', columns='Month', values='Return')
            st.dataframe(monthly_table.style.format("{:.2%}").background_gradient(cmap='RdYlGn', axis=None, vmin=-0.1, vmax=0.1))

        st.subheader(f"{next_chart_num}. ポートフォリオ内銘柄相関") 
        next_chart_num += 1
        if len(ticker_list) > 1:
            fig5, ax5 = plt.subplots(figsize=(8, 6))
            sns.heatmap(returns.corr(), annot=True, cmap='coolwarm', ax=ax5, fmt=".2f")
            ax5.set_title("相関係数マトリクス")
            st.pyplot(fig5)
        else:
            st.info("銘柄が1つのため相関図は表示されません。")
            
        st.subheader(f"{next_chart_num}. ドローダウンの推移") 
        next_chart_num += 1
        fig6, ax6 = plt.subplots(figsize=(10, 5))
        
        wealth_index = (1 + portfolio_returns).cumprod()
        previous_peaks = wealth_index.cummax()
        drawdown = (wealth_index - previous_peaks) / previous_peaks
        
        wealth_index_bm = (1 + bm_returns).cumprod()
        previous_peaks_bm = wealth_index_bm.cummax()
        drawdown_bm = (wealth_index_bm - previous_peaks_bm) / previous_peaks_bm

        ax6.plot(drawdown.index, drawdown.values, label="Portfolio", color='#d62728', linewidth=1.5)
        ax6.fill_between(drawdown.index, drawdown.values, 0, color='#d62728', alpha=0.3)
        
        ax6.plot(drawdown_bm.index, drawdown_bm.values, label=f"Benchmark ({benchmark_label})", color='gray', linestyle='--', linewidth=1.0, alpha=0.8)

        ax6.set_title("ドローダウン（下落率）の推移")
        ax6.set_ylabel("Drawdown")
        ax6.legend()
        ax6.grid(True, linestyle='--', alpha=0.6)
        ax6.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:.0%}".format(x)))
        st.pyplot(fig6)

        st.subheader(f"{next_chart_num}. リスク・リターン散布図") 
        
        if len(ticker_list) > 1:
            fig7, ax7 = plt.subplots(figsize=(10, 6))
            
            summary_data = []
            for col in returns.columns:
                r = qs.stats.cagr(returns[col])
                v = qs.stats.volatility(returns[col])
                summary_data.append({'Label': col, 'Return': r, 'Risk': v, 'Type': 'Asset'})
            
            p_r = qs.stats.cagr(portfolio_returns)
            p_v = qs.stats.volatility(portfolio_returns)
            summary_data.append({'Label': 'Portfolio', 'Return': p_r, 'Risk': p_v, 'Type': 'Portfolio'})
            
            b_r = qs.stats.cagr(bm_returns)
            b_v = qs.stats.volatility(bm_returns)
            summary_data.append({'Label': f"Benchmark\n({benchmark_label})", 'Return': b_r, 'Risk': b_v, 'Type': 'Benchmark'})
            
            df_scatter = pd.DataFrame(summary_data)
            
            colors = {'Asset': 'skyblue', 'Portfolio': 'blue', 'Benchmark': 'gray'}
            sizes = {'Asset': 100, 'Portfolio': 300, 'Benchmark': 200}
            
            for _, row in df_scatter.iterrows():
                ax7.scatter(row['Risk'], row['Return'], 
                            color=colors[row['Type']], 
                            s=sizes[row['Type']], 
                            edgecolors='black', 
                            label=row['Type'] if row['Type'] not in ax7.get_legend_handles_labels()[1] else "")
                
                ax7.text(row['Risk'], row['Return'] + 0.005, row['Label'], fontsize=9, ha='center')

            ax7.set_title("リスク・リターン散布図 (左上にあるほど優秀)")
            ax7.set_xlabel("Risk (Volatility)")
            ax7.set_ylabel("Return (CAGR)")
            
            ax7.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:.0%}".format(x)))
            ax7.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:.0%}".format(x)))
            ax7.grid(True, linestyle='--', alpha=0.6)
            
            ax7.axhline(0, color='black', linewidth=0.8)
            ax7.axvline(0, color='black', linewidth=0.8)
            
            st.pyplot(fig7)
        else:
            st.info("銘柄が1つのため散布図は表示されません。")

else:
    st.info("サイドバーで条件を設定し、「分析実行」ボタンを押してください。")
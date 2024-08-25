import pandas as pd
import tkinter as tk
from tkinter import ttk, filedialog
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from statsmodels.tsa.arima.model import ARIMA
import chardet

# CSVファイルのエンコーディングを検出する関数
def detect_encoding(file_path):
    with open(file_path, 'rb') as file:
        result = chardet.detect(file.read())
    return result['encoding']

# CSVファイルを読み込む関数
def load_csv(file_path):
    encoding = detect_encoding(file_path)
    
    try:
        # データを読み込む
        df = pd.read_csv(file_path, encoding=encoding, skiprows=[0, 1, 2, 3], header=None)
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return None
    
    # 実際の列数に合わせて列名を設定
    df.columns = ['Date', 'MeanTemp1', 'MeanTemp2', 'MeanTemp3', 'MaxTemp1', 'MaxTemp2', 'MaxTemp3',
                  'MinTemp1', 'MinTemp2', 'MinTemp3', 'Precipitation1', 'Precipitation2', 'Precipitation3',
                  'Sunshine1', 'Sunshine2', 'Sunshine3', 'Humidity1', 'Humidity2', 'Humidity3']
    
    # 必要な列だけを抽出
    columns_to_keep = ['Date', 'MeanTemp1', 'MaxTemp1', 'MinTemp1', 'Precipitation1', 'Sunshine1']
    df = df[columns_to_keep]
    
    # 新しい列名を設定
    new_columns = ['Date', 'MeanTemp', 'MaxTemp', 'MinTemp', 'Precipitation', 'Sunshine']
    df.columns = new_columns
    
    # データのクリーンアップ
    df = df.dropna(subset=['Date'])
    df['Date'] = pd.to_datetime(df['Date'], format='%Y/%m/%d', errors='coerce')
    
    for col in df.columns[1:]:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    print(df.dtypes)
    print(df.head())
    
    return df

# データをプロットする関数
def plot_data(df, ax, title):
    ax.clear()
    for column in df.columns[1:]:
        if df[column].dtype != 'object' and df[column].notna().any():
            ax.plot(df['Date'], df[column], label=column)
    ax.set_xlabel('Date')
    ax.set_ylabel('Values')
    ax.legend()
    ax.set_title(title)
    ax.set_xlim(df['Date'].min(), df['Date'].max())
    ax.figure.autofmt_xdate()

# 予測と可視化を行う関数
def plot_forecast(df, ax):
    # データの前処理
    df = df[['Date', 'MeanTemp']]
    df.set_index('Date', inplace=True)
    df = df.asfreq('D')  # 日単位の頻度を設定して欠損値を補完
    
    # 欠損値の補完
    df['MeanTemp'] = df['MeanTemp'].ffill()  # 前方補完
    
    # ARIMAモデルの作成と適合
    model = ARIMA(df['MeanTemp'], order=(5, 1, 0))  # パラメータはデータに応じて調整する必要があります
    model_fit = model.fit()
    
    # 予測を行う
    forecast_steps = 4 * 365  # 4年間の日数（うるう年を含むと約1461日）
    forecast = model_fit.get_forecast(steps=forecast_steps)
    forecast_mean = forecast.predicted_mean  # 予測値の平均
    forecast_conf_int = forecast.conf_int()  # 予測の信頼区間

    # 予測用のインデックスを生成
    forecast_index = pd.date_range(start='2020-04-01', periods=forecast_steps, freq='D')

    # 元のデータと予測結果をプロット
    ax.plot(df.index, df['MeanTemp'], label='Historical Data')
    ax.plot(forecast_index, forecast_mean, label='Forecast', color='red')

    # 予測の信頼区間をプロット
    ax.fill_between(forecast_index, forecast_conf_int.iloc[:, 0], forecast_conf_int.iloc[:, 1], color='red', alpha=0.3, label='95% Confidence Interval')

    # x軸の範囲を調整
    ax.set_xlim(df.index.min(), forecast_index.max())

    ax.set_title('Time Series Forecast')
    ax.set_xlabel('Year/Month')
    ax.set_ylabel('Mean Temperature')
    ax.legend()

# GUIでプロットを表示する関数
def show_gui(df):
    root = tk.Tk()
    root.title("Weather Data Visualization")

    notebook = ttk.Notebook(root)
    notebook.pack(expand=True, fill='both')

    # データテーブルタブ
    tab1 = ttk.Frame(notebook)
    notebook.add(tab1, text='Data Table')

    tree = ttk.Treeview(tab1, columns=list(df.columns), show='headings')
    tree.pack(expand=True, fill='both')

    for col in df.columns:
        tree.heading(col, text=col)
        tree.column(col, width=100)

    for _, row in df.iterrows():
        tree.insert('', 'end', values=list(row))

    # データ可視化タブ
    tab2 = ttk.Frame(notebook)
    notebook.add(tab2, text='Data Visualization')

    fig1, ax1 = plt.subplots(figsize=(10, 6))
    plot_data(df, ax1, 'Weather Data Visualization')
    canvas1 = FigureCanvasTkAgg(fig1, master=tab2)
    canvas1.get_tk_widget().pack(expand=True, fill='both')
    canvas1.draw()

    # 予測可視化タブ
    tab3 = ttk.Frame(notebook)
    notebook.add(tab3, text='Forecast Visualization')

    fig2, ax2 = plt.subplots(figsize=(10, 6))
    plot_forecast(df, ax2)
    canvas2 = FigureCanvasTkAgg(fig2, master=tab3)
    canvas2.get_tk_widget().pack(expand=True, fill='both')
    canvas2.draw()

    root.mainloop()

if __name__ == "__main__":
    file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
    if file_path:
        df = load_csv(file_path)
        if df is not None:
            show_gui(df)
        else:
            print("Failed to load CSV file.")
    else:
        print("No file selected.")

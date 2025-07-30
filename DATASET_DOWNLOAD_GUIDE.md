# Dataset Download Scripts

這裡提供了兩個 Python 腳本來下載 UFGVC 數據集的 parquet 文件。

## 可用的數據集

- `cotton80`: 棉花分類數據集，包含 80 個類別
- `soybean`: 大豆分類數據集  
- `soy_ageing_r1`: 大豆老化數據集 - Round 1
- `soy_ageing_r3`: 大豆老化數據集 - Round 3
- `soy_ageing_r4`: 大豆老化數據集 - Round 4
- `soy_ageing_r5`: 大豆老化數據集 - Round 5
- `soy_ageing_r6`: 大豆老化數據集 - Round 6

## 方法 1: 完整功能腳本 (download_dataset.py)

這是一個功能完整的命令行工具，支持多種選項。

### 安裝依賴
```bash
pip install requests pandas
```

### 使用方法

#### 查看所有可用數據集
```bash
python download_dataset.py --list
```

#### 下載單個數據集
```bash
python download_dataset.py --dataset cotton80
```

#### 下載到指定目錄
```bash
python download_dataset.py --dataset cotton80 --output ./my_datasets
```

#### 下載多個數據集
```bash
python download_dataset.py --dataset cotton80,soybean,soy_ageing_r1
```

#### 下載所有數據集
```bash
python download_dataset.py --all
```

#### 強制重新下載（覆蓋已存在的文件）
```bash
python download_dataset.py --dataset cotton80 --force
```

### 命令行選項

- `--dataset, -d`: 指定要下載的數據集名稱（多個用逗號分隔）
- `--output, -o`: 指定輸出目錄（默認：./data）
- `--list, -l`: 列出所有可用數據集
- `--all, -a`: 下載所有數據集
- `--force, -f`: 強制重新下載已存在的文件

## 方法 2: 快速下載腳本 (quick_download.py)

這是一個簡化的交互式腳本，使用更簡單。

### 使用方法

直接運行腳本：
```bash
python quick_download.py
```

然後按照提示選擇要下載的數據集：
- 輸入數據集名稱（如：cotton80）
- 輸入數字（如：1 代表 cotton80）
- 輸入 "all" 下載所有數據集

### 程式化使用

你也可以在其他 Python 腳本中導入使用：

```python
from quick_download import download_parquet_dataset

# 下載 cotton80 數據集到 ./data 目錄
filepath = download_parquet_dataset('cotton80', './data')
if filepath:
    print(f"Downloaded to: {filepath}")
```

## 數據集結構

下載的 parquet 文件包含以下列：

- `image`: 圖像數據（bytes）
- `label`: 數值標籤
- `class_name`: 類別名稱
- `split`: 數據分割（train/test/val）

## 示例用法

### 1. 快速開始

```bash
# 列出所有可用數據集
python download_dataset.py --list

# 下載 cotton80 數據集
python download_dataset.py --dataset cotton80
```

### 2. 批量下載
```bash
# 下載多個數據集
python download_dataset.py --dataset cotton80,soybean

# 下載所有數據集
python download_dataset.py --all
```

### 3. 自定義設置
```bash
# 下載到特定目錄並強制重新下載
python download_dataset.py --dataset cotton80 --output ./datasets --force
```

## 注意事項

1. 數據集文件較大（每個約 100-200MB），請確保有足夠的磁盤空間
2. 下載過程中會顯示進度條
3. 如果文件已存在，默認會跳過下載（除非使用 --force）
4. 腳本會自動創建輸出目錄
5. 下載失敗時會自動清理不完整的文件

## 故障排除

### 下載失敗
- 檢查網路連接
- 確認 Hugging Face 服務是否正常
- 嘗試使用 `--force` 重新下載

### 權限錯誤
- 確保對輸出目錄有寫入權限
- 在 Windows 上可能需要以管理員身份運行

### 磁盤空間不足
- 檢查可用磁盤空間
- 考慮使用不同的輸出目錄

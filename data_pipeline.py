import sys
import os

# Sửa lỗi in tiếng Việt trên Terminal Windows
sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
import torch
from torch.utils.data import Dataset, DataLoader

class TextDataset(Dataset):
    """
    Class Dataset tùy chỉnh của PyTorch để biến văn bản thành vector số (Tokenization).
    """
    def __init__(self, texts, targets, tokenizer, max_len):
        self.texts = texts
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_len = max_len
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        target = self.targets[idx]
        
        # Hàm encode_plus biến chữ thành số chuẩn bị cho mô hình Transformer
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'targets': torch.tensor(target, dtype=torch.float)
        }

def process_and_save_data(input_path, output_path):
    """
    Hàm này đọc file gốc, làm sạch, chia split và lưu ra file mới.
    """
    print("1. Đang đọc dữ liệu gốc (Có thể mất vài chục giây)...")
    df = pd.read_csv(input_path)
    df = df[['Text', 'Score']]
    df = df.rename(columns={'Text': 'text', 'Score': 'rating'})
    
    print("2. Đang làm sạch dữ liệu (Cleaning & Validation)...")
    df = df.dropna()
    df = df.drop_duplicates(subset=['text'])
    df = df[(df['rating'] >= 1) & (df['rating'] <= 5)]
    
    # Lấy mẫu 50,000 dòng để chống tràn RAM khi train
    if len(df) > 50000:
        df = df.sample(n=50000, random_state=42)
        
    print("3. Đang tính Target và chia Split...")
    # Map rating 1-5 sang 0.0-1.0
    df['target'] = (df['rating'] - 1) / 4.0
    
    # Chia tập train/val/test
    df_train, df_temp = train_test_split(df, test_size=0.2, random_state=42, stratify=df['rating'])
    df_val, df_test = train_test_split(df_temp, test_size=0.5, random_state=42, stratify=df_temp['rating'])
    
    df_train = df_train.copy(); df_train['split'] = 'train'
    df_val = df_val.copy(); df_val['split'] = 'val'
    df_test = df_test.copy(); df_test['split'] = 'test'
    
    final_df = pd.concat([df_train, df_val, df_test])
    final_df = final_df[['text', 'rating', 'target', 'split']]
    
    # Lưu file sạch
    final_df.to_csv(output_path, index=False)
    print(f"-> Đã lưu file sạch tại: {output_path}")
    
    return df_train, df_val, df_test

def get_dataloaders(df_train, df_val, df_test, model_name='bert-base-uncased', batch_size=16):
    """
    Hàm này tạo DataLoader để Member 3 mang đi huấn luyện.
    """
    print("4. Đang tạo Tokenization và DataLoader (Đang tải Model từ HuggingFace)...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    max_len = 128
    
    train_dataset = TextDataset(df_train['text'].to_numpy(), df_train['target'].to_numpy(), tokenizer, max_len)
    val_dataset = TextDataset(df_val['text'].to_numpy(), df_val['target'].to_numpy(), tokenizer, max_len)
    test_dataset = TextDataset(df_test['text'].to_numpy(), df_test['target'].to_numpy(), tokenizer, max_len)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    print("Hoàn tất quy trình DataLoader!")
    return train_loader, val_loader, test_loader

# =========================================================
# PHẦN THỰC THI CHÍNH CỦA CHƯƠNG TRÌNH NẰM NGOÀI CÙNG Ở ĐÂY
# =========================================================
if __name__ == "__main__":
    # 1. Lấy vị trí của thư mục 'src' (nơi chứa file code hiện tại)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 2. File dữ liệu hiện đang nằm chung thư mục với file code!
    input_file = os.path.join(current_dir, "Reviews.csv") 
    output_file = os.path.join(current_dir, "clean_dataset.csv")
    
    print("--- BẮT ĐẦU CHẠY PIPELINE ---")
    print(f"Máy tính đang tìm file tại vị trí chính xác là: {input_file}")
    
    # Kiểm tra xem file có tồn tại không trước khi chạy
    if not os.path.exists(input_file):
        print("LỖI: Vẫn không tìm thấy file! Hãy kiểm tra lại.")
    else:
        # 1. Gọi hàm xử lý dữ liệu
        df_train, df_val, df_test = process_and_save_data(input_file, output_file)
        
        # 2. Gọi hàm tạo dataloader
        train_loader, val_loader, test_loader = get_dataloaders(df_train, df_val, df_test)
        
        print("--- ĐÃ HOÀN TẤT MỌI VIỆC ---")
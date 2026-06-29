import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
from tqdm import tqdm
import json

class TibetanDataset(Dataset):
    def __init__(self, data_dir, split='train', char_to_idx=None):
        self.lines_dir = os.path.join(data_dir, 'acent', 'lines')
        self.trans_dir = os.path.join(data_dir, 'acent', 'transcriptions')
        all_images = sorted([f for f in os.listdir(self.lines_dir) if f.endswith('.jpg')])
        split_idx = int(len(all_images) * 0.8)
        self.image_files = all_images[:split_idx] if split == 'train' else all_images[split_idx:]
        self.transform = transforms.Compose([
            transforms.Resize((32, 512)),  # 增加宽度到512
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3)
        ])
        if char_to_idx is None:
            self.char_to_idx = self.build_char_dict(data_dir)
        else:
            self.char_to_idx = char_to_idx
        self.idx_to_char = {v: k for k, v in self.char_to_idx.items()}

    def build_char_dict(self, data_dir):
        char_to_idx = {'blank': 0}
        idx = 1
        with open(os.path.join(data_dir, 'character_list.txt'), 'r', encoding='utf-8') as f:
            for line in f:
                char = line.strip()
                if char and char not in char_to_idx:
                    char_to_idx[char] = idx
                    idx += 1
        return char_to_idx

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.lines_dir, self.image_files[idx])
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)
        txt_path = os.path.join(self.trans_dir, self.image_files[idx].replace('.jpg', '.txt'))
        label = open(txt_path, 'r', encoding='utf-8').read().strip() if os.path.exists(txt_path) else ''
        label_indices = [self.char_to_idx.get(c, 0) for c in label if c in self.char_to_idx]
        return image, torch.tensor(label_indices, dtype=torch.long), label

class CRNN(nn.Module):
    def __init__(self, num_classes):
        super(CRNN, self).__init__()
        # Input: 3 x 32 x 512
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 16 x 256
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 8 x 128
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.MaxPool2d((2, 2)),  # 4 x 64
            nn.Conv2d(256, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(),
            nn.MaxPool2d((1, 2)),  # 4 x 32
            nn.Conv2d(512, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(),
        )
        self.rnn = nn.LSTM(2048, 256, 2, bidirectional=True, batch_first=True, dropout=0.3)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.cnn(x)
        b, c, h, w = x.size()
        x = x.permute(0, 3, 1, 2).reshape(b, w, c * h)
        x, _ = self.rnn(x)
        x = self.fc(x)
        return x

def collate_fn(batch):
    images = torch.stack([item[0] for item in batch])
    labels = [item[1] for item in batch]
    texts = [item[2] for item in batch]
    return images, labels, texts

def ctc_decode(pred, idx_to_char, blank=0):
    result = []
    prev = -1
    for p in pred:
        if p != prev and p != blank:
            result.append(idx_to_char.get(p, ''))
        prev = p
    return ''.join(result)

def train():
    DATA_DIR = '/home/xufei/tibet_acent'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    train_dataset = TibetanDataset(DATA_DIR, 'train')
    test_dataset = TibetanDataset(DATA_DIR, 'test', train_dataset.char_to_idx)
    print(f'Train: {len(train_dataset)}, Test: {len(test_dataset)}')
    print(f'Num classes: {len(train_dataset.char_to_idx)}')
    
    # 检查样本
    sample_img, sample_label, sample_text = train_dataset[0]
    print(f'Sample text length: {len(sample_text)}, label length: {len(sample_label)}')
    
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=2, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=2, collate_fn=collate_fn)
    
    model = CRNN(num_classes=len(train_dataset.char_to_idx)).to(device)
    criterion = nn.CTCLoss(blank=0, reduction='mean', zero_infinity=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    best_char_acc = 0
    history = {'train_loss': [], 'test_char_acc': [], 'test_seq_acc': []}
    
    for epoch in range(20):
        print(f'\nEpoch {epoch+1}/20')
        model.train()
        total_loss = 0
        for images, labels, texts in tqdm(train_loader, desc='Training'):
            images = images.to(device)
            batch_size = images.size(0)
            outputs = model(images)
            T = outputs.size(1)
            
            # CTC Loss inputs
            log_probs = F.log_softmax(outputs, dim=2).permute(1, 0, 2)  # (T, N, C)
            input_lengths = torch.full((batch_size,), T, dtype=torch.long)
            target_lengths = torch.tensor([len(l) for l in labels], dtype=torch.long)
            targets = torch.cat(labels).long()
            
            optimizer.zero_grad()
            loss = criterion(log_probs.cpu(), targets.cpu(), input_lengths, target_lengths)
            
            if torch.isnan(loss) or torch.isinf(loss):
                print(f'Warning: loss is {loss}')
                continue
                
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        
        # Eval
        model.eval()
        total, correct = 0, 0
        total_chars, correct_chars = 0, 0
        with torch.no_grad():
            for images, labels, texts in tqdm(test_loader, desc='Evaluating'):
                images = images.to(device)
                outputs = model(images)
                log_probs = F.log_softmax(outputs, dim=2)
                preds = log_probs.argmax(dim=2).cpu().numpy()
                for i, (pred, true_text) in enumerate(zip(preds, texts)):
                    pred_text = ctc_decode(pred, train_dataset.idx_to_char, blank=0)
                    min_len = min(len(pred_text), len(true_text))
                    matches = sum(1 for j in range(min_len) if pred_text[j:j+1] == true_text[j:j+1])
                    correct_chars += matches
                    total_chars += len(true_text) if len(true_text) > 0 else 1
                    total += 1
                    if pred_text == true_text:
                        correct += 1
        char_acc = correct_chars / total_chars if total_chars > 0 else 0
        seq_acc = correct / total if total > 0 else 0
        print(f'Loss: {avg_loss:.4f}, Char Acc: {char_acc:.4f}, Seq Acc: {seq_acc:.4f}')
        
        history['train_loss'].append(avg_loss)
        history['test_char_acc'].append(char_acc)
        history['test_seq_acc'].append(seq_acc)
        
        if char_acc > best_char_acc:
            best_char_acc = char_acc
            torch.save({
                'epoch': epoch, 'model_state_dict': model.state_dict(),
                'char_acc': char_acc, 'seq_acc': seq_acc,
                'char_to_idx': train_dataset.char_to_idx,
            }, '/home/xufei/tibet_acent/checkpoints_v2/best.pth')
    
    with open('/home/xufei/tibet_acent/checkpoints_v2/history.json', 'w') as f:
        json.dump(history, f, indent=2)
    print(f'\nBest Char Acc: {best_char_acc:.4f}')

if __name__ == '__main__':
    train()

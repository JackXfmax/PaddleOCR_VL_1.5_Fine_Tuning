import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
from tqdm import tqdm
import json
import numpy as np

class TibetanDataset(Dataset):
    def __init__(self, data_dir, split='train', char_to_idx=None, max_length=200, augment=False):
        self.lines_dir = os.path.join(data_dir, 'acent', 'lines')
        self.trans_dir = os.path.join(data_dir, 'acent', 'transcriptions')
        self.max_length = max_length
        self.augment = augment and (split == 'train')
        
        all_images = sorted([f for f in os.listdir(self.lines_dir) if f.endswith('.jpg')])
        
        filtered_images = []
        for img_file in all_images:
            txt_path = os.path.join(self.trans_dir, img_file.replace('.jpg', '.txt'))
            if os.path.exists(txt_path):
                with open(txt_path, 'r', encoding='utf-8') as f:
                    text = f.read().strip()
                    if len(text) <= max_length:
                        filtered_images.append(img_file)
        
        split_idx = int(len(filtered_images) * 0.8)
        self.image_files = filtered_images[:split_idx] if split == 'train' else filtered_images[split_idx:]
        
        print(f'{split} set: {len(self.image_files)} samples')
        
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
        
        w, h = image.size
        aspect = w / h
        new_w = min(int(32 * aspect), 1600)
        new_w = max(new_w, 160)
        
        transform_list = [transforms.Resize((32, new_w))]
        
        # 数据增强（仅训练）
        if self.augment:
            transform_list.extend([
                transforms.RandomApply([
                    transforms.ColorJitter(brightness=0.2, contrast=0.2)
                ], p=0.3),
            ])
        
        transform_list.extend([
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3)
        ])
        
        transform = transforms.Compose(transform_list)
        image = transform(image)
        
        txt_path = os.path.join(self.trans_dir, self.image_files[idx].replace('.jpg', '.txt'))
        with open(txt_path, 'r', encoding='utf-8') as f:
            label = f.read().strip()
        
        label_indices = [self.char_to_idx.get(c, 0) for c in label if c in self.char_to_idx]
        return image, torch.tensor(label_indices, dtype=torch.long), label

def collate_fn(batch):
    max_width = max(img.shape[2] for img, _, _ in batch)
    
    images = []
    labels = []
    texts = []
    for img, label, text in batch:
        if img.shape[2] < max_width:
            pad = torch.zeros(3, 32, max_width - img.shape[2])
            img = torch.cat([img, pad], dim=2)
        images.append(img)
        labels.append(label)
        texts.append(text)
    
    return torch.stack(images), labels, texts

# ResNet风格的CRNN
class CRNN(nn.Module):
    def __init__(self, num_classes):
        super(CRNN, self).__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        
        self.layer1 = self._make_layer(64, 128, 2)
        self.layer2 = self._make_layer(128, 256, 2)
        self.layer3 = self._make_layer(256, 512, 2)
        
        self.conv_final = nn.Sequential(
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        
        # 投影层：2048 -> 512
        self.proj = nn.Linear(2048, 512)
        
        self.rnn = nn.LSTM(512, 256, 3,
                          bidirectional=True,
                          batch_first=True,
                          dropout=0.3)
        self.fc = nn.Linear(512, num_classes)
        
        self._initialize_weights()
    
    def _make_layer(self, in_ch, out_ch, num_blocks):
        layers = []
        # 下采样层
        layers.append(nn.Conv2d(in_ch, out_ch, 3, stride=(2, 1), padding=1))
        layers.append(nn.BatchNorm2d(out_ch))
        layers.append(nn.ReLU(inplace=True))
        
        # 残差块
        for _ in range(num_blocks):
            layers.append(nn.Conv2d(out_ch, out_ch, 3, padding=1))
            layers.append(nn.BatchNorm2d(out_ch))
            layers.append(nn.ReLU(inplace=True))
        
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # x: (B, 3, 32, W)
        x = self.conv1(x)      # (B, 64, 32, W)
        x = self.layer1(x)     # (B, 128, 16, W)
        x = self.layer2(x)     # (B, 256, 8, W)
        x = self.layer3(x)     # (B, 512, 4, W)
        x = self.conv_final(x) # (B, 512, 4, W)
        
        b, c, h, w = x.size()
        x = x.view(b, c * h, w)  # (B, 2048, W)
        x = x.permute(0, 2, 1)   # (B, W, 2048)
        
        # 投影到RNN输入维度
        x = self.proj(x)  # (B, W, 512)
        
        x, _ = self.rnn(x)       # (B, T, 512)
        x = self.fc(x)           # (B, T, num_classes)
        return x

def ctc_decode(pred, idx_to_char, blank=0):
    result = []
    prev = -1
    for p in pred:
        if p != prev and p != blank:
            result.append(idx_to_char.get(p, ''))
        prev = p
    return ''.join(result)

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    
    train_dataset = TibetanDataset('/home/xufei/tibet_acent', split='train', 
                                   max_length=200, augment=True)
    test_dataset = TibetanDataset('/home/xufei/tibet_acent', split='test',
                                   char_to_idx=train_dataset.char_to_idx, max_length=200)
    
    num_classes = len(train_dataset.char_to_idx)
    print(f'Num classes: {num_classes}')
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True,
                              collate_fn=collate_fn, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False,
                             collate_fn=collate_fn, num_workers=4)
    
    model = CRNN(num_classes).to(device)
    print(f'Model parameters: {sum(p.numel() for p in model.parameters()):,}')
    
    criterion = nn.CTCLoss(blank=0, reduction='mean', zero_infinity=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    
    # OneCycle学习率调度
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=0.001, epochs=100,
        steps_per_epoch=len(train_loader),
        pct_start=0.3, anneal_strategy='cos'
    )
    
    os.makedirs('/home/xufei/tibet_acent/checkpoints_v6', exist_ok=True)
    
    history = {'train_loss': [], 'test_char_acc': [], 'test_seq_acc': []}
    best_char_acc = 0
    patience = 20
    no_improve = 0
    
    for epoch in range(100):
        model.train()
        total_loss = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/100')
        for images, labels, texts in pbar:
            images = images.to(device)
            batch_size = images.size(0)
            
            outputs = model(images)
            T = outputs.size(1)
            
            log_probs = F.log_softmax(outputs, dim=2).permute(1, 0, 2)
            input_lengths = torch.full((batch_size,), T, dtype=torch.long)
            target_lengths = torch.tensor([len(l) for l in labels], dtype=torch.long)
            targets = torch.cat(labels).long()
            
            optimizer.zero_grad()
            loss = criterion(log_probs, targets, input_lengths, target_lengths)
            
            if torch.isnan(loss) or torch.isinf(loss):
                continue
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'lr': f'{scheduler.get_last_lr()[0]:.6f}'})
        
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
        
        history['train_loss'].append(avg_loss)
        history['test_char_acc'].append(char_acc)
        history['test_seq_acc'].append(seq_acc)
        
        print(f'Epoch {epoch+1}: Loss={avg_loss:.4f}, CharAcc={char_acc:.4f}, SeqAcc={seq_acc:.4f}')
        
        if char_acc > best_char_acc:
            best_char_acc = char_acc
            no_improve = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'char_acc': char_acc,
                'seq_acc': seq_acc,
                'char_to_idx': train_dataset.char_to_idx,
            }, '/home/xufei/tibet_acent/checkpoints_v6/best.pth')
            print(f'  -> New best! CharAcc={char_acc:.4f}')
        else:
            no_improve += 1
        
        if no_improve >= patience:
            print(f'Early stopping at epoch {epoch+1}')
            break
        
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
            }, f'/home/xufei/tibet_acent/checkpoints_v6/epoch_{epoch+1}.pth')
    
    with open('/home/xufei/tibet_acent/checkpoints_v6/history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f'Best Char Acc: {best_char_acc:.4f}')

if __name__ == '__main__':
    train()

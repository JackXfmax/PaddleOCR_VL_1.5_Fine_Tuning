import torch
ckpt = torch.load('/home/xufei/tibet_acent/checkpoints/crnn_baseline_best.pth', map_location='cpu')
print('Epoch:', ckpt.get('epoch', 'N/A'))
print('Char Acc:', ckpt.get('char_acc', 'N/A'))
print('Seq Acc:', ckpt.get('seq_acc', 'N/A'))

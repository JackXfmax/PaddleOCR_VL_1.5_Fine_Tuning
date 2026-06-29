# 藏文路牌智能识别与无障碍导航系统 · 微信小程序版

基于 PaddleOCR-VL-1.5 + TibetLoRA (B_manual) 微调模型，集成高德地图 API，实现藏文路牌 OCR 识别、中文翻译、实时路线导航、语音播报与无障碍辅助。

## 功能

- 📷 **拍照识别 / 相册选取** — 调用 PaddleOCR-VL-1.5 模型进行藏文 OCR
- 🔤 **藏→中翻译** — 内置 60+ 藏文 Wylie 转写词条词典，支持离线翻译
- 🗺️ **高德地图导航** — GPS 定位 + POI 搜索 + 驾车/步行路线规划
- 🧭 **路牌分类** — 自动识别指路牌、地点标识牌、警示标志等
- 🔊 **语音播报** — 识别完成自动朗读中文翻译
- ♿ **无障碍辅助** — 大字体模式、高对比度、自动播报开关

## 项目结构

```
roadsign_miniapp/
├── app.js                          # 全局应用逻辑
├── app.json                        # 全局配置
├── app.wxss                        # 全局样式
├── project.config.json             # 项目配置
├── sitemap.json
├── pages/
│   └── index/
│       ├── index.js                # 主页面逻辑
│       ├── index.json              # 页面配置
│       ├── index.wxml              # 页面模板
│       └── index.wxss              # 页面样式
└── utils/
    ├── api.js                      # OCR 服务器 API 封装
    ├── amap.js                     # 高德地图 Web API 封装
    └── dict.js                     # 藏汉词典 + 路牌分析
```

## 配置

### 1. 微信小程序 AppID

修改 `project.config.json`：

```json
{
  "appid": "你的小程序AppID"
}
```

### 2. OCR 服务器地址

修改 `app.js` 中的 `ocrServer`（开发环境可通过「不校验合法域名」使用 HTTP）：

```js
globalData: {
  ocrServer: 'http://222.19.225.132:8899',  // OCR 推理服务
  amapKey: '2382d62a9e6919eec1f45a2055370a91',  // 高德地图 Key
}
```

### 3. 服务器域名白名单

在微信公众平台「开发 → 开发管理 → 开发设置 → 服务器域名」中配置：

- **request 合法域名**: 添加 OCR 服务器域名
- **request 合法域名**: 添加 `https://restapi.amap.com`

> 开发阶段可在开发者工具中勾选「不校验合法域名」。

### 4. 高德地图 Key

当前使用 Key: `2382d62a9e6919eec1f45a2055370a91`

如需更换，修改 `app.js` 中的 `amapKey`。

## 使用流程

1. 打开小程序，点击「拍照识别」或「相册选取」
2. 系统自动上传图片到 OCR 服务器进行藏文识别
3. 识别完成后：
   - 展示原始藏文 OCR 结果
   - 展示中文翻译
   - 自动获取 GPS 定位
   - 根据识别的地标搜索高德 POI
   - 规划驾车/步行路线
   - 在地图上显示路线
4. 可点击「高德导航」跳转到高德地图 App 进行实时导航

## 后端服务

详见服务器上的 `roadsign_server.py`：

```bash
# 启动服务（需在服务器上执行）
source /home/xufei/miniconda3/etc/profile.d/conda.sh && conda activate ocr_vlm
nohup python3 /home/xufei/tibetan_ocr_lora/roadsign_server.py > /home/xufei/tibetan_ocr_lora/roadsign_server.log 2>&1 &
```

## 技术栈

| 层级 | 技术 |
|------|------|
| OCR 引擎 | PaddleOCR-VL-1.5 + TibetLoRA (B_manual) |
| 推理服务 | Flask + PaddleFormers (RTX 4090) |
| 地图服务 | 高德 Web API v3 (POI搜索 + 路线规划) |
| 小程序框架 | 微信原生框架 (WXML + WXSS + JS) |
| 词典翻译 | 客户端本地 Wylie → 中文 60+ 词条 |

/**
 * 智能藏文路牌识别与无障碍导航系统 - 微信小程序版
 * 主页面逻辑
 */

const api = require('../../utils/api');
const amap = require('../../utils/amap');
const dict = require('../../utils/dict');
const app = getApp();

Page({
  data: {
    // 服务器状态
    serverStatus: 'dot-yellow',
    statusText: '正在连接服务器…',

    // 图片
    imagePath: '',       // 临时路径，用于显示
    imageBase64: '',     // base64 编码

    // 推理
    inferring: false,
    progressPct: 0,
    progressText: '准备识别…',
    hasResult: false,

    // 识别结果
    rawText: '',
    translated: '',
    signType: '未知',
    signTypeClass: 'guide',
    navSuggestions: [],
    navIcons: ['📍', '🧭', '⚠️', '💡'],
    resultModel: 'PaddleOCR-VL-1.5 + TibetLoRA',
    inferTime: 0,

    // 地图
    mapCenter: {
      latitude: 29.65,    // 默认拉萨
      longitude: 91.11
    },
    mapScale: 14,
    markers: [],
    polylines: [],

    // 路线
    routeDistance: 0,
    routeDuration: 0,
    routeDistanceText: '',
    routeDurationText: '',
    routeMode: '',        // 'driving' or 'walking'
    routeSteps: [],

    // 无障碍
    largeTextActive: false,
    highContrastActive: false,
    autoVoiceActive: true,

    // Toast
    toastShow: false,
    toastMsg: '',
    toastTimer: null,

    // 进度定时器
    progressTimer: null,
  },

  onLoad: function () {
    // 恢复设置
    this.setData({
      largeTextActive: app.globalData.largeText,
      highContrastActive: app.globalData.highContrast,
      autoVoiceActive: app.globalData.autoVoice
    });

    // 应用设置
    if (app.globalData.largeText) this._applyLargeText(true);
    if (app.globalData.highContrast) this._applyContrast(true);

    // 检测连接
    this.checkHealth();
  },

  onShow: function () {
    // 每次显示时检测连接（如果之前没有结果）
    if (!this.data.hasResult) {
      this.checkHealth();
    }
  },

  // ═══════════════════════════════════════════════
  //  服务器健康检测
  // ═══════════════════════════════════════════════
  checkHealth: function () {
    this.setData({
      serverStatus: 'dot-yellow',
      statusText: '检测中…'
    });

    api.checkHealth()
      .then(data => {
        if (data.model_loaded) {
          this.setData({
            serverStatus: 'dot-green',
            statusText: '服务正常 · 模型已就绪'
          });
        } else {
          this.setData({
            serverStatus: 'dot-yellow',
            statusText: '服务运行中 · 模型加载中…'
          });
        }
      })
      .catch(err => {
        this.setData({
          serverStatus: 'dot-red',
          statusText: '无法连接服务器'
        });
        console.error('Health check failed:', err);
      });
  },

  // ═══════════════════════════════════════════════
  //  拍照
  // ═══════════════════════════════════════════════
  takePhoto: function () {
    const that = this;
    wx.chooseMedia({
      count: 1,
      mediaType: ['image'],
      sourceType: ['camera'],
      camera: 'back',
      success(res) {
        const tempFilePath = res.tempFiles[0].tempFilePath;
        that._processImage(tempFilePath);
      },
      fail(err) {
        if (err.errMsg.indexOf('cancel') < 0) {
          that._toast('无法使用相机: ' + err.errMsg);
        }
      }
    });
  },

  // ═══════════════════════════════════════════════
  //  从相册选择
  // ═══════════════════════════════════════════════
  chooseImage: function () {
    const that = this;
    wx.chooseMedia({
      count: 1,
      mediaType: ['image'],
      sourceType: ['album'],
      success(res) {
        const tempFilePath = res.tempFiles[0].tempFilePath;
        that._processImage(tempFilePath);
      },
      fail(err) {
        if (err.errMsg.indexOf('cancel') < 0) {
          that._toast('选择图片失败');
        }
      }
    });
  },

  // ═══════════════════════════════════════════════
  //  处理图片：压缩 + 转 base64
  // ═══════════════════════════════════════════════
  _processImage: function (filePath) {
    const that = this;

    // 显示预览
    this.setData({
      imagePath: filePath,
      hasResult: false,
      markers: [],
      polylines: [],
      routeDistance: 0,
      routeDuration: 0,
      routeSteps: []
    });

    // 压缩图片（微信小程序限制 base64 大小，服务端也需要合理大小）
    wx.compressImage({
      src: filePath,
      quality: 85,
      success(res) {
        that._toBase64(res.tempFilePath);
      },
      fail() {
        // 压缩失败就用原图
        that._toBase64(filePath);
      }
    });
  },

  _toBase64: function (filePath) {
    const that = this;
    wx.getFileSystemManager().readFile({
      filePath: filePath,
      encoding: 'base64',
      success(res) {
        that.setData({ imageBase64: res.data });
        // 自动开始识别
        that.doInfer();
      },
      fail(err) {
        that._toast('读取图片失败: ' + err.errMsg);
      }
    });
  },

  // ═══════════════════════════════════════════════
  //  OCR 推理 + 导航规划
  // ═══════════════════════════════════════════════
  doInfer: function () {
    if (!this.data.imageBase64) {
      this._toast('请先拍摄或选择路牌照片');
      return;
    }

    this.setData({
      inferring: true,
      progressPct: 0,
      progressText: '模型推理中…',
      hasResult: false
    });

    // 进度动画
    let elapsed = 0;
    const progressTimer = setInterval(() => {
      elapsed += 200;
      const pct = Math.min(85, 85 * elapsed / 15000);
      this.setData({
        progressPct: pct,
        progressText: 'AI 模型推理中… ' + (elapsed / 1000).toFixed(1) + 's'
      });
    }, 200);
    this._progressTimer = progressTimer;

    const t0 = Date.now();

    // 调用 OCR 服务
    api.ocrRoadsign(this.data.imageBase64)
      .then(ocrResult => {
        const elapsed = ((Date.now() - t0) / 1000).toFixed(1);

        clearInterval(progressTimer);
        this.setData({
          progressPct: 100,
          progressText: '识别完成 · ' + elapsed + 's',
        });

        // 本地增强：用内置词典补充翻译
        const localTranslated = dict.translateWylie(ocrResult.raw_text);
        const localInfo = dict.analyzeRoadsign(ocrResult.raw_text);

        // 合并服务器和本地结果
        const mergedDirections = [...new Set([
          ...(ocrResult.directions || []),
          ...localInfo.directions
        ])];
        const mergedLandmarks = [...new Set([
          ...(ocrResult.landmarks || []),
          ...localInfo.landmarks
        ])];
        const mergedWarnings = [...new Set([
          ...(ocrResult.warnings || []),
          ...localInfo.warnings
        ])];

        // 生成导航建议
        const suggestions = [];
        if (mergedDirections.length > 0) {
          const dirs = mergedDirections.join(' → ');
          if (mergedLandmarks.length > 0) {
            suggestions.push('前往 ' + mergedLandmarks.slice(0, 3).join('、') + ' 方向: ' + dirs);
          } else {
            suggestions.push('行驶方向: ' + dirs);
          }
        }
        if (mergedLandmarks.length > 0 && mergedDirections.length === 0) {
          suggestions.push('当前位置附近: ' + mergedLandmarks.slice(0, 3).join('、'));
        }
        mergedWarnings.forEach(w => suggestions.push(w));
        if (suggestions.length === 0) {
          suggestions.push('已识别路牌文字，请根据实际情况判断行驶方向');
        }

        // 使用服务器返回的翻译，如果不够好就用本地翻译
        const translated = ocrResult.translated && ocrResult.translated !== ocrResult.raw_text
          ? ocrResult.translated
          : localTranslated;

        const signType = ocrResult.sign_type || localInfo.type;

        this.setData({
          inferring: false,
          hasResult: true,
          rawText: ocrResult.raw_text || '',
          translated: translated,
          signType: signType,
          signTypeClass: this._signTypeClass(signType),
          navSuggestions: suggestions,
          resultModel: 'PaddleOCR-VL-1.5 + TibetLoRA (B_manual)',
          inferTime: elapsed,
        });

        // 保存到全局
        app.globalData.lastResult = {
          raw_text: ocrResult.raw_text,
          translated: translated,
          sign_type: signType,
          landmarks: mergedLandmarks
        };

        // 开始导航规划（高德地图）
        this._planRoute(mergedLandmarks);

        // 自动语音播报
        if (this.data.autoVoiceActive) {
          setTimeout(() => this.readAloud(), 800);
        }

        this._toast('识别完成！耗时 ' + elapsed + 's');
      })
      .catch(err => {
        clearInterval(progressTimer);
        this.setData({
          inferring: false,
          progressPct: 0,
          progressText: '识别失败',
          hasResult: false,
          rawText: '识别失败: ' + err.message,
        });
        this._toast('识别失败: ' + err.message);
        console.error('OCR failed:', err);
      });
  },

  // ═══════════════════════════════════════════════
  //  高德地图路线规划
  // ═══════════════════════════════════════════════
  _planRoute: function (landmarks) {
    if (!landmarks || landmarks.length === 0) {
      // 仍然尝试获取用户位置来显示在地图上
      amap.getCurrentLocation().then(loc => {
        this.setData({
          mapCenter: { latitude: loc.latitude, longitude: loc.longitude },
          mapScale: 14,
          markers: [{
            id: 0,
            latitude: loc.latitude,
            longitude: loc.longitude,
            width: 30,
            height: 30,
            callout: {
              content: '📍 当前位置',
              color: '#ffffff',
              fontSize: 11,
              borderRadius: 6,
              bgColor: '#c0392b',
              padding: 4,
              display: 'ALWAYS'
            }
          }]
        });
      }).catch(() => {});
      return;
    }

    // 调用完整的导航规划
    amap.planNavigation(landmarks).then(navResult => {
      // 更新地图中心
      const centerLat = navResult.destination
        ? (navResult.currentLocation.latitude + navResult.destination.latitude) / 2
        : navResult.currentLocation.latitude;
      const centerLng = navResult.destination
        ? (navResult.currentLocation.longitude + navResult.destination.longitude) / 2
        : navResult.currentLocation.longitude;

      const updateData = {
        mapCenter: { latitude: centerLat, longitude: centerLng },
        mapScale: 13,
        markers: navResult.markers,
        routeDistance: navResult.distance,
        routeDuration: navResult.duration,
        routeDistanceText: amap.formatDistance(navResult.distance),
        routeDurationText: amap.formatDuration(navResult.duration),
        routeMode: navResult.mode || 'driving',
        routeSteps: navResult.steps || [],
        polylines: []
      };

      // 如果有路线点，构建 polyline
      if (navResult.routePoints.length > 0) {
        updateData.polylines = [{
          points: navResult.routePoints,
          color: '#1a6b4b',
          width: 6,
          dottedLine: false,
          arrowLine: true,
          borderColor: '#ffffff',
          borderWidth: 1
        }];
      }

      this.setData(updateData);

      // 将地图视野调整到包含所有 markers
      if (navResult.routePoints.length > 0) {
        const mapCtx = wx.createMapContext('navMap', this);
        mapCtx.includePoints({
          points: navResult.routePoints,
          padding: [60, 40, 60, 40]
        });
      }

      if (navResult.error) {
        this._toast(navResult.error);
      }
    }).catch(err => {
      console.error('路线规划失败:', err);
      this._toast('路线规划失败，请查看文字导航');
    });
  },

  // ═══════════════════════════════════════════════
  //  路牌类型 → CSS 类名
  // ═══════════════════════════════════════════════
  _signTypeClass: function (type) {
    const map = {
      '指路牌': 'guide',
      '地点标识牌': 'place',
      '警示/禁令标志': 'warn',
      '道路名称牌': 'road',
      '行政区划牌': 'district'
    };
    return map[type] || 'guide';
  },

  // ═══════════════════════════════════════════════
  //  地图事件
  // ═══════════════════════════════════════════════
  onMarkerTap: function (e) {
    console.log('Marker tapped:', e.detail.markerId);
  },

  onRegionChange: function (e) {
    // 地图区域变化
  },

  // ═══════════════════════════════════════════════
  //  操作按钮
  // ═══════════════════════════════════════════════
  copyResult: function () {
    const result = app.globalData.lastResult;
    if (!result) return;
    const text = '藏文原文:\n' + result.raw_text + '\n\n中文翻译:\n' + result.translated;
    wx.setClipboardData({
      data: text,
      success: () => this._toast('已复制到剪贴板')
    });
  },

  readAloud: function () {
    const result = app.globalData.lastResult;
    if (!result) return;

    const text = result.translated || result.raw_text;
    if (!text) return;

    // 微信小程序没有内置 TTS，使用插件或服务
    // 这里使用 wx.createInnerAudioContext + TTS 服务的方式
    // 简化版本：弹窗提示
    wx.showModal({
      title: '语音播报',
      content: text,
      showCancel: false,
      confirmText: '知道了',
      confirmColor: '#c0392b'
    });
  },

  openInAmap: function () {
    const result = app.globalData.lastResult;
    if (!result || !result.landmarks || result.landmarks.length === 0) {
      this._toast('未识别到目的地');
      return;
    }

    const dest = result.landmarks[0];

    // 尝试打开高德地图 App
    wx.openLocation({
      latitude: this.data.mapCenter.latitude,
      longitude: this.data.mapCenter.longitude,
      name: dest,
      address: result.translated || dest,
      scale: 15,
      fail() {
        // 如果用户没有安装或打开失败，使用 navigateToMiniProgram
        wx.showModal({
          title: '打开导航',
          content: '是否使用微信内置地图查看位置？',
          success(res) {
            if (res.confirm) {
              wx.openLocation({
                latitude: this.data.mapCenter.latitude,
                longitude: this.data.mapCenter.longitude,
                name: dest,
                scale: 15
              });
            }
          }
        });
      }
    });
  },

  // ═══════════════════════════════════════════════
  //  无障碍功能
  // ═══════════════════════════════════════════════
  toggleLargeText: function () {
    const active = !this.data.largeTextActive;
    this.setData({ largeTextActive: active });
    app.globalData.largeText = active;
    wx.setStorageSync('roadsign_largetext', active ? '1' : '0');
    this._applyLargeText(active);
    this._toast(active ? '已开启大字体模式' : '已关闭大字体模式');
  },

  _applyLargeText: function (active) {
    const pages = getCurrentPages();
    const page = pages[pages.length - 1];
    // 通过动态修改页面类名实现
    if (active) {
      wx.setPageStyle({
        style: { overflow: 'scroll' }
      });
    }
  },

  toggleContrast: function () {
    const active = !this.data.highContrastActive;
    this.setData({ highContrastActive: active });
    app.globalData.highContrast = active;
    wx.setStorageSync('roadsign_contrast', active ? '1' : '0');
    this._applyContrast(active);
    this._toast(active ? '已开启高对比度模式' : '已关闭高对比度模式');
  },

  _applyContrast: function (active) {
    // 小程序可以通过 wx.setPageStyle 或动态修改
  },

  toggleAutoVoice: function () {
    const active = !this.data.autoVoiceActive;
    this.setData({ autoVoiceActive: active });
    app.globalData.autoVoice = active;
    wx.setStorageSync('roadsign_autovoice', active ? '1' : '0');
    this._toast(active ? '自动播报: 开启' : '自动播报: 关闭');
  },

  // ═══════════════════════════════════════════════
  //  重置
  // ═══════════════════════════════════════════════
  resetAll: function () {
    if (this._progressTimer) {
      clearInterval(this._progressTimer);
      this._progressTimer = null;
    }

    this.setData({
      imagePath: '',
      imageBase64: '',
      inferring: false,
      progressPct: 0,
      progressText: '准备识别…',
      hasResult: false,
      rawText: '',
      translated: '',
      signType: '未知',
      signTypeClass: 'guide',
      navSuggestions: [],
      inferTime: 0,
      markers: [],
      polylines: [],
      routeDistance: 0,
      routeDuration: 0,
      routeDistanceText: '',
      routeDurationText: '',
      routeSteps: [],
      mapCenter: { latitude: 29.65, longitude: 91.11 },
      mapScale: 14,
    });
  },

  // ═══════════════════════════════════════════════
  //  Toast
  // ═══════════════════════════════════════════════
  _toast: function (msg) {
    if (this.data.toastTimer) {
      clearTimeout(this.data.toastTimer);
    }
    this.setData({ toastShow: true, toastMsg: msg });
    const timer = setTimeout(() => {
      this.setData({ toastShow: false });
    }, 2500);
    this.setData({ toastTimer: timer });
  },

  // ═══════════════════════════════════════════════
  //  生命周期
  // ═══════════════════════════════════════════════
  onUnload: function () {
    if (this._progressTimer) {
      clearInterval(this._progressTimer);
    }
    if (this.data.toastTimer) {
      clearTimeout(this.data.toastTimer);
    }
  }
});

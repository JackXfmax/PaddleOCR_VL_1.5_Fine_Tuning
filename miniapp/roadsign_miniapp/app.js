// app.js
App({
  onLaunch: function () {
    // 恢复本地存储的设置
    const voiceMode = wx.getStorageSync('roadsign_voice');
    const largeText = wx.getStorageSync('roadsign_largetext');
    const contrast = wx.getStorageSync('roadsign_contrast');
    const autoVoice = wx.getStorageSync('roadsign_autovoice');

    this.globalData.voiceMode = voiceMode === '' ? true : voiceMode;
    this.globalData.largeText = largeText === '1';
    this.globalData.highContrast = contrast === '1';
    this.globalData.autoVoice = autoVoice === '' ? true : autoVoice;
  },

  globalData: {
    // OCR 服务器地址
    ocrServer: 'http://222.19.225.132:8899',

    // 高德地图 Web API Key（微信小程序专用）
    amapKey: '2382d62a9e6919eec1f45a2055370a91',

    // 设置
    voiceMode: true,
    largeText: false,
    highContrast: false,
    autoVoice: true,

    // 当前结果
    lastResult: null,
    currentImage: null,
  }
});

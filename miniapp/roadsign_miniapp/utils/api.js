/**
 * OCR 服务器 API 封装
 */

const app = getApp();

/**
 * 检测服务器健康状态
 */
function checkHealth() {
  return new Promise((resolve, reject) => {
    wx.request({
      url: app.globalData.ocrServer + '/health',
      method: 'GET',
      timeout: 8000,
      success: (res) => {
        resolve(res.data);
      },
      fail: (err) => {
        reject(err);
      }
    });
  });
}

/**
 * OCR 识别路牌图片
 * @param {string} imageBase64 - 图片 base64 编码（不含前缀）
 * @returns {Promise<Object>} OCR 结果
 */
function ocrRoadsign(imageBase64) {
  return new Promise((resolve, reject) => {
    wx.request({
      url: app.globalData.ocrServer + '/ocr',
      method: 'POST',
      header: { 'Content-Type': 'application/json' },
      data: { image: imageBase64 },
      timeout: 120000,
      success: (res) => {
        if (res.statusCode === 200 && !res.data.error) {
          resolve(res.data);
        } else {
          reject(new Error(res.data.error || 'OCR 识别失败'));
        }
      },
      fail: (err) => {
        reject(new Error('网络请求失败: ' + err.errMsg));
      }
    });
  });
}

/**
 * 翻译接口（独立）
 */
function translateText(text) {
  return new Promise((resolve, reject) => {
    wx.request({
      url: app.globalData.ocrServer + '/translate',
      method: 'POST',
      header: { 'Content-Type': 'application/json' },
      data: { text: text },
      timeout: 15000,
      success: (res) => {
        resolve(res.data);
      },
      fail: (err) => {
        reject(err);
      }
    });
  });
}

/**
 * 导航分析接口（纯文本）
 */
function navigateText(text) {
  return new Promise((resolve, reject) => {
    wx.request({
      url: app.globalData.ocrServer + '/navigate',
      method: 'POST',
      header: { 'Content-Type': 'application/json' },
      data: { text: text },
      timeout: 15000,
      success: (res) => {
        resolve(res.data);
      },
      fail: (err) => {
        reject(err);
      }
    });
  });
}

module.exports = {
  checkHealth,
  ocrRoadsign,
  translateText,
  navigateText
};

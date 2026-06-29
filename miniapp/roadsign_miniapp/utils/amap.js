/**
 * 高德地图 Web API 封装（微信小程序专用）
 * Key: 2382d62a9e6919eec1f45a2055370a91
 */

const app = getApp();
const AMAP_KEY = app.globalData.amapKey;
const AMAP_BASE = 'https://restapi.amap.com/v3';

/**
 * 获取用户当前位置
 * @returns {Promise<{latitude: number, longitude: number}>}
 */
function getCurrentLocation() {
  return new Promise((resolve, reject) => {
    wx.getLocation({
      type: 'gcj02',  // 高德/国测局坐标系
      success: (res) => {
        resolve({
          latitude: res.latitude,
          longitude: res.longitude
        });
      },
      fail: (err) => {
        // 如果用户拒绝授权，返回默认坐标（拉萨市中心）
        if (err.errMsg.indexOf('auth deny') >= 0 || err.errMsg.indexOf('authorize') >= 0) {
          console.warn('用户拒绝位置授权，使用默认位置');
          resolve({
            latitude: 29.65,
            longitude: 91.11,
            isDefault: true
          });
        } else {
          reject(err);
        }
      }
    });
  });
}

/**
 * 逆地理编码：坐标 → 地址
 */
function reverseGeocode(lat, lng) {
  return new Promise((resolve, reject) => {
    wx.request({
      url: AMAP_BASE + '/geocode/regeo',
      data: {
        key: AMAP_KEY,
        location: lng + ',' + lat,
        extensions: 'base'
      },
      success: (res) => {
        if (res.data.status === '1') {
          resolve(res.data.regeocode);
        } else {
          reject(new Error('逆地理编码失败: ' + res.data.info));
        }
      },
      fail: reject
    });
  });
}

/**
 * POI 搜索：地名 → 坐标
 * @param {string} keyword - 搜索关键词
 * @param {string} city - 城市名（可选，默认拉萨）
 */
function searchPOI(keyword, city) {
  return new Promise((resolve, reject) => {
    wx.request({
      url: AMAP_BASE + '/place/text',
      data: {
        key: AMAP_KEY,
        keywords: keyword,
        city: city || '拉萨',
        offset: 5,
        page: 1,
        extensions: 'all'
      },
      success: (res) => {
        if (res.data.status === '1' && res.data.pois && res.data.pois.length > 0) {
          resolve(res.data.pois);
        } else {
          resolve([]);
        }
      },
      fail: reject
    });
  });
}

/**
 * 驾车路径规划
 * @param {{lat:number, lng:number}} origin - 起点
 * @param {{lat:number, lng:number}} destination - 终点
 */
function drivingRoute(origin, destination) {
  return new Promise((resolve, reject) => {
    wx.request({
      url: AMAP_BASE + '/direction/driving',
      data: {
        key: AMAP_KEY,
        origin: origin.lng + ',' + origin.lat,
        destination: destination.lng + ',' + destination.lat,
        strategy: 0,       // 速度优先
        extensions: 'all'  // 返回详细信息
      },
      success: (res) => {
        if (res.data.status === '1' && res.data.route && res.data.route.paths) {
          resolve(res.data.route.paths[0]);
        } else {
          reject(new Error('路线规划失败: ' + (res.data.info || '无可用路线')));
        }
      },
      fail: reject
    });
  });
}

/**
 * 步行路径规划
 */
function walkingRoute(origin, destination) {
  return new Promise((resolve, reject) => {
    wx.request({
      url: AMAP_BASE + '/direction/walking',
      data: {
        key: AMAP_KEY,
        origin: origin.lng + ',' + origin.lat,
        destination: destination.lng + ',' + destination.lat
      },
      success: (res) => {
        if (res.data.status === '1' && res.data.route && res.data.route.paths) {
          resolve(res.data.route.paths[0]);
        } else {
          reject(new Error('步行路线规划失败'));
        }
      },
      fail: reject
    });
  });
}

/**
 * 解码高德地图折线编码为坐标数组
 * 用于 <map> 组件的 polyline
 */
function decodePolyline(encoded) {
  if (!encoded) return [];
  let len = encoded.length;
  let index = 0;
  let points = [];
  let lat = 0, lng = 0;

  while (index < len) {
    let b, shift = 0, result = 0;
    do {
      b = encoded.charCodeAt(index++) - 63;
      result |= (b & 0x1f) << shift;
      shift += 5;
    } while (b >= 0x20);
    let dlat = (result & 1) ? ~(result >> 1) : (result >> 1);
    lat += dlat;

    shift = 0;
    result = 0;
    do {
      b = encoded.charCodeAt(index++) - 63;
      result |= (b & 0x1f) << shift;
      shift += 5;
    } while (b >= 0x20);
    let dlng = (result & 1) ? ~(result >> 1) : (result >> 1);
    lng += dlng;

    points.push({
      latitude: lat / 1e6,
      longitude: lng / 1e6
    });
  }
  return points;
}

/**
 * 把高德路径步骤转为 polyline 点集
 */
function stepsToPolyline(steps) {
  let allPoints = [];
  steps.forEach(step => {
    const pts = decodePolyline(step.polyline);
    allPoints = allPoints.concat(pts);
  });
  return allPoints;
}

/**
 * 完整的导航规划流程
 * 1. 获取当前位置
 * 2. 根据OCR识别的地名搜索POI
 * 3. 规划驾车/步行路线
 *
 * @param {string[]} landmarks - OCR识别出的地标列表
 * @returns {Promise<Object>} 完整导航信息
 */
async function planNavigation(landmarks) {
  const result = {
    currentLocation: null,
    destination: null,
    route: null,
    routePoints: [],
    markers: [],
    address: '',
    distance: 0,
    duration: 0,
    steps: [],
    error: null
  };

  try {
    // 1. 获取当前位置
    result.currentLocation = await getCurrentLocation();
  } catch (e) {
    result.error = '无法获取位置信息';
    return result;
  }

  // 添加当前位置标记（使用默认蓝色标记点）
  result.markers.push({
    id: 0,
    latitude: result.currentLocation.latitude,
    longitude: result.currentLocation.longitude,
    width: 30,
    height: 30,
    callout: {
      content: '📍 当前位置',
      color: '#ffffff',
      fontSize: 11,
      borderRadius: 8,
      bgColor: '#c0392b',
      padding: 6,
      display: 'ALWAYS'
    }
  });

  if (!landmarks || landmarks.length === 0) {
    return result;
  }

  // 2. 搜索目的地 POI
  let pois = [];
  for (const lm of landmarks.slice(0, 3)) {
    try {
      const found = await searchPOI(lm);
      if (found.length > 0) {
        pois = pois.concat(found);
      }
    } catch (e) {
      console.warn('POI搜索失败:', lm, e);
    }
  }

  if (pois.length > 0) {
    // 取第一个有效POI作为目的地
    const dest = pois[0];
    const destLocation = dest.location.split(',');
    result.destination = {
      name: dest.name,
      address: dest.address,
      latitude: parseFloat(destLocation[1]),
      longitude: parseFloat(destLocation[0])
    };
    result.address = dest.address || dest.name;

    // 添加目的地标记（使用默认红色标记点）
    result.markers.push({
      id: 1,
      latitude: result.destination.latitude,
      longitude: result.destination.longitude,
      width: 30,
      height: 30,
      callout: {
        content: '🏁 ' + dest.name,
        color: '#ffffff',
        fontSize: 11,
        borderRadius: 8,
        bgColor: '#1a6b4b',
        padding: 6,
        display: 'ALWAYS'
      }
    });

    // 3. 规划路线
    try {
      const route = await drivingRoute(
        { lat: result.currentLocation.latitude, lng: result.currentLocation.longitude },
        { lat: result.destination.latitude, lng: result.destination.longitude }
      );
      result.route = route;
      result.distance = parseInt(route.distance);
      result.duration = parseInt(route.duration);

      // 解析路线步骤
      if (route.steps) {
        result.routePoints = stepsToPolyline(route.steps);
        result.steps = route.steps.map((s, i) => ({
          index: i + 1,
          instruction: s.instruction,
          road: s.road,
          distance: s.distance,
          duration: s.duration
        }));
      }
    } catch (e) {
      // 驾车路线失败，尝试步行
      try {
        const walkRoute = await walkingRoute(
          { lat: result.currentLocation.latitude, lng: result.currentLocation.longitude },
          { lat: result.destination.latitude, lng: result.destination.longitude }
        );
        result.route = walkRoute;
        result.distance = parseInt(walkRoute.distance);
        result.duration = parseInt(walkRoute.duration);
        result.mode = 'walking';
        if (walkRoute.steps) {
          result.routePoints = stepsToPolyline(walkRoute.steps);
          result.steps = walkRoute.steps.map((s, i) => ({
            index: i + 1,
            instruction: s.instruction,
            road: s.road,
            distance: s.distance,
            duration: s.duration
          }));
        }
      } catch (e2) {
        result.error = '路线规划失败，请查看文字导航';
      }
    }
  } else {
    result.error = '未找到目的地坐标，仅显示文字导航';
  }

  return result;
}

/**
 * 格式化距离
 */
function formatDistance(meters) {
  if (!meters) return '';
  if (meters < 1000) return meters + '米';
  return (meters / 1000).toFixed(1) + '公里';
}

/**
 * 格式化时间
 */
function formatDuration(seconds) {
  if (!seconds) return '';
  if (seconds < 60) return seconds + '秒';
  if (seconds < 3600) return Math.floor(seconds / 60) + '分钟';
  const h = Math.floor(seconds / 3600);
  const m = Math.floor((seconds % 3600) / 60);
  return h + '小时' + (m > 0 ? m + '分钟' : '');
}

module.exports = {
  getCurrentLocation,
  reverseGeocode,
  searchPOI,
  drivingRoute,
  walkingRoute,
  decodePolyline,
  stepsToPolyline,
  planNavigation,
  formatDistance,
  formatDuration
};

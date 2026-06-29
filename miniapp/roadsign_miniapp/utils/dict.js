/**
 * 藏文路牌常用词汇 Wylie → 中文 对照表
 * 客户端本地翻译，无需网络请求
 */

const WYLIE_CN_DICT = {
  // 城市 地区
  "lha sa": "拉萨",
  "gzhis ka rtse": "日喀则",
  "chab mdo": "昌都",
  "nying khri": "林芝",
  "nag chu": "那曲",
  "snga ris": "阿里",
  "lho kha": "山南",
  "rkang tshugs": "岗堆",
  "gnye mo": "尼木",
  "gong dkar": "贡嘎",
  "sne gdong": "乃东",

  // 道路相关
  "gtsang po": "藏布",
  "chu": "河",
  "lam": "路",
  "lam ka": "道路",
  "lam chen": "大路/主路",
  "lam chung": "小巷",
  "sgang lam": "上坡路",
  "khrong khyer": "城市",
  "grong gseb": "乡镇",
  "grong tsho": "村庄",

  // 地点/设施
  "gsol ras": "餐厅",
  "mgron khang": "酒店/宾馆",
  "sman khang": "医院",
  "slob grwa": "学校",
  "sgyel khang": "厕所",
  "tshong khang": "商店",
  "sprul sku": "佛塔",
  "dgon pa": "寺院",
  "rnyog khang": "加油站",
  "rta bab": "停车场",
  "lcags lam": "铁路",
  "gnam gru thang": "机场",
  "me 'khor": "火车",
  "rlangs 'khor": "汽车",
  "dngul khang": "银行",
  "sbrid khang": "邮局",
  "bdag skyong": "公安局",
  "me gsod": "消防",
  "lto zan": "食物",
  "chang khang": "酒吧",

  // 道路类型
  "nye lam": "附近道路",
  "rkang lam": "步行道",
  "rgyal lam": "国道",
  "zhing lam": "公路",
  "sa khul": "地区",
  "rdzong": "县",

  // 方向
  "byang": "北",
  "lho": "南",
  "shar": "东",
  "nub": "西",
  "g.yon": "左",
  "g.yas": "右",
  "mdun": "前",
  "rgyab": "后",
  "gong": "上",
  "'og": "下",
  "bar": "中间",

  // 交通标志
  "bkag": "禁止",
  "chog": "允许",
  "nyen": "危险",
  "dal": "慢",
  "mgyogs": "快",
  "thog": "停",
  "gtong": "通行",
  "gcod": "禁止通行",
  "gyang": "注意",
  "sgrig": "规则",
  "sgrig lam": "交通规则",
  "lam sgrig": "交通信号",
};

/**
 * 本地 Wylie → 中文翻译（基于词典替换）
 */
function translateWylie(text) {
  if (!text) return '';
  let result = text;
  // 按 key 长度降序排列，避免短词误匹配
  const entries = Object.entries(WYLIE_CN_DICT).sort((a, b) => b[0].length - a[0].length);
  for (const [wy, cn] of entries) {
    const regex = new RegExp(wy.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'), 'gi');
    result = result.replace(regex, cn);
  }
  return result;
}

/**
 * 分析路牌内容
 */
function analyzeRoadsign(ocrText) {
  if (!ocrText) return { type: '未知', directions: [], landmarks: [], warnings: [] };

  const textLower = ocrText.toLowerCase();
  const info = { type: '未知', directions: [], landmarks: [], warnings: [] };

  // 方向检测
  const dirPairs = [
    ['byang', '北'], ['lho', '南'], ['shar', '东'], ['nub', '西'],
    ['g.yon', '左'], ['g.yas', '右'], ['mdun', '前'], ['rgyab', '后'],
  ];
  dirPairs.forEach(([wy, cn]) => {
    if (textLower.indexOf(wy) >= 0) info.directions.push(cn);
  });

  // 地标检测
  const landmarkPairs = [
    ['dgon pa', '寺院'], ['sprul sku', '佛塔'], ['slob grwa', '学校'],
    ['sman khang', '医院'], ['mgron khang', '酒店'], ['gsol ras', '餐厅'],
    ['tshong khang', '商店'], ['dngul khang', '银行'], ['rnyog khang', '加油站'],
    ['rta bab', '停车场'], ['gnam gru thang', '机场'], ['lcags lam', '铁路'],
    ["me 'khor", '火车站'], ['sgyel khang', '公共厕所'],
    ['bdag skyong', '公安局'], ['me gsod', '消防站'],
  ];
  landmarkPairs.forEach(([wy, cn]) => {
    if (textLower.indexOf(wy) >= 0) info.landmarks.push(cn);
  });

  // 警告/限制
  if (['bkag', 'gcod'].some(w => textLower.indexOf(w) >= 0)) info.warnings.push('禁止通行');
  if (['nyen', 'gyang'].some(w => textLower.indexOf(w) >= 0)) info.warnings.push('危险/注意区域');
  if (textLower.indexOf('dal') >= 0) info.warnings.push('减速慢行');
  if (['lam sgrig', 'sgrig lam'].some(w => textLower.indexOf(w) >= 0)) info.warnings.push('注意交通规则');

  // 路牌类型推断
  if (info.directions.length > 0 && info.landmarks.length > 0) {
    info.type = '指路牌';
  } else if (info.landmarks.length > 0 && info.directions.length === 0) {
    info.type = '地点标识牌';
  } else if (info.warnings.length > 0) {
    info.type = '警示/禁令标志';
  } else if (['lam', 'chu', 'zhing lam', 'rgyal lam'].some(w => textLower.indexOf(w) >= 0)) {
    info.type = '道路名称牌';
  } else if (['rdzong', 'grong', 'sa khul'].some(w => textLower.indexOf(w) >= 0)) {
    info.type = '行政区划牌';
  }

  return info;
}

module.exports = {
  WYLIE_CN_DICT,
  translateWylie,
  analyzeRoadsign
};

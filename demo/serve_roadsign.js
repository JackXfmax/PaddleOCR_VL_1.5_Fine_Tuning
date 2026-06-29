const http = require('http');
const fs = require('fs');
const path = require('path');
const root = __dirname;

const mime = {
  html: 'text/html', css: 'text/css', js: 'application/javascript',
  png: 'image/png', jpg: 'image/jpeg', jpeg: 'image/jpeg',
  svg: 'image/svg+xml', json: 'application/json',
  ico: 'image/x-icon'
};

http.createServer((req, res) => {
  let f = req.url === '/' ? '/roadsign.html' : req.url;
  f = path.join(root, f.replace(/^\//, ''));
  try {
    const d = fs.readFileSync(f);
    const ext = path.extname(f).slice(1);
    res.writeHead(200, {
      'Content-Type': mime[ext] || 'application/octet-stream',
      'Access-Control-Allow-Origin': '*'
    });
    res.end(d);
  } catch(e) {
    res.writeHead(404);
    res.end('404 Not Found');
  }
}).listen(8080, () => console.log('RoadSign UI: http://localhost:8080'));

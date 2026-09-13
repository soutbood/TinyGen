// index_html.h
// Web frontend served by the ESP32 (all inline, no external assets)
const char INDEX_HTML[] PROGMEM = R"rawliteral(
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>TinyGen</title>
<style>
:root{
  --bg:#f4f5f7; --card:#ffffff; --text:#1f2430; --muted:#69707d;
  --border:#e3e5ea; --accent:#3b6ef6; --accent-h:#2f5ad0;
  --ok:#177245; --ok-bg:#e9f7ef; --err:#b3372a; --err-bg:#fdecea;
  --busy:#8a6100; --busy-bg:#fff4d4;
}
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
background:var(--bg);color:var(--text);min-height:100vh;
display:flex;flex-direction:column;align-items:center;padding:48px 20px 32px}
header{text-align:center;margin-bottom:28px}
h1{font-size:28px;font-weight:650;letter-spacing:-.02em}
.sub{color:var(--muted);font-size:14px;margin-top:6px}
.card{width:100%;max-width:860px;background:var(--card);border:1px solid var(--border);
border-radius:12px;padding:28px;
box-shadow:0 1px 3px rgba(16,24,40,.06),0 8px 24px rgba(16,24,40,.05)}
.label{display:block;font-size:12px;font-weight:600;color:var(--muted);
text-transform:uppercase;letter-spacing:.05em;margin-bottom:10px}
.row{display:flex;gap:10px;margin-bottom:24px}
input{flex:1;padding:11px 14px;font:inherit;font-size:15px;color:var(--text);
background:#fbfbfc;border:1px solid var(--border);border-radius:8px}
input:focus{outline:none;border-color:var(--accent);box-shadow:0 0 0 3px rgba(59,110,246,.15)}
button{padding:11px 22px;font:inherit;font-size:15px;font-weight:600;color:#fff;
background:var(--accent);border:none;border-radius:8px;cursor:pointer}
button:hover{background:var(--accent-h)}
button:disabled{background:#c4c8d0;cursor:not-allowed}
.chips{margin-bottom:24px}
.chip{display:inline-block;padding:7px 14px;margin:0 6px 6px 0;font-size:13px;
color:var(--text);background:#f0f1f4;border:1px solid var(--border);
border-radius:999px;cursor:pointer;user-select:none}
.chip:hover{background:#e6e8ee;border-color:#cfd3db}
.chip:active{background:var(--accent);color:#fff;border-color:var(--accent)}
.status{padding:10px 14px;margin-bottom:24px;font-size:14px;border-radius:8px;
background:#f0f1f4;color:var(--muted);border:1px solid var(--border)}
.status.ok{background:var(--ok-bg);color:var(--ok);border-color:#bfe6d2}
.status.err{background:var(--err-bg);color:var(--err);border-color:#f2c4be}
.status.busy{background:var(--busy-bg);color:var(--busy);border-color:#ecd9a0}
.outputs{display:flex;gap:24px;justify-content:center;flex-wrap:wrap;margin-bottom:20px}
figure{text-align:center}
figcaption{font-size:13px;color:var(--muted);margin-bottom:8px}
canvas{border:1px solid var(--border);border-radius:8px;background:#fff;
image-rendering:pixelated}
.info{font-size:13px;color:var(--muted);text-align:center}
@media (max-width:560px){
  body{padding:28px 14px}
  .card{padding:20px}
  .row{flex-direction:column}
}
</style>
</head>
<body>

<header>
<h1>TinyGen</h1>
<p class="sub">A tiny neural network on an ESP32-S2 draws 32&times;32 images of clothing, entirely on the chip</p>
</header>

<div class="card">

<span class="label">Describe an item</span>
<div class="row">
<input type="text" id="p" placeholder="e.g. shoe, bag, dress, sandal" autocomplete="off">
<button id="g" onclick="gen()">Generate</button>
</div>

<span class="label">Or pick a class</span>
<div class="chips" id="cb"></div>

<div class="status" id="st">Connecting to device...</div>

<span class="label">Output</span>
<div class="outputs">
<figure>
<figcaption>Grayscale</figcaption>
<canvas id="c1" width="32" height="32" style="width:240px;height:240px"></canvas>
</figure>
<figure>
<figcaption>Thermal</figcaption>
<canvas id="c2" width="32" height="32" style="width:240px;height:240px"></canvas>
</figure>
</div>

<p class="info" id="info">All inference runs on the ESP32-S2. No cloud, no external services.</p>

</div>

<script>
const CLASSES=["T-shirt","Trouser","Pullover","Dress","Coat",
               "Sandal","Shirt","Sneaker","Bag","Ankle boot"];
let ws,ok=false,busy=false,t0=0;

function conn(){
  const p=location.protocol==='https:'?'wss:':'ws:';
  ws=new WebSocket(p+'//'+location.host+'/ws');
  ws.onopen=()=>{ok=true;ss('Connected to device.','ok');setBtn();};
  ws.onclose=()=>{ok=false;ss('Disconnected. Retrying...','err');setBtn();setTimeout(conn,3000);};
  ws.onerror=()=>ss('WebSocket error','err');
  ws.onmessage=e=>msg(JSON.parse(e.data));
}

function msg(d){
  if(d.type==='status'){ ss(d.message,'busy'); }
  else if(d.type==='error'){ ss('Error: '+d.message,'err'); busy=false; setBtn(); }
  else if(d.type==='image'){
    draw(d);
    const secs=((Date.now()-t0)/1000).toFixed(1);
    ss('Generated '+d.class_name+' in '+secs+' seconds','ok');
    document.getElementById('info').textContent=
      'Last generation: '+d.class_name+' (class '+d.class+') in '+secs+'s on the ESP32-S2';
    busy=false; setBtn();
  }
}

function ss(m,c){const e=document.getElementById('st');e.textContent=m;e.className='status '+(c||'');}
function setBtn(){document.getElementById('g').disabled=(!ok||busy);}

function gen(name){
  const v=name||document.getElementById('p').value.trim();
  if(!v||!ok||busy)return;
  busy=true;setBtn();
  t0=Date.now();
  ss('Generating "'+v+'" - this usually takes around 30 seconds','busy');
  ws.send(JSON.stringify({action:'generate',prompt:v}));
}

document.getElementById('p').addEventListener('keypress',e=>{if(e.key==='Enter')gen();});

function draw(d){
  const W=d.width,H=d.height,hx=d.data;
  const cl=new Uint8Array(W*H);
  for(let i=0;i<W*H;i++)cl[i]=parseInt(hx.substr(i*2,2),16);
  let c1=document.getElementById('c1').getContext('2d');
  let id1=c1.createImageData(W,H);
  for(let i=0;i<cl.length;i++){
    const v=Math.round(cl[i]/23*255);
    id1.data[i*4]=v;id1.data[i*4+1]=v;id1.data[i*4+2]=v;id1.data[i*4+3]=255;
  }
  c1.putImageData(id1,0,0);
  const cm=[[0,0,0],[15,0,25],[30,0,50],[50,0,80],[70,0,110],[90,0,130],
  [110,0,150],[130,10,160],[150,30,150],[170,50,130],[185,70,110],[200,90,90],
  [210,110,70],[220,130,50],[230,150,35],[240,170,25],[245,185,20],[250,200,20],
  [252,215,30],[254,225,50],[255,235,80],[255,242,120],[255,248,160],[255,255,200]];
  let c2=document.getElementById('c2').getContext('2d');
  let id2=c2.createImageData(W,H);
  for(let i=0;i<cl.length;i++){
    const c=cm[Math.min(23,cl[i])];
    id2.data[i*4]=c[0];id2.data[i*4+1]=c[1];id2.data[i*4+2]=c[2];id2.data[i*4+3]=255;
  }
  c2.putImageData(id2,0,0);
}

window.onload=()=>{
  const wrap=document.getElementById('cb');
  CLASSES.forEach(n=>{
    const b=document.createElement('span');
    b.className='chip';b.textContent=n;
    b.onclick=()=>{document.getElementById('p').value=n;gen(n);};
    wrap.appendChild(b);
  });
  conn();
  setBtn();
};
</script>
</body>
</html>
)rawliteral";
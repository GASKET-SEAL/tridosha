'use strict';
const $=id=>document.getElementById(id), names=['Vata','Pitta','Kapha'], colors=['#1f9e8f','#9bc23c','#1b5e3a'], labelColors=['#16776c','#6d8a1f','#1b5e3a'];
let result=null, currentConfig=null;
const val=id=>Number($(id).value), fmt=(n,d=2)=>Number(n).toFixed(d);
function setStatus(text,kind=''){const s=$('status');s.textContent=text;s.className=kind;}

function config(){const p=[0,1,2].map(i=>val('p'+i)),total=p.reduce((a,b)=>a+b,0);const target=Math.max(0,['vata','pitta','kapha'].indexOf($('scenario').value));let initial=[-.3,-.3,-.3];initial[target]=.85;initial[(target+2)%3]=-.55;if($('scenario').value==='equilibrium')initial=[0,0,0];return {baseline:p.map(x=>x/total),initial,days:val('days'),gain:val('gain'),amplitude:val('amplitude'),start:val('start'),duration:val('duration'),target,decay:[0,1,2].map(i=>val('r'+i)),coupling:val('coupling')};}
function path(points){return points.map((p,i)=>(i?'L':'M')+p.map(x=>fmt(x,2)).join(',')).join(' ');}
function svgLine(points,color,width=2,extra=''){return `<path d="${path(points)}" stroke="${color}" stroke-width="${width}" fill="none" stroke-linecap="round" stroke-linejoin="round" ${extra}/>`;}

function recoveryChart(){
  const c=result.controlled,n=result.natural,w=760,h=260,left=40,right=15,top=20,bottom=30,max=Math.max(...c.distance,...n.distance)*1.12||1;
  const X=t=>left+t/currentConfig.days*(w-left-right),Y=d=>h-bottom-d/max*(h-top-bottom);
  let s=`<rect x="${X(currentConfig.start)}" y="${top}" width="${X(currentConfig.start+currentConfig.duration)-X(currentConfig.start)}" height="${h-top-bottom}" fill="#cde8d6" opacity=".7" rx="3"/>`;
  for(let i=0;i<=4;i++){const d=max*i/4,y=Y(d);s+=`<line x1="${left}" y1="${y}" x2="${w-right}" y2="${y}" stroke="#e3f1e8"/><text x="${left-10}" y="${y+4}" text-anchor="end">${fmt(d,1)}</text>`;}
  for(let i=0;i<=7;i++){const t=currentConfig.days*i/7;s+=`<text x="${X(t)}" y="${h-8}" text-anchor="middle">${fmt(t,0)}</text>`;}
  s+=svgLine(n.time.map((t,i)=>[X(t),Y(n.distance[i])]),'#3fa876',2,'stroke-dasharray="6 5"');
  s+=svgLine(c.time.map((t,i)=>[X(t),Y(c.distance[i])]),'#1b6b4a',3);
  s+=`<line x1="${left}" y1="${Y(.05)}" x2="${w-right}" y2="${Y(.05)}" stroke="#9fd4b4" stroke-dasharray="2 4"/>`;
  $('recovery-chart').innerHTML=s;
}

function triangle(index){
  const c=result.controlled,xy=v=>[45+270*(v[1]+v[2]/2),225-200*v[2]],grid='#d6e8dc';
  let s='<path d="M45,225 L315,225 L180,25 Z" fill="#f3faf5" stroke="#9fd4b4" stroke-width="1.5"/>';
  for(let j=1;j<5;j++){const k=j/5;s+=svgLine([xy([1-k,0,k]),xy([0,1-k,k])],grid,.9);s+=svgLine([xy([k,1-k,0]),xy([k,0,1-k])],grid,.9);s+=svgLine([xy([1-k,k,0]),xy([0,k,1-k])],grid,.9);}
  s+=svgLine(c.shares.map(xy),'#9fd4b4',1.8);
  s+=svgLine(c.shares.slice(0,index+1).map(xy),'#1b6b4a',2.6);
  const b=xy(c.baseline),a=xy(c.shares[0]),v=xy(c.shares[index]);
  s+=`<path d="M${b[0]},${b[1]-7} l7,7 l-7,7 l-7,-7 Z" fill="white" stroke="#14513b" stroke-width="1.8"/><circle cx="${a[0]}" cy="${a[1]}" r="5" fill="white" stroke="#3fa876" stroke-width="2"/><circle cx="${v[0]}" cy="${v[1]}" r="6" fill="#1b6b4a" stroke="white" stroke-width="2"/>`;
  s+=`<text x="35" y="248" text-anchor="middle" style="fill:${labelColors[0]};font-size:13px;font-weight:600">Vata</text><text x="325" y="248" text-anchor="middle" style="fill:${labelColors[1]};font-size:13px;font-weight:600">Pitta</text><text x="180" y="14" text-anchor="middle" style="fill:${labelColors[2]};font-size:13px;font-weight:600">Kapha</text>`;
  $('triangle').innerHTML=s;
}

function snapshot(){
  if(!result)return;
  const i=val('scrubber'),c=result.controlled;
  $('day-label').textContent='Day '+fmt(c.time[i],1);
  $('selected-distance').textContent=fmt(c.distance[i],3);
  $('shares').innerHTML=c.shares[i].map((v,j)=>`<div class="share"><div class="share-head"><span style="color:${labelColors[j]}">● ${names[j]}</span><strong>${fmt(v*100,1)} <span class="pct">%</span></strong></div><div class="bar"><span style="width:${v*100}%;background:${colors[j]}"></span></div></div>`).join('');
  triangle(i);
}

function render(){
  const c=result.controlled,n=result.natural;
  $('stable').textContent=c.stability.stable?'Stable':'Unstable';
  $('recovery').textContent=c.metrics.settling_days===null?'Not reached':fmt(c.metrics.settling_days,1)+' days';
  $('recovery-note').textContent=n.metrics.settling_days===null?'Natural: not reached':'Natural: '+fmt(n.metrics.settling_days,1)+' days';
  $('reduction').textContent=n.metrics.integrated_error>1e-10?fmt(100*(1-c.metrics.integrated_error/n.metrics.integrated_error),1)+'%':'—';
  $('scrubber').max=c.time.length-1;$('scrubber').value=0;
  recoveryChart();snapshot();
}

async function run(event){
  event?.preventDefault();
  if(!$('controls').reportValidity())return;
  const q=config();
  if(q.start+q.duration>q.days){setStatus('The disturbance must end within the simulation length.','err');return;}
  $('run').disabled=true;$('export').disabled=true;setStatus('Running…');
  try{
    const response=await fetch('/api/simulate',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(q)});
    if(!response.ok){const error=await response.json();throw new Error(typeof error.detail==='string'?error.detail:'Please check the settings.');}
    result=await response.json();currentConfig=q;render();
    $('export').disabled=false;setStatus('Done ✓','ok');
  }catch(e){setStatus('Could not run the simulation. '+e.message+' Make sure the local server is running.','err');}
  finally{$('run').disabled=false;}
}

$('controls').addEventListener('submit',run);
$('controls').addEventListener('input',()=>{$('amplitude-out').textContent=fmt(val('amplitude'));$('gain-out').textContent=fmt(val('gain'));setStatus('Settings changed — run to update.');});
$('scenario').addEventListener('change',()=>{$('amplitude').value=$('scenario').value==='equilibrium'?0:.6;$('amplitude-out').textContent=fmt(val('amplitude'));});
$('reset-experiment').addEventListener('click',()=>{$('controls').reset();$('amplitude-out').textContent='0.60';$('gain-out').textContent='0.45';run();});
$('scrubber').addEventListener('input',snapshot);
$('export').addEventListener('click',()=>{if(!result)return;const payload={schema_version:1,description:'Illustrative ODE experiment, not clinical prediction',configuration:currentConfig,results:result};const url=URL.createObjectURL(new Blob([JSON.stringify(payload,null,2)],{type:'application/json'}));const a=document.createElement('a');a.href=url;a.download='tridosha-experiment.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);});

const navLinks=[...document.querySelectorAll('.topbar nav a')];
navLinks.forEach(a=>a.addEventListener('click',()=>{navLinks.forEach(n=>n.classList.remove('active'));a.classList.add('active');}));
if('IntersectionObserver' in window){
  const io=new IntersectionObserver(es=>{es.forEach(e=>{if(e.isIntersecting){navLinks.forEach(n=>n.classList.toggle('active',n.getAttribute('href')==='#'+e.target.id));}});},{rootMargin:'-40% 0px -55% 0px'});
  document.querySelectorAll('section[id]').forEach(s=>io.observe(s));
}

async function evidence(){
  try{
    const response=await fetch('/api/evidence');
    if(!response.ok)throw new Error('Validation results are not available yet.');
    const e=await response.json(),b=e.cohort_benchmark;
    $('evidence-content').innerHTML=`<article class="panel"><p class="card-title">Solver accuracy</p><strong class="evidence-value">${e.analytic_max_error.toExponential(2)}</strong><p class="muted">Max error vs analytic solution</p></article><article class="panel"><p class="card-title">Calibration error</p><strong class="evidence-value">${fmt(e.calibration.heldout_rmse_after,4)}</strong><p class="muted">Before fitting: ${fmt(e.calibration.heldout_rmse_before,4)}</p></article><article class="panel"><p class="card-title">Prakriti benchmark</p><table class="evidence-table"><tbody>${Object.entries(b).map(([key,v])=>`<tr><td>${key}</td><td>${fmt(100*v.accuracy,1)}%</td></tr>`).join('')}</tbody></table></article>`;
  }catch(e){$('evidence-content').innerHTML=`<article class="panel"><p class="muted">${e.message}</p></article>`;}
}
run();evidence();

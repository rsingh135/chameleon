import addOnUISdk from 'https://new.express.adobe.com/static/add-on-sdk/sdk.js';
import {ViewerSession,canvas,engineOrigin,pngBytes,readJson,resultBundle,task,type Result} from '../shared/viewer.js';

const $=<T extends HTMLElement>(id:string)=>document.getElementById(id) as T;
const session=new ViewerSession();let busy=false,hostReady=false,controller:AbortController|undefined,loadGeneration=0,operation=0;
const buttons=['download-json','download-png','place'];
function message(text:string):void {$('status').textContent=text;$('error').hidden=true;}
function failure(error:unknown):void {$('error').textContent=error instanceof Error?error.message:String(error);$('error').hidden=false;$('status').textContent='Operation stopped. Check the message and retry.';}
function controls():void {
  $<HTMLButtonElement>('resize').disabled=busy||!session.current;
  $('cancel').hidden=!busy;
  for(const id of buttons)$<HTMLButtonElement>(id).disabled=busy||!session.result||(id==='place'&&!hostReady);
}
function invalidate():void {
  loadGeneration++;operation++;busy=false;session.invalidate();controller?.abort();controller=undefined;
  if(!$('comparison').hidden)$('preview-state').textContent='Previous result';
  controls();
}
function setTask(source:ReturnType<typeof task>,label:string):void {
  invalidate();session.load(source);
  $<HTMLInputElement>('width').value=String(source.target_canvas.width);$<HTMLInputElement>('height').value=String(source.target_canvas.height);
  $<HTMLSelectElement>('preset').value='custom';$('summary').textContent=`${label} · ${source.source.elements.length} elements · ${source.source.canvas.width} × ${source.source.canvas.height}`;
  message('Source loaded. Choose a target and generate, or open an engine result.');controls();
}
async function decode(preview:string,width:number,height:number):Promise<HTMLImageElement> {
  const image=new Image();image.src=preview;await image.decode();
  if(image.naturalWidth!==width||image.naturalHeight!==height)throw new Error('The decoded PNG has unexpected dimensions.');return image;
}
async function show(result:Result,generation:number):Promise<boolean> {
  const [original,resized]=await Promise.all([decode(result.source_png,result.source.canvas.width,result.source.canvas.height),decode(result.output_png,result.target_canvas.width,result.target_canvas.height)]);
  if(generation!==session.generation)return false;
  $<HTMLImageElement>('source-image').src=original.src;$<HTMLImageElement>('result-image').src=resized.src;
  $('source-size').textContent=`${result.source.canvas.width} × ${result.source.canvas.height}`;
  $('result-size').textContent=`${result.target_canvas.width} × ${result.target_canvas.height}`;$('preview-state').textContent='Current result';
  const warnings=result.warnings??[];$('warnings').hidden=!warnings.length;
  $('warnings').textContent=warnings.map(w=>typeof w==='string'?w:JSON.stringify(w)).join(' · ');
  $('elements').replaceChildren(...result.design.elements.map(element=>{const li=document.createElement('li');li.textContent=`${element.id} · ${element.type} · ${Math.round(Number(element.x))}, ${Math.round(Number(element.y))} · ${Math.round(Number(element.width))} × ${Math.round(Number(element.height))}`;return li;}));
  $('comparison').hidden=false;return true;
}
async function loadFile(input:HTMLInputElement,kind:'task'|'result'):Promise<void> {
  const file=input.files?.[0];if(!file)return;
  invalidate();const ticket=loadGeneration,owner=++operation;busy=true;controls();message('Reading JSON file…');
  try {
    if(file.size>64*1024*1024)throw new Error('Choose a JSON file smaller than 64 MiB.');
    const value=readJson(await file.text());if(ticket!==loadGeneration)return;
    if(kind==='task')setTask(task(value),file.name);
    else {
      const result=resultBundle(value),generation=session.generation;
      if(!await show(result,generation))return;
      session.load({source:result.source,target_canvas:result.target_canvas});session.result=result;$<HTMLInputElement>('width').value=String(result.target_canvas.width);$<HTMLInputElement>('height').value=String(result.target_canvas.height);$<HTMLSelectElement>('preset').value='custom';$<HTMLSelectElement>('sample').value='';
      $('summary').textContent=`${file.name} · ${result.source.elements.length} elements`;
      message('Engine result opened. JSON and full-resolution PNG are ready to export.');
    }
  } catch(error){if(ticket===loadGeneration){session.result=undefined;failure(error);}}
  finally{if(owner===operation)busy=false;input.value='';controls();}
}
for(const [id,kind] of [['task-file','task'],['result-file','result']] as const)$<HTMLInputElement>(id).addEventListener('change',()=>void loadFile($<HTMLInputElement>(id),kind));
$<HTMLSelectElement>('sample').addEventListener('change',async()=>{
  const name=$<HTMLSelectElement>('sample').value;if(!name)return;
  invalidate();const ticket=loadGeneration,owner=++operation;busy=true;controls();message('Loading bundled sample…');
  try{const response=await fetch(new URL(`../samples/${encodeURIComponent(name)}.json`,import.meta.url));if(!response.ok)throw new Error('The bundled sample could not be loaded. Open its task JSON instead.');const value:unknown=await response.json();if(ticket===loadGeneration)setTask(task(value),name.replace('chameleon_sample_',''));}
  catch(error){if(ticket===loadGeneration)failure(error);}finally{if(owner===operation)busy=false;controls();}
});
for(const id of ['width','height','engine'])$(id).addEventListener('input',()=>{invalidate();if(id!=='engine')$<HTMLSelectElement>('preset').value='custom';message('Inputs changed. Generate a new result to enable exports.');});
$('preset').addEventListener('change',()=>{const value=$<HTMLSelectElement>('preset').value;if(value==='custom')return;const [width,height]=value.split(',');$<HTMLInputElement>('width').value=width!;$<HTMLInputElement>('height').value=height!;invalidate();message('Target changed. Generate a new result to enable exports.');});
$('cancel').addEventListener('click',()=>{invalidate();busy=false;message('Generation cancelled. You can retry.');controls();});
$('reflow').addEventListener('submit',async event=>{
  event.preventDefault();if(busy)return;
  invalidate();let pending:ReturnType<ViewerSession['begin']>;
  try{pending=session.begin(canvas({width:Number($<HTMLInputElement>('width').value),height:Number($<HTMLInputElement>('height').value)}),crypto.randomUUID());}catch(error){failure(error);controls();return;}
  const {request,generation}=pending,owner=++operation;
  busy=true;const active=new AbortController();controller=active;controls();message('Generating with the local engine. Complex designs can take several minutes…');
  const timeout=setTimeout(()=>active.abort(),600000);
  try{
    const base=engineOrigin($<HTMLInputElement>('engine').value);
    const response=await fetch(new URL('/api/addon/resize',base),{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(request),signal:active.signal});
    const payload:unknown=await response.json();if(!response.ok)throw new Error((payload as {error?:string}).error??`The local engine returned ${response.status}.`);
    if(!session.accept(payload,request,generation))return;
    const result=session.result!;session.result=undefined;
    if(!await show(result,generation))return;session.result=result;message('Resized design ready. Compare the render before exporting.');
  }catch(error){if(generation===session.generation){session.result=undefined;failure(error instanceof TypeError?new Error('The local engine could not be reached. Check it is running, allow local-network access if offered, or open an engine-result bundle.'):active.signal.aborted?new Error('Generation timed out. Retry, or generate an engine-result bundle from the offline CLI.'):error);}}
  finally{clearTimeout(timeout);if(controller===active)controller=undefined;if(owner===operation)busy=false;controls();}
});
function download(blob:Blob,name:string):void {const url=URL.createObjectURL(blob),link=document.createElement('a');link.href=url;link.download=name;document.body.append(link);link.click();link.remove();setTimeout(()=>URL.revokeObjectURL(url),1000);}
$('download-json').addEventListener('click',()=>{const result=session.result;if(result)download(new Blob([JSON.stringify(result.design,null,2)+'\n'],{type:'application/json'}),`chameleon-${result.target_canvas.width}x${result.target_canvas.height}.json`);});
$('download-png').addEventListener('click',()=>{const result=session.result;if(result)download(new Blob([pngBytes(result.output_png,result.target_canvas) as Uint8Array<ArrayBuffer>],{type:'image/png'}),`chameleon-${result.target_canvas.width}x${result.target_canvas.height}.png`);});
$('place').addEventListener('click',async()=>{
  const result=session.result;if(busy||!hostReady||!result)return;busy=true;controls();message('Adding the rendered PNG to Adobe Express…');
  try{await addOnUISdk.app.document.addImage(new Blob([pngBytes(result.output_png,result.target_canvas) as Uint8Array<ArrayBuffer>],{type:'image/png'}),{title:`Chameleon ${result.target_canvas.width} × ${result.target_canvas.height}`});message('PNG added to the current Adobe Express page. Your JSON download retains the design elements.');}
  catch(error){failure(error);}finally{busy=false;controls();}
});
void addOnUISdk.ready.then(()=>{hostReady=true;$('host-status').textContent='Connected to Adobe Express. Add PNG places a rendered image on the current page.';controls();}).catch(()=>{$('host-status').textContent='Open this add-on inside Adobe Express to add a PNG. File viewing and downloads remain available.';});
controls();

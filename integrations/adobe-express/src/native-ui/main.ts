import addOnUISdk from 'https://new.express.adobe.com/static/add-on-sdk/sdk.js';
import { resizeRequest, planWriteback, previewSource, type Snapshot, type Design, type AssetResource, type FontResource } from '../shared/adapter.js';

interface SandboxApi {
  capture():Promise<{snapshot:Snapshot;assets:AssetResource[]}>;
  createEditableCopy(snapshotId:string,result:Design):Promise<{page_id:string}>;
}
const $=<T extends HTMLElement>(id:string)=>document.getElementById(id) as T;
const captureButton=$<HTMLButtonElement>('capture'),resizeButton=$<HTMLButtonElement>('resize'),applyButton=$<HTMLButtonElement>('apply');
let current:{snapshot:Snapshot;assets:AssetResource[]}|undefined;
let result:Design|undefined, generation=0, busy=false;
const files=new Map<string,HTMLInputElement>();
function invalidate():void { generation++; result=undefined;applyButton.disabled=true;$('preview').hidden=true;$<HTMLImageElement>('preview-image').removeAttribute('src'); }
function message(text:string):void { $('status').textContent=text;$('error').hidden=true; }
function failure(error:unknown):void { $('error').textContent=error instanceof Error?error.message:String(error);$('error').hidden=false;messageWithoutClearing('Operation stopped.'); }
function messageWithoutClearing(text:string):void { $('status').textContent=text; }
function lock(value:boolean):void {busy=value;captureButton.disabled=value;resizeButton.disabled=value||!current;applyButton.disabled=value||!result;}
async function bytes(file:File):Promise<string> {
  const data=new Uint8Array(await file.arrayBuffer());let binary='';
  for(let i=0;i<data.length;i+=8192) binary+=String.fromCharCode(...data.subarray(i,i+8192));
  return btoa(binary);
}
await addOnUISdk.ready;
const runtime=addOnUISdk.instance.runtime;
if(!runtime.apiProxy)throw new Error('The Adobe document sandbox is not available. Open this add-on inside Adobe Express.');
const sandbox=await runtime.apiProxy<SandboxApi>(addOnUISdk.constants.RuntimeType.documentSandbox);
captureButton.disabled=false;message('Ready. Read the current Adobe Express page.');
captureButton.addEventListener('click',async()=>{
  if(busy)return;invalidate();current=undefined;lock(true);message('Reading native elements and permitted image resources…');
  try {
    current=await sandbox.capture();
    $('summary').textContent=`${current.snapshot.source.elements.length-1} editable elements · ${current.snapshot.page.width} × ${current.snapshot.page.height}`;
    const fontContainer=$('fonts');fontContainer.replaceChildren();files.clear();
    const legend=document.createElement('legend');legend.textContent='Exact document fonts';fontContainer.append(legend);
    if(!current.snapshot.required_fonts.length){const p=document.createElement('p');p.textContent='No text fonts required.';fontContainer.append(p);}
    for(const font of current.snapshot.required_fonts) {
      const label=document.createElement('label'),input=document.createElement('input');
      label.textContent=font;input.type='file';input.accept='.ttf,.otf';input.required=true;
      input.addEventListener('change',invalidate);label.append(input);fontContainer.append(label);files.set(font,input);
    }
    message('Page read. Supply each exact font file, then generate a layout.');
  } catch(error){failure(error);} finally{lock(false);}
});
for(const id of ['width','height','engine']) $(id).addEventListener('input',invalidate);
$('reflow').addEventListener('submit',async event=>{
  event.preventDefault();if(busy||!current)return;invalidate();const token=generation,captured=current;
  lock(true);message('Generating layout with the local engine…');
  try {
    const fonts:FontResource[]=[];
    for(const [name,input] of files){const file=input.files?.[0];if(!file)throw new Error(`Choose the exact font file for ${name}.`);fonts.push({font_family:name,postscript_name:name,weight:'regular',style:'normal',data_base64:await bytes(file)});}
    const target={width:Number($<HTMLInputElement>('width').value),height:Number($<HTMLInputElement>('height').value)};
    const request=resizeRequest(captured.snapshot,target,captured.assets,fonts);
    const base=new URL($<HTMLInputElement>('engine').value);
    if(!['127.0.0.1','localhost','[::1]'].includes(base.hostname))throw new Error('Use the local engine on localhost or 127.0.0.1.');
    const response=await fetch(new URL('/api/document/resize',base),{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(request),signal:AbortSignal.timeout(120000)});
    const payload=await response.json() as {design?:Design;error?:string;render_url?:string;preview_url?:string;output_png?:string;document?:{snapshot_id?:string}};
    if(!response.ok)throw new Error(payload.error??`Local engine returned ${response.status}.`);
    if(token!==generation)return;
    if(!payload.design)throw new Error('The engine response did not contain a design.');
    if(payload.document?.snapshot_id!==captured.snapshot.snapshot_id)throw new Error('The engine response belongs to a different document capture. Read the page again.');
    if(payload.design.canvas.width!==target.width||payload.design.canvas.height!==target.height)throw new Error('The engine returned a different target canvas.');
    planWriteback(captured.snapshot,payload.design);result=payload.design;
    const preview=payload.render_url??payload.preview_url??payload.output_png;
    if(preview){
      const image=$<HTMLImageElement>('preview-image');image.src=previewSource(preview,base.href);await image.decode();
      if(token!==generation)return;$('preview').hidden=false;}
    message('Layout ready. Create a separate editable Adobe page, then inspect it.');
  } catch(error){if(token===generation){result=undefined;failure(error);}} finally{lock(false);}
});
applyButton.addEventListener('click',async()=>{
  if(busy||!current||!result)return;lock(true);message('Creating an editable copy…');
  try {await sandbox.createEditableCopy(current.snapshot.snapshot_id,result);invalidate();current=undefined;message('Editable page created. Review it in Adobe Express before exporting.');}
  catch(error){failure(error);}finally{lock(false);}
});

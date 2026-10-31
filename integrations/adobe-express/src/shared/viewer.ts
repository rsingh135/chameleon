/** Contract and state for the distributable stage-one viewer. No native sandbox APIs. */
export interface Canvas { width:number; height:number }
export interface Element { id:string; type:'text'|'image'|'shape'; [key:string]:unknown }
export interface Design { canvas:Canvas; elements:Element[]; [key:string]:unknown }
export interface Task { source:Design; target_canvas:Canvas }
export interface Request extends Task { request_id:string }
export interface Result extends Request {
  schema_version?:number;
  design:Design;
  source_png:string;
  output_png:string;
  report?:{violations?:unknown[];[key:string]:unknown};
  warnings?:unknown[];
  [key:string]:unknown;
}
const MAX_JSON_BYTES=64*1024*1024;
const FROZEN=['type','content','font_family','font_weight','font_style','color','text_align',
  'line_height','letter_spacing','style_runs','asset_id','shape_type','fill_color','stroke_color',
  'group_id','z_index','opacity','rotation','native_text_effects','source_art'];
function record(value:unknown):value is Record<string,unknown> { return !!value&&typeof value==='object'&&!Array.isArray(value); }
function finiteTree(value:unknown,depth=0):void {
  if(depth>40)throw new Error('The JSON is nested too deeply.');
  if(typeof value==='number'&&!Number.isFinite(value))throw new Error('JSON numbers must be finite.');
  if(value&&typeof value==='object')for(const child of Object.values(value))finiteTree(child,depth+1);
}
export function canvas(value:unknown):Canvas {
  if(!record(value))throw new Error('A canvas needs width and height.');
  const {width,height}=value;
  if(typeof width!=='number'||typeof height!=='number'||!Number.isInteger(width)||!Number.isInteger(height)
    ||Math.min(width,height)<1||Math.max(width,height)>8192||width*height>20_000_000)
    throw new Error('Use whole canvas dimensions from 1 to 8192, with at most 20 million pixels.');
  return {width,height};
}
export function design(value:unknown):Design {
  if(!record(value)||!Array.isArray(value.elements)||value.elements.length<1||value.elements.length>200)
    throw new Error('A design needs between 1 and 200 elements.');
  canvas(value.canvas);const ids=new Set<string>();
  for(const element of value.elements) {
    if(!record(element)||typeof element.id!=='string'||!element.id||ids.has(element.id)
      ||!['text','image','shape'].includes(String(element.type)))throw new Error('Element IDs must be unique and types must be text, image, or shape.');
    ids.add(element.id);
    for(const key of ['x','y','width','height'])if(typeof element[key]!=='number'||!Number.isFinite(element[key]))throw new Error(`Element ${element.id} has invalid geometry.`);
    for(const key of ['rotation','opacity','corner_radius','stroke_width','line_height','letter_spacing'])if(key in element&&(typeof element[key]!=='number'||!Number.isFinite(element[key])))throw new Error(`Element ${element.id} has invalid ${key}.`);
    if((element.width as number)<=0||(element.height as number)<=0)throw new Error(`Element ${element.id} has an empty size.`);
    if(element.type==='text'&&(typeof element.content!=='string'||typeof element.font_family!=='string'
      ||typeof element.font_size!=='number'||element.font_size<1||typeof element.color!=='string'))throw new Error(`Text ${element.id} is missing its copy, font, size, or color.`);
    if(element.type==='image'&&typeof element.asset_id!=='string')throw new Error(`Image ${element.id} is missing its asset reference.`);
    if(element.type==='shape'&&(typeof element.shape_type!=='string'||typeof element.fill_color!=='string'))throw new Error(`Shape ${element.id} is missing its shape or fill.`);
  }
  finiteTree(value);return structuredClone(value) as unknown as Design;
}
export function readJson(text:string):unknown {
  if(new TextEncoder().encode(text).length>MAX_JSON_BYTES)throw new Error('Choose a JSON file smaller than 64 MiB.');
  const result:unknown=JSON.parse(text);finiteTree(result);return result;
}
export function task(value:unknown):Task {
  if(!record(value))throw new Error('Choose a task JSON containing source and target_canvas.');
  return {source:design(value.source),target_canvas:canvas(value.target_canvas)};
}
export function engineOrigin(value:string):string {
  const url=new URL(value);
  if(!['http:','https:'].includes(url.protocol)||!['localhost','127.0.0.1','[::1]'].includes(url.hostname)
    ||url.username||url.password||url.search||url.hash||url.pathname!=='/')throw new Error('Use a local engine origin such as http://127.0.0.1:8765.');
  return url.origin;
}
export function pngBytes(value:unknown,expected:Canvas):Uint8Array {
  if(typeof value!=='string'||value.length>90_000_000||!/^data:image\/png;base64,[A-Za-z0-9+/]+={0,2}$/.test(value))throw new Error('The engine must provide a PNG data URL.');
  let binary:string;try{binary=atob(value.slice(value.indexOf(',')+1));}catch{throw new Error('The PNG data is not valid base64.');}
  const bytes=Uint8Array.from(binary,c=>c.charCodeAt(0));
  if(bytes.length<33||[137,80,78,71,13,10,26,10].some((n,i)=>bytes[i]!==n)
    ||String.fromCharCode(...bytes.slice(12,16))!=='IHDR')throw new Error('The preview is not a PNG image.');
  const view=new DataView(bytes.buffer);
  if(view.getUint32(16)!==expected.width||view.getUint32(20)!==expected.height)throw new Error('The PNG dimensions do not match the design canvas.');
  return bytes;
}
function canonical(value:unknown):string|undefined {
  if(Array.isArray(value))return '['+value.map(canonical).join(',')+']';
  if(record(value))return '{'+Object.keys(value).sort().map(key=>JSON.stringify(key)+':'+canonical(value[key])).join(',')+'}';
  return JSON.stringify(value);
}
function same(a:unknown,b:unknown):boolean { return canonical(a)===canonical(b); }
export function validateResult(value:unknown,expected:Request):Result {
  if(!record(value)||value.request_id!==expected.request_id)throw new Error('This result belongs to a different request. Generate again.');
  if(!same(canvas(value.target_canvas),expected.target_canvas))throw new Error('This result uses a different target canvas.');
  const output=design(value.design);
  if(!same(output.canvas,expected.target_canvas))throw new Error('The output canvas does not match the selected target.');
  const byId=new Map(output.elements.map(e=>[e.id,e]));
  if(output.elements.length!==expected.source.elements.length)throw new Error('The engine changed the number of elements.');
  for(const source of expected.source.elements) {
    const candidate=byId.get(source.id);if(!candidate)throw new Error(`The engine dropped element ${source.id}.`);
    for(const key of FROZEN)if(key in source&&!same(source[key],candidate[key]))throw new Error(`The engine changed ${source.id}.${key}.`);
    if(source.type==='image') {
      // Only the Python contract can approve a source fill-image as a cover
      // background. A foreground contain-image never gains that exception.
      const backgroundCover=source.fit==='fill'&&candidate.fit==='cover';
      if(!same(source.fit??'contain',candidate.fit??'contain')&&!backgroundCover)throw new Error(`The engine changed ${source.id}.fit.`);
      if(!same(source.image_position,candidate.image_position)&&!backgroundCover)throw new Error(`The engine changed ${source.id}.image_position.`);
      if(source.fit!=='cover'&&!backgroundCover) {
        const sourceRatio=Number(source.width)/Number(source.height),outputRatio=Number(candidate.width)/Number(candidate.height);
        if(Math.abs(outputRatio-sourceRatio)/sourceRatio>.05)throw new Error(`The engine distorted image ${source.id}.`);
      }
    }
  }
  for(const element of output.elements) {
    const x=element.x as number,y=element.y as number,w=element.width as number,h=element.height as number;
    const angle=Number(element.rotation??0)*Math.PI/180;
    const rotatedW=Math.abs(w*Math.cos(angle))+Math.abs(h*Math.sin(angle));
    const rotatedH=Math.abs(w*Math.sin(angle))+Math.abs(h*Math.cos(angle));
    const cx=x+w/2,cy=y+h/2;
    if(cx-rotatedW/2<-.000001||cy-rotatedH/2<-.000001||cx+rotatedW/2>output.canvas.width+.000001
      ||cy+rotatedH/2>output.canvas.height+.000001)throw new Error(`Element ${element.id} is outside the target canvas.`);
  }
  if(value.report!==undefined&&(!record(value.report)||value.report.violations!==undefined
    &&(!Array.isArray(value.report.violations)||value.report.violations.length)))throw new Error('The engine reported a contract violation.');
  if(value.warnings!==undefined&&!Array.isArray(value.warnings))throw new Error('The engine returned an invalid warning list.');
  pngBytes(value.source_png,expected.source.canvas);pngBytes(value.output_png,expected.target_canvas);
  return structuredClone({...value,source:expected.source,target_canvas:expected.target_canvas,design:output}) as unknown as Result;
}
export function resultBundle(value:unknown):Result {
  if(!record(value)||value.schema_version!==1||typeof value.request_id!=='string'||!value.request_id||value.request_id.length>200)
    throw new Error('Choose a Chameleon engine-result bundle (schema_version 1).');
  return validateResult(value,{...task(value),request_id:value.request_id});
}
export class ViewerSession {
  current?:Task;
  result?:Result;
  generation=0;
  invalidate():void { this.generation++;this.result=undefined; }
  load(value:Task):void { this.invalidate();this.current=structuredClone(value); }
  begin(target:Canvas,id:string):{request:Request;generation:number} {
    if(!this.current)throw new Error('Load a source task first.');
    this.invalidate();this.current.target_canvas=canvas(target);
    return {request:{...structuredClone(this.current),request_id:id},generation:this.generation};
  }
  accept(value:unknown,request:Request,generation:number):boolean {
    if(generation!==this.generation)return false;
    this.result=validateResult(value,request);return true;
  }
  openBundle(value:unknown):void {
    const result=resultBundle(value);
    this.load({source:result.source,target_canvas:result.target_canvas});this.result=result;
  }
}

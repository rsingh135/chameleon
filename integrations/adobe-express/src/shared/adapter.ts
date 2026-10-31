/** Pure, fail-closed Adobe Express snapshot ↔ Chameleon boundary.
 * No Adobe classes cross the iframe/sandbox boundary. Native objects are never
 * retained between requests. Unsupported features are errors, not rasterization.
 */
export type Matrix = readonly [number, number, number, number, number, number];
export type Rect = { x: number; y: number; width: number; height: number };
export type RGBA = { red: number; green: number; blue: number; alpha: number };
export interface NativeRun {
  length: number; fontSize: number; font: { postscriptName: string; family: string; style: string; availableForEditing: boolean };
  color: RGBA; letterSpacing: number; underline: boolean; baselineShift: number; link?: string;
}
export interface NativeNode {
  id: string; kind: 'group' | 'text' | 'rectangle' | 'ellipse' | 'line' | 'image';
  transform: Matrix; bounds: Rect; opacity: number; children?: NativeNode[];
  text?: string; runs?: NativeRun[]; alignment?: 'left' | 'center' | 'right';
  lineSpacing?: number; layout?: 'autoWidth' | 'autoHeight';
  fill?: string; stroke?: { color: string; width: number }; radius?: number;
  assetId?: string;
}
export interface NativePage {
  pageId: string; artboardId: string; width: number; height: number;
  background: string; nodes: NativeNode[];
}
export interface Element extends Rect {
  id: string; type: 'text' | 'image' | 'shape'; rotation: number; opacity: number;
  z_index: number; group_id?: string; [key: string]: unknown;
}
export interface Design { canvas: { width: number; height: number }; elements: Element[] }
export interface AssetResource { asset_id: string; mime_type: string; data_base64: string }
export interface FontResource { font_family: string; postscript_name: string; weight: 'regular'; style: 'normal'; data_base64: string }
export interface Binding { id: string; root_id: string; path: string[] }
export interface Snapshot {
  snapshot_id: string; page: NativePage; source: Design;
  node_bindings: Binding[]; group_tree: NativeNode[]; required_fonts: string[];
  font_pixels_per_point: number;
}
export class UnsupportedDesignError extends Error {
  constructor(public code: string, message: string, public nodeId?: string) {
    super(`${nodeId ? `${nodeId}: ` : ''}${message}`); this.name = 'UnsupportedDesignError';
  }
}
// The document sandbox does not promise the browser structuredClone API.
export function copy<T>(value:T):T { return JSON.parse(JSON.stringify(value)) as T; }
/** Accept the local engine's PNG data URL or a URL on that engine's origin. */
export function previewSource(value:string,engineUrl:string):string {
  requireDesign(typeof value==='string'&&value.length<=44*1024*1024,'invalid_preview','The engine preview is missing or too large.');
  if(value.startsWith('data:')) {
    requireDesign(/^data:image\/png;base64,iVBORw0KGgo[A-Za-z0-9+/]*={0,2}$/.test(value)&&(value.length-'data:image/png;base64,'.length)%4===0,
      'invalid_preview','Only a bounded PNG base64 preview is accepted.');
    return value;
  }
  const base=new URL(engineUrl),url=new URL(value,base);
  requireDesign(['http:','https:'].includes(url.protocol)&&url.origin===base.origin,'invalid_preview','The preview URL must belong to the configured local engine.');
  return url.href;
}
export function requireDesign(condition: unknown, code: string, message: string, id?: string): asserts condition {
  if (!condition) throw new UnsupportedDesignError(code, message, id);
}
export const identity: Matrix = [1, 0, 0, 1, 0, 0];
export function multiply(a: Matrix, b: Matrix): Matrix {
  return [a[0]*b[0]+a[2]*b[1], a[1]*b[0]+a[3]*b[1], a[0]*b[2]+a[2]*b[3], a[1]*b[2]+a[3]*b[3], a[0]*b[4]+a[2]*b[5]+a[4], a[1]*b[4]+a[3]*b[5]+a[5]];
}
export function point(m: Matrix, x: number, y: number): {x: number; y: number} {
  return {x:m[0]*x+m[2]*y+m[4], y:m[1]*x+m[3]*y+m[5]};
}
function near(a: number, b: number, tolerance=1e-5): boolean {
  return Math.abs(a-b) <= tolerance*Math.max(1, Math.abs(a), Math.abs(b));
}
function similarity(m: Matrix, id: string): {scale: number; rotation: number} {
  requireDesign(m.length===6 && m.every(Number.isFinite), 'invalid_transform', 'Transform must contain six finite numbers.', id);
  const scale = Math.hypot(m[0], m[1]);
  requireDesign(scale>0 && near(scale,Math.hypot(m[2],m[3])) && near(m[0]*m[2]+m[1]*m[3],0) && m[0]*m[3]-m[1]*m[2]>0,
    'unsupported_transform', 'Skew, reflection and nonuniform ancestor scaling cannot be represented faithfully.', id);
  return {scale,rotation:Math.atan2(m[1],m[0])*180/Math.PI};
}
export function cssColor(c: RGBA): string {
  requireDesign([c.red,c.green,c.blue,c.alpha].every(n=>Number.isFinite(n)&&n>=0&&n<=1), 'invalid_color', 'Invalid RGBA color.');
  return `rgba(${Math.round(c.red*255)},${Math.round(c.green*255)},${Math.round(c.blue*255)},${c.alpha})`;
}
/** Engine offsets count Unicode code points; Express ranges count UTF-16 units. */
export function codePointOffset(text: string, utf16Offset: number): number {
  requireDesign(Number.isInteger(utf16Offset)&&utf16Offset>=0&&utf16Offset<=text.length, 'invalid_text_range', 'Text range is outside the content.');
  const previous=text.charCodeAt(utf16Offset-1), next=text.charCodeAt(utf16Offset);
  requireDesign(!(previous>=0xD800&&previous<=0xDBFF&&next>=0xDC00&&next<=0xDFFF), 'invalid_text_range', 'Text range splits a Unicode character.');
  return Array.from(text.slice(0,utf16Offset)).length;
}
function textFields(node: NativeNode, scale: number, pixelsPerPoint: number): Record<string,unknown> {
  const text=node.text??'', runs=node.runs??[], first=runs[0];
  requireDesign(text.length>0&&first, 'unsupported_text', 'Empty text or missing character styles cannot be imported.',node.id);
  requireDesign(node.layout==='autoWidth'||node.layout==='autoHeight','unsupported_text_layout','Only standalone auto-width and auto-height text is supported.',node.id);
  const size=first.fontSize*pixelsPerPoint*scale;
  requireDesign(size>=1&&Number.isFinite(size),'invalid_font_size','Font size must be at least one pixel.',node.id);
  let offset=0;
  const styleRuns: Record<string,unknown>[]=[];
  for (const run of runs) {
    requireDesign(Number.isInteger(run.length)&&run.length>0,'invalid_text_range','Character ranges must have positive lengths.',node.id);
    requireDesign(run.font.availableForEditing,'unavailable_font','The font is not available for editing in this document.',node.id);
    requireDesign(run.font.postscriptName===first.font.postscriptName&&near(run.fontSize,first.fontSize)&&near(run.letterSpacing,first.letterSpacing),
      'mixed_text_metrics','Mixed font faces, sizes or tracking are outside the current engine schema.',node.id);
    requireDesign(!run.underline&&!run.link&&run.baselineShift===0,'unsupported_text_style','Underline, links and baseline shifts are not representable.',node.id);
    const start=codePointOffset(text,offset), end=codePointOffset(text,offset+run.length);
    if(cssColor(run.color)!==cssColor(first.color)) styleRuns.push({start,end,color:cssColor(run.color)});
    offset+=run.length;
  }
  requireDesign(offset===text.length,'invalid_text_range','Character ranges must cover the complete literal text.',node.id);
  requireDesign(node.alignment&&['left','center','right'].includes(node.alignment),'unsupported_alignment','Text alignment is not representable.',node.id);
  return {content:text,font_family:first.font.postscriptName,font_size:size,font_weight:'regular',font_style:'normal',
    color:cssColor(first.color),text_align:node.alignment,line_height:(node.lineSpacing??1)*1.2,
    letter_spacing:first.letterSpacing/1000*size,...(styleRuns.length?{style_runs:styleRuns}:{}),
    adobe_font_family:first.font.family,adobe_font_style:first.font.style,adobe_postscript_name:first.font.postscriptName};
}
/** Fingerprint is only a stale-edit guard, not a cryptographic identity. */
export function fingerprint(value: unknown): string {
  let hash=2166136261;const serialized=JSON.stringify(value);
  for(let index=0;index<serialized.length;index++) { hash^=serialized.charCodeAt(index); hash=Math.imul(hash,16777619); }
  return (hash>>>0).toString(16).padStart(8,'0');
}
export function importPage(input: NativePage, pixelsPerPoint=96/72): Snapshot {
  const page=copy(input);
  requireDesign(Number.isFinite(pixelsPerPoint)&&pixelsPerPoint>0,'invalid_font_units','Invalid font unit conversion.');
  requireDesign(Number.isInteger(page.width)&&page.width>0&&Number.isInteger(page.height)&&page.height>0,'invalid_canvas','Canvas dimensions must be positive integer pixels.');
  const ids=new Set<string>(), bindings:Binding[]=[], elements:Element[]=[];
  // Canvas fill is explicit; it is never inferred from a screenshot.
  const backgroundId=`adobe-artboard-background:${page.artboardId}`;
  elements.push({id:backgroundId,type:'shape',x:0,y:0,width:page.width,height:page.height,rotation:0,opacity:1,z_index:0,shape_type:'rectangle',fill_color:page.background,adobe_artboard_background:true});
  function walk(node:NativeNode,parent:Matrix,path:string[],root:string,inheritedOpacity:number):void {
    requireDesign(['group','text','rectangle','ellipse','line','image'].includes(node.kind),'unsupported_node','Unsupported native node; no element was dropped.',node.id);
    requireDesign(!ids.has(node.id)&&node.id!==backgroundId,'duplicate_id','Native IDs must be unique.',node.id); ids.add(node.id);
    requireDesign(node.bounds.width>0&&node.bounds.height>0&&Object.values(node.bounds).every(Number.isFinite),'invalid_bounds','Node bounds must have positive finite dimensions.',node.id);
    requireDesign(Number.isFinite(node.opacity)&&node.opacity>=0&&node.opacity<=1,'invalid_opacity','Invalid node opacity.',node.id);
    const world=multiply(parent,node.transform), metric=similarity(world,node.id);
    const opacity=inheritedOpacity*node.opacity;
    if(node.kind==='group') {
      requireDesign((node.children?.length??0)>0,'empty_group','Empty native groups are not supported.',node.id);
      requireDesign(near(node.opacity,1),'group_compositing','Group opacity cannot be flattened into child opacity faithfully.',node.id);
      for(const child of node.children!) walk(child,world,[...path,node.id],root,opacity);
      return;
    }
    const center=point(world,node.bounds.x+node.bounds.width/2,node.bounds.y+node.bounds.height/2);
    const width=node.bounds.width*metric.scale,height=node.bounds.height*metric.scale;
    const element:Element={id:node.id,type:node.kind==='text'?'text':node.kind==='image'?'image':'shape',
      x:center.x-width/2,y:center.y-height/2,width,height,rotation:metric.rotation,opacity,z_index:elements.length,
      ...(path.length?{group_id:path[0]}:{})};
    if(node.kind==='text') Object.assign(element,textFields(node,metric.scale,pixelsPerPoint));
    else if(node.kind==='image') {
      requireDesign(node.assetId,'missing_image','The image must be exported as an actual document asset.',node.id);
      Object.assign(element,{asset_id:node.assetId,fit:'fill'});
    } else Object.assign(element,{shape_type:node.kind,fill_color:node.fill??'rgba(0,0,0,0)',
      stroke_color:node.stroke?.color??null,stroke_width:(node.stroke?.width??0)*metric.scale,corner_radius:(node.radius??0)*metric.scale});
    elements.push(element); bindings.push({id:node.id,root_id:root,path:[...path,node.id]});
  }
  for(const node of page.nodes) walk(node,identity,[],node.id,1);
  requireDesign(bindings.length>0,'empty_page','The current page has no supported editable elements.');
  const source:Design={canvas:{width:page.width,height:page.height},elements};
  return {snapshot_id:fingerprint(page),page,source,node_bindings:bindings,group_tree:copy(page.nodes),
    required_fonts:[...new Set(elements.filter(e=>e.type==='text').map(e=>String(e.font_family)))].sort(),font_pixels_per_point:pixelsPerPoint};
}
export function resizeRequest(snapshot:Snapshot,target:{width:number;height:number},assets:AssetResource[],fonts:FontResource[]) {
  requireDesign(Number.isInteger(target.width)&&target.width>0&&Number.isInteger(target.height)&&target.height>0,'invalid_canvas','Target dimensions must be positive integer pixels.');
  for(const name of snapshot.required_fonts) requireDesign(fonts.some(f=>f.font_family===name&&f.postscript_name===name&&f.data_base64.length>0),
    'missing_font_bytes',`Supply the licensed or open font file for ${name}; exact preview cannot use a substitute.`);
  for(const element of snapshot.source.elements.filter(e=>e.type==='image')) requireDesign(assets.some(a=>a.asset_id===element.asset_id&&a.data_base64.length>0),
    'missing_asset_bytes',`Missing exported image bytes for ${String(element.asset_id)}.`,element.id);
  return {source:copy(snapshot.source),target_canvas:{...target},assets:copy(assets),fonts:copy(fonts),
    document:{snapshot_id:snapshot.snapshot_id,page_id:snapshot.page.pageId,artboard_id:snapshot.page.artboardId,
      node_bindings:copy(snapshot.node_bindings),group_tree:copy(snapshot.group_tree),font_pixels_per_point:snapshot.font_pixels_per_point}};
}
export interface RootTransform { rootId:string; scale:number; translation:{x:number;y:number}; sourceCenter:{x:number;y:number}; targetCenter:{x:number;y:number} }
export interface WritePlan { transforms:RootTransform[]; background:string; canvas:{width:number;height:number} }
/** Native groups remain intact. Refuse any output that would require deforming
 * a group's relative spacing, text, styles, z-order, crop or child proportions.
 * First integration deliberately writes only similarity transforms per root.
 */
export function planWriteback(snapshot:Snapshot,result:Design):WritePlan {
  requireDesign(result.canvas&&Number.isInteger(result.canvas.width)&&result.canvas.width>0&&Number.isInteger(result.canvas.height)&&result.canvas.height>0,'invalid_canvas','Invalid result canvas.');
  const byId=new Map(result.elements.map(e=>[e.id,e]));
  requireDesign(byId.size===result.elements.length&&byId.size===snapshot.source.elements.length&&snapshot.source.elements.every(e=>byId.has(e.id)),
    'changed_membership','Output must contain each original element exactly once.');
  const expectedOrder=snapshot.source.elements.map(e=>e.id).join('\0');
  const actualOrder=[...result.elements].sort((a,b)=>a.z_index-b.z_index).map(e=>e.id).join('\0');
  requireDesign(expectedOrder===actualOrder,'changed_stacking','Output changed native stacking order.');
  const background=snapshot.source.elements[0]!;
  const bg=byId.get(background.id)!;
  requireDesign(bg.type==='shape'&&bg.shape_type==='rectangle'&&bg.fill_color===background.fill_color&&bg.x===0&&bg.y===0&&near(bg.width,result.canvas.width)&&near(bg.height,result.canvas.height)&&near(bg.rotation??0,0)&&near(bg.opacity??1,1),
    'changed_background','Artboard fill must remain the same and fill the complete target canvas.');
  const transforms:RootTransform[]=[];
  for(const root of snapshot.page.nodes) {
    const ids=snapshot.node_bindings.filter(b=>b.root_id===root.id).map(b=>b.id);
    const first=snapshot.source.elements.find(e=>e.id===ids[0])!, output=byId.get(first.id)!;
    const scale=output.width/first.width;
    requireDesign(Number.isFinite(scale)&&scale>0,'invalid_scale','Invalid output scale.',root.id);
    const sourceCenter={x:first.x+first.width/2,y:first.y+first.height/2};
    const targetCenter={x:output.x+output.width/2,y:output.y+output.height/2};
    const translation={x:targetCenter.x-scale*sourceCenter.x,y:targetCenter.y-scale*sourceCenter.y};
    for(const id of ids) {
      const before=snapshot.source.elements.find(e=>e.id===id)!, after=byId.get(id)!;
      requireDesign(Object.values({x:after.x,y:after.y,width:after.width,height:after.height,rotation:after.rotation,opacity:after.opacity}).every(Number.isFinite),
        'invalid_geometry','Output contains non-finite geometry.',id);
      requireDesign(near(after.width,before.width*scale)&&near(after.height,before.height*scale)&&near(after.x,before.x*scale+translation.x)&&near(after.y,before.y*scale+translation.y)&&near(after.rotation??0,before.rotation??0),
        'native_group_deformation','Editable native writeback requires one uniform scale and translation per original root group; this layout changes its internal geometry.',id);
      for(const key of ['font_size','letter_spacing','stroke_width','corner_radius']) if(typeof before[key]==='number')
        requireDesign(typeof after[key]==='number'&&near(Number(after[key]),Number(before[key])*scale),'changed_metric',`${key} must scale with the complete native group.`,id);
      const mutable=new Set(['x','y','width','height','font_size','letter_spacing','stroke_width','corner_radius']);
      for(const key of Object.keys(before)) if(!mutable.has(key)) requireDesign(JSON.stringify(after[key])===JSON.stringify(before[key]),'changed_native_style',`Output changed ${key}; native styles and literal content must be preserved.`,id);
    }
    transforms.push({rootId:root.id,scale,translation,sourceCenter,targetCenter});
  }
  return {transforms,background:String(background.fill_color),canvas:{...result.canvas}};
}
/** Check actual native geometry after writing; this is not a raster comparison. */
export function verifyNativeResult(expected:Design,actual:Design):void {
  const byId=new Map(actual.elements.map(e=>[e.id,e]));
  requireDesign(byId.size===expected.elements.length,'native_membership_mismatch','Adobe changed the output membership.');
  for(const before of expected.elements) {
    const after=byId.get(before.id);
    requireDesign(after,'native_membership_mismatch','Adobe changed an output element.',before.id);
    for(const key of ['x','y','width','height','rotation','opacity','font_size','letter_spacing','stroke_width','corner_radius']) {
      if(typeof before[key]==='number') requireDesign(typeof after[key]==='number'&&Math.abs(Number(before[key])-Number(after[key]))<=0.05,
        'native_resize_mismatch',`Adobe's actual ${key} differs from the planned result; the copy was rolled back.`,before.id);
    }
    for(const key of ['content','font_family','color','style_runs','text_align','line_height','fill_color','stroke_color','asset_id','shape_type']) {
      if(before[key]!==undefined) requireDesign(JSON.stringify(after[key])===JSON.stringify(before[key]),'native_style_mismatch',`Adobe changed ${key}; the copy was rolled back.`,before.id);
    }
  }
}

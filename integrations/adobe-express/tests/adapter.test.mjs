import test from 'node:test';
import assert from 'node:assert/strict';
import { importPage, planWriteback, resizeRequest, codePointOffset, verifyNativeResult, multiply, point, previewSource } from '../dist/shared/adapter.js';

const white={red:1,green:1,blue:1,alpha:1}, orange={red:1,green:.4,blue:0,alpha:1};
const identity=[1,0,0,1,0,0];
function run(length,changes={}) {return {length,fontSize:24,font:{postscriptName:'SourceSans3-Semibold',family:'Source Sans 3',style:'Semibold',availableForEditing:true},color:white,letterSpacing:30,underline:false,baselineShift:0,...changes};}
function text(id='heading') {return {id,kind:'text',bounds:{x:-2,y:5,width:240,height:40},transform:[1,0,0,1,30,40],opacity:1,text:'A😀B',runs:[run(1),run(2,{color:orange}),run(1)],alignment:'left',layout:'autoWidth',lineSpacing:1};}
function rectangle(id='badge') {return {id,kind:'rectangle',bounds:{x:0,y:0,width:100,height:50},transform:[1,0,0,1,50,110],opacity:1,fill:'rgba(240,80,30,1)',radius:8};}
function page() {return {pageId:'opaque/page',artboardId:'opaque/board',width:600,height:800,background:'rgba(20,20,20,1)',nodes:[{id:'opaque:campaign',kind:'group',bounds:{x:20,y:40,width:280,height:130},transform:[1,0,0,1,40,50],opacity:1,children:[text(),{id:'nested',kind:'group',bounds:{x:50,y:110,width:100,height:50},transform:identity,opacity:1,children:[rectangle()]}]}]};}
function reflow(snapshot,scale=2,dx=20,dy=30) {
  const design=structuredClone(snapshot.source);design.canvas={width:1300,height:1700};
  for(const element of design.elements) {
    if(element.adobe_artboard_background){element.width=design.canvas.width;element.height=design.canvas.height;continue;}
    element.x=element.x*scale+dx;element.y=element.y*scale+dy;element.width*=scale;element.height*=scale;
    for(const name of ['font_size','letter_spacing','corner_radius','stroke_width'])if(typeof element[name]==='number')element[name]*=scale;
  }
  return design;
}
function rejectsCode(action,code) {assert.throws(action,error=>error.code===code,code);}

test('imports opaque IDs and nested author groups without flattening their ownership',()=>{
  const original=page(),before=structuredClone(original),snapshot=importPage(original);
  assert.equal(snapshot.source.elements.length,3);
  assert.deepEqual(snapshot.source.elements.slice(1).map(e=>e.group_id),['opaque:campaign','opaque:campaign']);
  assert.deepEqual(snapshot.node_bindings[1].path,['opaque:campaign','nested','badge']);
  assert.equal(snapshot.group_tree[0].children[1].kind,'group');
  snapshot.group_tree[0].children[0].text='changed copy';
  assert.deepEqual(original,before,'import and returned metadata must never mutate native input');
});
test('keeps exact PostScript face aliases and literal Unicode content',()=>{
  const snapshot=importPage(page()),heading=snapshot.source.elements[1];
  assert.equal(heading.font_family,'SourceSans3-Semibold');
  assert.equal(heading.font_weight,'regular','regular selects the exact supplied face, not a synthesized weight');
  assert.equal(heading.adobe_font_family,'Source Sans 3');assert.equal(heading.adobe_font_style,'Semibold');
  assert.equal(heading.content,'A😀B');assert.equal(heading.font_size,32);
  assert.equal(heading.letter_spacing,.96);
  assert.deepEqual(heading.style_runs,[{start:1,end:2,color:'rgba(255,102,0,1)'}]);
});
test('converts UTF-16 style ranges to engine code points and rejects split surrogates',()=>{
  assert.equal(codePointOffset('A😀B',3),2);assert.equal(codePointOffset('A😀B',4),3);
  rejectsCode(()=>codePointOffset('A😀B',2),'invalid_text_range');
  rejectsCode(()=>codePointOffset('A😀B',5),'invalid_text_range');
});
test('composes rotated ancestor frames around actual local centers',()=>{
  const input=page();input.nodes[0].transform=[0,1,-1,0,300,50];
  const result=importPage(input).source.elements[1];
  // Native text center (118,25), translated locally (30,40), then rotated parent.
  assert.equal(result.x,115);assert.equal(result.y,178);assert.equal(result.rotation,90);
  assert.equal(result.width,240);assert.equal(result.height,40);
  assert.deepEqual(point(multiply([0,1,-1,0,300,50],[1,0,0,1,30,40]),118,25),{x:235,y:198});
});
test('roundtrip plan preserves every member, size, font and gap ratio with a common root scale',()=>{
  const snapshot=importPage(page()),before=JSON.stringify(snapshot),result=reflow(snapshot,1.5,17,21);
  const plan=planWriteback(snapshot,result);
  assert.equal(plan.transforms.length,1);assert.equal(plan.transforms[0].rootId,'opaque:campaign');
  assert.equal(plan.transforms[0].scale,1.5);assert.deepEqual(plan.transforms[0].translation,{x:17,y:21});
  const first=result.elements[1],second=result.elements[2],a=snapshot.source.elements[1],b=snapshot.source.elements[2];
  assert.equal(second.y-first.y,(b.y-a.y)*1.5);assert.equal(second.width/first.width,b.width/a.width);
  assert.equal(first.font_size/a.font_size,1.5);assert.equal(second.corner_radius/b.corner_radius,1.5);
  assert.equal(JSON.stringify(snapshot),before);verifyNativeResult(result,structuredClone(result));
});
test('declines independent group-member movement and text scale drift',()=>{
  const snapshot=importPage(page()),detached=reflow(snapshot);detached.elements[2].x+=12;
  rejectsCode(()=>planWriteback(snapshot,detached),'native_group_deformation');
  const fonts=reflow(snapshot);fonts.elements[1].font_size+=2;
  rejectsCode(()=>planWriteback(snapshot,fonts),'changed_metric');
});
test('declines changed literal text, ownership, color ranges and stacking',()=>{
  const snapshot=importPage(page());
  for(const [name,value] of [['content','replaced'],['group_id','another-offer'],['style_runs',[]]]) {
    const result=reflow(snapshot);result.elements[1][name]=value;
    rejectsCode(()=>planWriteback(snapshot,result),'changed_native_style');
  }
  const order=reflow(snapshot);order.elements[1].z_index=9;
  rejectsCode(()=>planWriteback(snapshot,order),'changed_stacking');
});
test('declines duplicate or missing output membership',()=>{
  const snapshot=importPage(page()),missing=reflow(snapshot);missing.elements.pop();
  rejectsCode(()=>planWriteback(snapshot,missing),'changed_membership');
  const duplicate=reflow(snapshot);duplicate.elements[2]=structuredClone(duplicate.elements[1]);
  rejectsCode(()=>planWriteback(snapshot,duplicate),'changed_membership');
});
test('requires exact supplied font bytes before contacting the engine',()=>{
  const snapshot=importPage(page());
  rejectsCode(()=>resizeRequest(snapshot,{width:1080,height:1920},[],[]),'missing_font_bytes');
  rejectsCode(()=>resizeRequest(snapshot,{width:1080,height:1920},[],[{font_family:'Source Sans 3',postscript_name:'SourceSans3-Semibold',weight:'regular',style:'normal',data_base64:'AA=='}]),'missing_font_bytes');
  const font={font_family:'SourceSans3-Semibold',postscript_name:'SourceSans3-Semibold',weight:'regular',style:'normal',data_base64:'AA=='};
  const request=resizeRequest(snapshot,{width:1080,height:1920},[],[font]);
  assert.equal(request.document.snapshot_id,snapshot.snapshot_id);assert.equal(request.fonts[0].postscript_name,'SourceSans3-Semibold');
  request.source.elements[1].content='local copy';assert.equal(snapshot.source.elements[1].content,'A😀B');
});
test('requires actual per-document image bytes and does not substitute brandkit assets',()=>{
  const input=page();input.nodes=[{id:'picture',kind:'image',bounds:{x:0,y:0,width:200,height:150},transform:identity,opacity:1,assetId:'document-picture.png'}];
  const snapshot=importPage(input);
  rejectsCode(()=>resizeRequest(snapshot,{width:500,height:500},[],[]),'missing_asset_bytes');
  const resource={asset_id:'document-picture.png',mime_type:'image/png',data_base64:'AA=='};
  const request=resizeRequest(snapshot,{width:500,height:500},[resource],[]);
  assert.deepEqual(request.assets,[resource]);assert.equal(request.source.elements[1].asset_id,'document-picture.png');
});
test('fails unsupported transform and group compositing rather than approximating',()=>{
  for(const transform of [[2,0,0,1,0,0],[-1,0,0,1,0,0],[1,0,.2,1,0,0]]){
    const input=page();input.nodes[0].transform=transform;
    rejectsCode(()=>importPage(input),'unsupported_transform');
  }
  const input=page();input.nodes[0].opacity=.6;
  rejectsCode(()=>importPage(input),'group_compositing');
});
test('fails unsupported mixed metrics, unavailable faces and decoration',()=>{
  for(const [change,code] of [[{fontSize:25},'mixed_text_metrics'],[{letterSpacing:31},'mixed_text_metrics'],[{font:{postscriptName:'DifferentFace',family:'Different',style:'Regular',availableForEditing:true}},'mixed_text_metrics'],[{font:{postscriptName:'SourceSans3-Semibold',family:'Source Sans 3',style:'Semibold',availableForEditing:false}},'unavailable_font'],[{underline:true},'unsupported_text_style'],[{baselineShift:1},'unsupported_text_style']]){
    const input=page();Object.assign(input.nodes[0].children[0].runs[1],change);
    rejectsCode(()=>importPage(input),code);
  }
});
test('range validation catches the reported style_runs-outside-content class of error',()=>{
  const input=page();input.nodes[0].children[0].runs[2].length=2;
  rejectsCode(()=>importPage(input),'invalid_text_range');
  const truncated=page();truncated.nodes[0].children[0].runs.pop();
  rejectsCode(()=>importPage(truncated),'invalid_text_range');
});
test('re-reading native output catches size and font changes despite correct root bounds',()=>{
  const expected=reflow(importPage(page())),actual=structuredClone(expected);
  actual.elements[1].font_size+=1;
  rejectsCode(()=>verifyNativeResult(expected,actual),'native_resize_mismatch');
  actual.elements[1].font_size=expected.elements[1].font_size;actual.elements[2].x+=2;
  rejectsCode(()=>verifyNativeResult(expected,actual),'native_resize_mismatch');
});
test('snapshot identities change with content, geometry and native font style',()=>{
  const input=page(),baseline=importPage(input).snapshot_id;
  input.nodes[0].children[0].runs[0].font.style='Display Semibold';
  assert.notEqual(importPage(input).snapshot_id,baseline);
  const translated=page();translated.nodes[0].transform[4]+=1;
  assert.notEqual(importPage(translated).snapshot_id,baseline);
  const emoji=page();emoji.nodes[0].children[0].text='A😁B';
  assert.notEqual(importPage(emoji).snapshot_id,baseline,'both UTF-16 halves must contribute to the snapshot identity');
});
test('rejects invalid canvases and duplicate native IDs',()=>{
  const input=page();input.width=0;rejectsCode(()=>importPage(input),'invalid_canvas');
  const duplicates=page();duplicates.nodes.push(text('heading'));rejectsCode(()=>importPage(duplicates),'duplicate_id');
  const snapshot=importPage(page());rejectsCode(()=>resizeRequest(snapshot,{width:100.5,height:500},[],[]),'invalid_canvas');
});
test('accepts engine PNG data previews and rejects active or external preview sources',()=>{
  const png='data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO7+0S8AAAAASUVORK5CYII=';
  assert.equal(previewSource(png,'http://127.0.0.1:8765'),png);
  assert.equal(previewSource('/preview.png','http://127.0.0.1:8765'),'http://127.0.0.1:8765/preview.png');
  for(const source of ['data:image/svg+xml;base64,PHN2Zz4=', 'data:image/png;base64,not-a-png','https://example.com/preview.png','javascript:alert(1)'])
    rejectsCode(()=>previewSource(source,'http://127.0.0.1:8765'),'invalid_preview');
});

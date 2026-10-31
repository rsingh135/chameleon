import test from 'node:test';
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {ViewerSession,canvas,task,engineOrigin,resultBundle,pngBytes,validateResult} from '../dist/shared/viewer.js';

const png='data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII=';
function request(){return {request_id:'opaque-request',source:{canvas:{width:1,height:1},elements:[{id:'opaque-id',type:'shape',x:0,y:0,width:1,height:1,shape_type:'rectangle',fill_color:'#fff'}]},target_canvas:{width:1,height:1}};}
function result(){const r=request();return {...r,schema_version:1,design:structuredClone(r.source),source_png:png,output_png:png,report:{violations:[]}};}

test('an engine-result bundle opens without network and preserves the exact engine JSON',()=>{
  const original=result(),session=new ViewerSession();session.openBundle(original);
  assert.deepEqual(session.result.design,original.design);assert.equal(session.result.output_png,png);
  session.result.design.elements[0].width=9;assert.equal(original.design.elements[0].width,1);
});
test('changing source or target invalidates exports and refuses late responses',()=>{
  const session=new ViewerSession();session.load(task(request()));
  const first=session.begin({width:1,height:1},'opaque-request');assert.equal(session.accept(result(),first.request,first.generation),true);
  session.invalidate();assert.equal(session.result,undefined);assert.equal(session.accept(result(),first.request,first.generation),false);
  session.load(task(request()));assert.equal(session.accept(result(),first.request,first.generation),false);
});
for(const [name,change,pattern] of [
  ['wrong request',r=>r.request_id='another',/different request/],
  ['wrong target',r=>r.target_canvas.width=2,/different target/],
  ['dropped element',r=>r.design.elements=[],/between 1 and 200/],
  ['duplicate element',r=>r.design.elements.push(structuredClone(r.design.elements[0])),/unique/],
  ['changed identity',r=>r.design.elements[0].fill_color='#000',/changed/],
  ['off-canvas element',r=>r.design.elements[0].x=.1,/outside/],
  ['nonnumeric rotation',r=>r.design.elements[0].rotation='invalid',/invalid rotation/],
  ['rotated overflow',r=>{r.source.elements[0].rotation=45;r.design.elements[0].rotation=45;},/outside/],
  ['bad PNG dimensions',r=>r.output_png=png.replace('AAAAEAAAAB','AAAAIAAAAB'),/dimensions/],
  ['engine violation',r=>r.report.violations=['off_canvas'],/contract violation/],
])test(`refuses ${name}`,()=>{const value=result();change(value);assert.throws(()=>validateResult(value,request()),pattern);});
test('checks PNG type and dimensions before exports',()=>{assert.equal(pngBytes(png,{width:1,height:1}).length,68);assert.throws(()=>pngBytes('https://external.example/image.png',{width:1,height:1}),/PNG data URL/);assert.throws(()=>pngBytes(png,{width:2,height:1}),/dimensions/);});
test('limits canvas work and accepts only local engine origins',()=>{
  for(const value of [{width:0,height:1},{width:1.2,height:1},{width:8192,height:8192}])assert.throws(()=>canvas(value));
  assert.equal(engineOrigin('http://127.0.0.1:8765'),'http://127.0.0.1:8765');
  for(const value of ['https://remote.example','http://localhost@remote.example','http://127.0.0.1/path'])assert.throws(()=>engineOrigin(value));
});
test('file bundles cannot turn a foreground contain-image into a cropped cover image',()=>{
  const value=result(),image={id:'art',type:'image',x:0,y:0,width:1,height:1,asset_id:'supplied.png',fit:'contain'};
  value.source.elements=[image];value.design.elements=[{...image,fit:'cover',width:.5}];
  assert.throws(()=>validateResult(value,{...request(),source:value.source}),/\.fit/);
  value.design.elements=[{...image,image_position:[0,0]}];
  assert.throws(()=>validateResult(value,{...request(),source:value.source}),/image_position/);
  value.source.elements=[{...image,fit:undefined}];value.design.elements=[{...image,fit:'cover'}];
  assert.throws(()=>validateResult(value,{...request(),source:value.source}),/\.fit/);
});
test('production manifest and viewer runtime have no experimental/native document entrypoint',async()=>{
  const manifest=JSON.parse(await readFile(new URL('../dist/manifest.json',import.meta.url),'utf8'));
  assert.equal('experimentalApis' in manifest.requirements,false);assert.equal('documentSandbox' in manifest.entryPoints[0],false);
  const main=await readFile(new URL('../dist/ui/main.js',import.meta.url),'utf8');
  assert.doesNotMatch(main,/apiProxy|express-document-sdk|cloneInPlace|createEditableCopy/);
  assert.match(main,/app\.document\.addImage/);assert.match(main,/api\/addon\/resize/);
});

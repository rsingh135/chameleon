import test from 'node:test';
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';

const png='data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII=';
function bundle(label='first') {const source={canvas:{width:1,height:1},elements:[{id:label,type:'shape',x:0,y:0,width:1,height:1,shape_type:'rectangle',fill_color:'#fff'}]};return {schema_version:1,request_id:label,source,target_canvas:source.canvas,design:structuredClone(source),source_png:png,output_png:png,report:{violations:[]}};}
class Element {
  constructor(){this.value='';this.hidden=false;this.disabled=false;this.textContent='';this.events={};this.children=[];this.files=[];}
  addEventListener(type,listener){(this.events[type]??=[]).push(listener);}
  async fire(type){for(const fn of this.events[type]??[])await fn({preventDefault(){}});await new Promise(resolve=>setImmediate(resolve));}
  replaceChildren(...children){this.children=children;}
  append(child){this.children.push(child);}
  click(){this.clicked=true;}
  remove(){}
}
let imports=0;
async function setup(){
  const ids=['resize','cancel','download-json','download-png','place','status','error','comparison','preview-state','width','height','preset','summary','source-image','result-image','source-size','result-size','warnings','elements','task-file','result-file','sample','engine','reflow','host-status'];
  const nodes=Object.fromEntries(ids.map(id=>[id,new Element()]));nodes.comparison.hidden=true;nodes.engine.value='http://127.0.0.1:8765';
  const saved={document:globalThis.document,Image:globalThis.Image,fetch:globalThis.fetch};
  globalThis.document={getElementById:id=>nodes[id],createElement:()=>new Element(),body:new Element()};
  globalThis.Image=class {naturalWidth=1;naturalHeight=1;async decode(){if(this.src!==png)throw new Error('Image decoding failed');}};
  globalThis.__viewerSDK={ready:Promise.resolve(),app:{document:{async addImage(blob){globalThis.__placed=blob;}}}};
  let source=await readFile(new URL('../dist/ui/main.js',import.meta.url),'utf8');
  source=source.replace(/import addOnUISdk from ['"]https:[^;]+;/,'const addOnUISdk = globalThis.__viewerSDK;');
  source=source.replace(/(['"])\.\.\/shared\/viewer\.js\1/,JSON.stringify(new URL('../dist/shared/viewer.js',import.meta.url).href));
  await import('data:text/javascript;base64,'+Buffer.from(source+`\n// execution ${++imports}`).toString('base64'));
  await new Promise(resolve=>setImmediate(resolve));
  return {nodes,async open(value){nodes['result-file'].files=[{name:'engine-result.json',size:100,text:async()=>JSON.stringify(value)}];await nodes['result-file'].fire('change');},restore(){Object.assign(globalThis,saved);delete globalThis.__viewerSDK;delete globalThis.__placed;}};
}
test('actual viewer enables exports only after imported previews decode, then invalidates on target edit',async()=>{
  const ui=await setup();try {
    assert.equal(ui.nodes['download-json'].disabled,true);await ui.open(bundle());
    assert.equal(ui.nodes['download-json'].disabled,false);assert.equal(ui.nodes['download-png'].disabled,false);assert.equal(ui.nodes.place.disabled,false);
    assert.equal(ui.nodes['result-image'].src,png);assert.equal(ui.nodes.elements.children[0].textContent.startsWith('first'),true);
    ui.nodes.width.value='2';await ui.nodes.width.fire('input');
    assert.equal(ui.nodes['download-json'].disabled,true);assert.equal(ui.nodes.place.disabled,true);assert.equal(ui.nodes['preview-state'].textContent,'Previous result');
  }finally{ui.restore();}
});
test('a late file read cannot replace a newer engine result or unlock an invalid result',async()=>{
  const ui=await setup();try {
    let resolveOld;ui.nodes['result-file'].files=[{name:'old.json',size:1,text:()=>new Promise(resolve=>resolveOld=resolve)}];
    await ui.nodes['result-file'].fire('change');await ui.open(bundle('new'));
    resolveOld(JSON.stringify(bundle('old')));await new Promise(resolve=>setImmediate(resolve));
    assert.equal(ui.nodes.elements.children[0].textContent.startsWith('new'),true);assert.equal(ui.nodes['download-json'].disabled,false);
    const bad=bundle('bad');bad.output_png='data:image/png;base64,AAAA';await ui.open(bad);
    assert.equal(ui.nodes['download-json'].disabled,true);assert.equal(ui.nodes.error.hidden,false);assert.equal(ui.nodes['preview-state'].textContent,'Previous result');
  }finally{ui.restore();}
});
test('PNG placement uses the stable Adobe API and passes exact output PNG bytes',async()=>{
  const ui=await setup();try {
    await ui.open(bundle());await ui.nodes.place.fire('click');
    assert.equal(globalThis.__placed.type,'image/png');assert.deepEqual(Buffer.from(await globalThis.__placed.arrayBuffer()),Buffer.from(png.split(',')[1],'base64'));
  }finally{ui.restore();}
});

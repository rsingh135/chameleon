import test from 'node:test';
import assert from 'node:assert/strict';
import { clonePageTransaction } from '../dist/sandbox/transaction.js';

// These are small test doubles for official methods, not a real Adobe host.
function harness(options={}) {
  const events=[],clones=[];
  const original={id:'native-root',centerPointLocal:{x:20,y:10},boundsLocal:{width:40,height:20},
    localPointInNode(){return {x:120,y:210};},
    resize(){throw new Error('Original must never be resized');},
    cloneInPlace(){events.push('clone');const clone={id:'copy',boundsLocal:{width:40,height:20},centerPointLocal:{x:20,y:10},
      resize(config){events.push(['resize',config]);if(options.resizeFails)throw new Error('resize failed');},
      setPositionInParent(parent,local){events.push(['position',parent,local]);},
      removeFromParent(){events.push('remove clone');if(options.cleanupFails)throw new Error('remove clone failed');}};clones.push(clone);return clone;}};
  const destination={children:{append(node){events.push(['append',node.id]);}},fill:undefined};
  const source={children:[original],fill:{type:'solid',color:'native fill object'}};
  const page={id:'new-native-page',artboards:{first:destination}};
  const pages={addPage(canvas){events.push(['add page',canvas]);if(options.addFails)throw new Error('add page failed');return page;},remove(value){events.push(['remove page',value.id]);}};
  const plan={canvas:{width:800,height:1200},background:'rgba(0,0,0,1)',transforms:[{rootId:'native-root',scale:2,translation:{x:10,y:20},sourceCenter:{x:120,y:210},targetCenter:{x:250,y:440}}]};
  return {events,clones,original,destination,source,pages,page,plan};
}
test('creates only native clones, copies actual fill, and scales all visual details together',()=>{
  const h=harness(),before=structuredClone({id:h.original.id,bounds:h.original.boundsLocal});
  const output=clonePageTransaction(h.pages,h.source,h.plan,'proportional',page=>h.events.push(['verify',page.id]));
  assert.equal(output,h.page);assert.equal(h.destination.fill,h.source.fill);
  assert.deepEqual(h.events,[ 'clone',['add page',{width:800,height:1200}],['append','copy'],
    ['resize',{behavior:'proportional',width:80,avoidScalingVisualDetailsIfPossible:false}],['position',{x:250,y:440},{x:20,y:10}],['verify','new-native-page']]);
  assert.deepEqual({id:h.original.id,bounds:h.original.boundsLocal},before);
});
test('cleans source-side temporary clones if destination page creation fails',()=>{
  const h=harness({addFails:true});
  assert.throws(()=>clonePageTransaction(h.pages,h.source,h.plan,'proportional',()=>{}),/add page failed/);
  assert.deepEqual(h.events.slice(-1),['remove clone']);
});
test('removes the copy and new page when SDK resize or actual-leaf validation fails',()=>{
  for(const resizeFails of [true,false]) {
    const h=harness({resizeFails});
    assert.throws(()=>clonePageTransaction(h.pages,h.source,h.plan,'proportional',()=>{throw new Error('font size mismatch');}),resizeFails?/resize failed/:/font size mismatch/);
    assert.deepEqual(h.events.slice(-2),['remove clone',['remove page','new-native-page']]);
  }
});
test('attempts page cleanup even if clone cleanup fails and reports incomplete rollback',()=>{
  const h=harness({cleanupFails:true});
  assert.throws(()=>clonePageTransaction(h.pages,h.source,h.plan,'proportional',()=>{throw new Error('native validation failed');}),error=>error.code==='native_rollback_failed');
  assert.deepEqual(h.events.at(-1),['remove page','new-native-page']);
});
test('missing source roots fail before the first native mutation',()=>{
  const h=harness();h.plan.transforms[0].rootId='removed-source';
  assert.throws(()=>clonePageTransaction(h.pages,h.source,h.plan,'proportional',()=>{}),error=>error.code==='missing_native_root');
  assert.deepEqual(h.events,[]);
});

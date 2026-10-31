import test from 'node:test';
import assert from 'node:assert/strict';
import {execFileSync} from 'node:child_process';
import {mkdtemp,readFile,rm} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import {join} from 'node:path';
import {fileURLToPath} from 'node:url';

test('the actual distribution ZIP is reproducible and excludes all native development code',async()=>{
  const root=fileURLToPath(new URL('../',import.meta.url)),out=await mkdtemp(join(tmpdir(),'chameleon-addon-package-'));
  try {
    const run=()=>JSON.parse(execFileSync('python',['scripts/package.py','--out',out],{cwd:root,encoding:'utf8'}));
    const first=run(),bytes=await readFile(first.archive),second=run();
    assert.equal(first.sha256,second.sha256);assert.deepEqual(bytes,await readFile(second.archive));
    assert.ok(first.bytes<50*1024*1024);
    const inventory=JSON.parse(execFileSync('python',['-c','import json,sys,zipfile; z=zipfile.ZipFile(sys.argv[1]); print(json.dumps({"names":z.namelist(),"bad":z.testzip(),"manifest":json.loads(z.read("manifest.json"))}))',first.archive],{encoding:'utf8'}));
    assert.equal(inventory.bad,null);assert.equal(inventory.names[0],'manifest.json');
    assert.ok(inventory.names.includes('ui/index.html'));assert.ok(inventory.names.includes('shared/viewer.js'));
    assert.equal(inventory.names.filter(name=>name.startsWith('samples/')).length,3);
    assert.ok(inventory.names.every(name=>!/(sandbox|native|adapter|node_modules|^\.|\.ts$)/.test(name)));
    assert.equal(inventory.manifest.entryPoints[0].main,'ui/index.html');
    assert.equal('experimentalApis' in inventory.manifest.requirements,false);
    const icon=await readFile(join(out,'icon-144.png'));
    assert.equal(icon.readUInt32BE(16),144);assert.equal(icon.readUInt32BE(20),144);
    assert.ok((await readFile(join(out,'release-notes.txt'),'utf8')).length<=1000);
  }finally{await rm(out,{recursive:true,force:true});}
});

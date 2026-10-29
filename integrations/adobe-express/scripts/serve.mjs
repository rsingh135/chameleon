/** Serve an already-built add-on using explicitly supplied TLS material.
 * This script never creates, installs or trusts a certificate. */
import https from 'node:https';
import { readFile } from 'node:fs/promises';
import { resolve, relative, extname } from 'node:path';
import { fileURLToPath } from 'node:url';
const root=fileURLToPath(new URL('../dist',import.meta.url));
const cert=process.env.CHAMELEON_ADDON_TLS_CERT,key=process.env.CHAMELEON_ADDON_TLS_KEY;
if(!cert||!key)throw new Error('Set CHAMELEON_ADDON_TLS_CERT and CHAMELEON_ADDON_TLS_KEY to existing trusted localhost certificate/key files. No certificates are installed by this command.');
const types={'.html':'text/html','.js':'text/javascript','.css':'text/css','.json':'application/json'};
const server=https.createServer({cert:await readFile(cert),key:await readFile(key)},async(req,res)=>{
  try{const pathname=decodeURIComponent(new URL(req.url,'https://localhost').pathname),file=resolve(root,`.${pathname==='/'?'/manifest.json':pathname}`);
    if(relative(root,file).startsWith('..')){res.writeHead(403);res.end();return;}
    const body=await readFile(file);res.writeHead(200,{'Content-Type':types[extname(file)]??'application/octet-stream','Access-Control-Allow-Origin':'*','Cache-Control':'no-store'});res.end(body);
  }catch{res.writeHead(404);res.end('Not found');}
});
server.listen(5241,'127.0.0.1',()=>process.stdout.write('Chameleon add-on: https://localhost:5241/\nSideload through https://www.adobe.com/go/addon-cli\n'));

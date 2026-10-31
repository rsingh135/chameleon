/** Serve an already-built add-on using explicitly supplied TLS material.
 * This script never creates, installs or trusts a certificate. */
import https from 'node:https';
import http from 'node:http';
import { readFile } from 'node:fs/promises';
import { resolve, relative, extname } from 'node:path';
import { fileURLToPath } from 'node:url';
const root=fileURLToPath(new URL('../dist',import.meta.url));
const localQa=process.argv.includes('--http');
const cert=process.env.CHAMELEON_ADDON_TLS_CERT,key=process.env.CHAMELEON_ADDON_TLS_KEY;
if(!localQa&&(!cert||!key))throw new Error('Set CHAMELEON_ADDON_TLS_CERT and CHAMELEON_ADDON_TLS_KEY to existing trusted localhost certificate/key files. No certificates are installed by this command. Use --http only for standalone local QA on port 5242.');
const types={'.html':'text/html','.js':'text/javascript','.css':'text/css','.json':'application/json'};
const handler=async(req,res)=>{
  try{const pathname=decodeURIComponent(new URL(req.url,'https://localhost').pathname),manifest=process.argv.includes('--native')?'/manifest.native.json':'/manifest.json',file=resolve(root,`.${pathname==='/'?manifest:pathname}`);
    if(relative(root,file).startsWith('..')){res.writeHead(403);res.end();return;}
    const body=await readFile(file);res.writeHead(200,{'Content-Type':types[extname(file)]??'application/octet-stream','Access-Control-Allow-Origin':'*','Cache-Control':'no-store'});res.end(body);
  }catch{res.writeHead(404);res.end('Not found');}
};
const server=localQa?http.createServer(handler):https.createServer({cert:await readFile(cert),key:await readFile(key)},handler);
const port=localQa?5242:5241;
server.listen(port,'127.0.0.1',()=>process.stdout.write(localQa?'Chameleon standalone QA: http://127.0.0.1:5242/ui/index.html\nHTTPS is required for Adobe Express sideloading.\n':'Chameleon add-on: https://localhost:5241/\nSideload through https://www.adobe.com/go/addon-cli\n'));

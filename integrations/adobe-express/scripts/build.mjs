import { copyFile, mkdir } from 'node:fs/promises';
await mkdir(new URL('../dist/ui/', import.meta.url), { recursive: true });
for (const name of ['index.html', 'style.css']) {
  await copyFile(new URL(`../src/ui/${name}`, import.meta.url), new URL(`../dist/ui/${name}`, import.meta.url));
}
await copyFile(new URL('../manifest.json', import.meta.url), new URL('../dist/manifest.json', import.meta.url));

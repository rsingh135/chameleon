import { copyFile, mkdir } from 'node:fs/promises';
for (const folder of ['ui', 'native-ui']) {
  await mkdir(new URL(`../dist/${folder}/`, import.meta.url), { recursive: true });
  for (const name of ['index.html', 'style.css']) {
    await copyFile(new URL(`../src/${folder}/${name}`, import.meta.url), new URL(`../dist/${folder}/${name}`, import.meta.url));
  }
}
await copyFile(new URL('../manifest.json', import.meta.url), new URL('../dist/manifest.json', import.meta.url));
await copyFile(new URL('../manifest.native.json', import.meta.url), new URL('../dist/manifest.native.json', import.meta.url));
await mkdir(new URL('../dist/samples/', import.meta.url), { recursive: true });
for (const level of ['level1_warmup', 'level2_core', 'level3_advanced']) {
  const file = `chameleon_sample_${level}.json`;
  await copyFile(new URL(`../../../input/${file}`, import.meta.url), new URL(`../dist/samples/${file}`, import.meta.url));
}

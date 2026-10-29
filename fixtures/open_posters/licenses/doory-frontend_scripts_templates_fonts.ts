/**
 * Node side of the font handling: makes sure the woff2 files of every family in `fontPalette.ts` are on disk and
 * describes them the way the API's `GET /fonts/` does, so the editor's own `FontManager` loads them unchanged.
 *
 * The files come from the same place `manage.py sync_fonts` gets them (the Fontsource npm packages on jsDelivr, file
 * names `<id>-latin-<weight>-<style>.woff2`), or from a directory that already has them (`--fonts-dir`, the layout
 * `sync_fonts --from-dir` reads). Nothing is committed.
 */
import { existsSync } from 'node:fs'
import { mkdir, readFile, writeFile } from 'node:fs/promises'
import { join } from 'node:path'
import { FONT_PICKS, type FontPick } from './fontPalette.ts'

const API = 'https://api.fontsource.org/v1/fonts'
const CDN = 'https://cdn.jsdelivr.net/npm/@fontsource'
const MAX_FILE_BYTES = 5 * 1024 * 1024

/** What the browser side needs: the catalog entries plus the served URL prefix. */
export interface FontCatalogEntry {
  id: string
  family: string
  category: string
  scripts: string[]
  is_variable: boolean
  is_user_font: boolean
  popularity: number
  unicode_ranges: Record<string, string>
  variants: { weight: number; style: 'normal' | 'italic'; url: string; subsets: [] }[]
}

interface FontMeta {
  npmVersion?: string
}

const fileName = (pick: FontPick, weight: number, style: string): string => `${pick.id}-latin-${weight}-${style}.woff2`

async function download(url: string): Promise<Buffer> {
  let lastError: unknown
  for (let attempt = 0; attempt < 3; attempt++) {
    try {
      const response = await fetch(url)
      if (!response.ok) throw new Error(`HTTP ${response.status}`)
      const bytes = Buffer.from(await response.arrayBuffer())
      if (bytes.length > MAX_FILE_BYTES) throw new Error('file too large')
      return bytes
    } catch (error) {
      lastError = error
      await new Promise((resolve) => setTimeout(resolve, 500 * 2 ** attempt))
    }
  }
  throw new Error(`Could not download ${url}: ${String(lastError)}`)
}

async function version(pick: FontPick): Promise<string> {
  try {
    const response = await fetch(`${API}/${pick.id}`)
    const meta = (await response.json()) as FontMeta
    if (typeof meta.npmVersion === 'string' && /^\d+\.\d+\.\d+$/.test(meta.npmVersion)) return meta.npmVersion
  } catch {
    // Offline metadata: fall back to the major version sync_fonts uses.
  }
  return '5'
}

function wantedFaces(pick: FontPick): { weight: number; style: 'normal' | 'italic' }[] {
  const faces: { weight: number; style: 'normal' | 'italic' }[] = pick.weights.map((weight) => ({ weight, style: 'normal' as const }))
  if (pick.italic) for (const weight of pick.weights) faces.push({ weight, style: 'italic' })
  return faces
}

/**
 * Downloads (once) every face the palette needs into `dir` and returns the catalog the font lister serves.
 * `unicodeRanges` is `backend/data/unicode_ranges.json`.
 */
export async function prepareFonts(dir: string, unicodeRangesFile: string, log: (line: string) => void): Promise<FontCatalogEntry[]> {
  await mkdir(dir, { recursive: true })
  const ranges = JSON.parse(await readFile(unicodeRangesFile, 'utf8')) as Record<string, string>
  const catalog: FontCatalogEntry[] = []
  let rank = 1
  for (const pick of Object.values(FONT_PICKS) as FontPick[]) {
    const faces = wantedFaces(pick)
    let base: string | null = null
    for (const face of faces) {
      const target = join(dir, fileName(pick, face.weight, face.style))
      if (existsSync(target)) continue
      base ??= `${CDN}/${pick.id}@${await version(pick)}/files`
      log(`  downloading ${fileName(pick, face.weight, face.style)}`)
      await writeFile(target, await download(`${base}/${fileName(pick, face.weight, face.style)}`))
    }
    catalog.push({
      id: pick.id,
      family: pick.family,
      category: pick.category,
      scripts: ['latin'],
      is_variable: false,
      is_user_font: false,
      popularity: rank++,
      unicode_ranges: ranges.latin ? { latin: ranges.latin } : {},
      variants: faces.map((face) => ({ ...face, url: `/__fonts/${fileName(pick, face.weight, face.style)}`, subsets: [] })),
    })
  }
  return catalog
}

/**
 * The fonts the starter templates use. Pure data, shared by the Node side (what to download) and the browser side
 * (which family a text gets). Every family is OFL-1.1 and among the 100 most popular of `backend/data/popular_fonts.json`, so
 * a stock `manage.py sync_fonts --popular 100` (what `make content` and the Docker first-run flags install) has it, and only
 * weights 400 and 700 are used: those are the weights `sync_fonts`
 * downloads by default (and a family that only has one weight is used at that weight, never faux-bolded).
 */
export type FontCategory = 'sans-serif' | 'serif' | 'display' | 'handwriting' | 'monospace'

export interface FontPick {
  /** Family name exactly as the catalog lists it. */
  family: string
  /** Fontsource id (the file name prefix of the woff2 files). */
  id: string
  category: FontCategory
  /** Weights this family really has among 400 and 700. */
  weights: readonly (400 | 700)[]
  /** Has italic faces that the templates use. */
  italic?: boolean
}

export const FONT_PICKS = {
  anton: { family: 'Anton', id: 'anton', category: 'sans-serif', weights: [400] },
  archivo: { family: 'Archivo', id: 'archivo', category: 'sans-serif', weights: [400, 700] },
  lora: { family: 'Lora', id: 'lora', category: 'serif', weights: [400, 700], italic: true },
  oswald: { family: 'Oswald', id: 'oswald', category: 'sans-serif', weights: [400, 700] },
  pinyon: { family: 'Pinyon Script', id: 'pinyon-script', category: 'handwriting', weights: [400] },
  bebas: { family: 'Bebas Neue', id: 'bebas-neue', category: 'sans-serif', weights: [400] },
  crimson: { family: 'Crimson Text', id: 'crimson-text', category: 'serif', weights: [400, 700], italic: true },
  dmSans: { family: 'DM Sans', id: 'dm-sans', category: 'sans-serif', weights: [400, 700] },
  greatVibes: { family: 'Great Vibes', id: 'great-vibes', category: 'handwriting', weights: [400] },
  inter: { family: 'Inter', id: 'inter', category: 'sans-serif', weights: [400, 700] },
  josefin: { family: 'Josefin Sans', id: 'josefin-sans', category: 'sans-serif', weights: [400, 700] },
  manrope: { family: 'Manrope', id: 'manrope', category: 'sans-serif', weights: [400, 700] },
  montserrat: { family: 'Montserrat', id: 'montserrat', category: 'sans-serif', weights: [400, 700] },
  playfair: { family: 'Playfair Display', id: 'playfair-display', category: 'serif', weights: [400, 700], italic: true },
  poppins: { family: 'Poppins', id: 'poppins', category: 'sans-serif', weights: [400, 700] },
} as const satisfies Record<string, FontPick>

export type FontKey = keyof typeof FONT_PICKS

const GENERIC: Record<FontCategory, string> = {
  'sans-serif': 'sans-serif',
  serif: 'serif',
  display: 'sans-serif',
  handwriting: 'cursive',
  monospace: 'monospace',
}

/** CSS font stack the way the editor writes it (`"Name", generic`). */
export function stackOf(key: FontKey): string {
  const pick: FontPick = FONT_PICKS[key]
  return `"${pick.family}", ${GENERIC[pick.category]}`
}

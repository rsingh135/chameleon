import addOnSandboxSdk from 'add-on-sdk-document-sandbox';
import { editor, constants, type ActivePageNode, type ArtboardNode, type GroupNode, type TextNode,
  type RectangleNode, type EllipseNode, type LineNode, type MediaContainerNode, type ImageRectangleNode,
  type Fill, type ColorFill, type SolidColorStroke, type StrokableNode } from 'express-document-sdk';
import { importPage, planWriteback, verifyNativeResult, requireDesign, cssColor, fingerprint, type NativePage, type NativeNode,
  type Matrix, type Snapshot, type Design, type AssetResource } from '../shared/adapter.js';
import { clonePageTransaction } from './transaction.js';

type SceneNode = NonNullable<ArtboardNode['children']['first']>;
type CaptureResult = {snapshot:Snapshot;assets:AssetResource[]};
const { runtime } = addOnSandboxSdk.instance;
let lastSnapshot: Snapshot|undefined;
let lastAssets: AssetResource[]=[];
let captureSequence=0;
const same=(a:number,b:number)=>Math.abs(a-b)<1e-4;
function solidFill(fill:Readonly<Fill>|undefined,id:string):string {
  if(!fill) return 'rgba(0,0,0,0)';
  requireDesign(fill.type===constants.FillType.color,'unsupported_fill','Gradient and other non-solid fills cannot be imported.',id);
  return cssColor((fill as ColorFill).color);
}
function matrix(node:SceneNode):Matrix {
  const m=node.transformMatrix;
  return [m[0]!,m[1]!,m[2]!,m[3]!,m[4]!,m[5]!];
}
function strokeOf(node:StrokableNode):NativeNode['stroke'] {
  if(!node.stroke) return undefined;
  requireDesign(node.stroke.type===constants.StrokeType.color,'unsupported_stroke','Only solid color strokes are supported.',node.id);
  const s=node.stroke as SolidColorStroke;
  requireDesign(s.dashPattern.length===0&&s.dashOffset===0,'unsupported_stroke','Dashed strokes cannot be represented.',node.id);
  return {color:cssColor(s.color),width:s.width};
}
function plain(node:SceneNode,images:Map<string,ImageRectangleNode>):NativeNode {
  const base={id:node.id,transform:matrix(node),bounds:{...node.boundsLocal},opacity:node.opacity};
  const t=constants.SceneNodeType;
  const isGroup=node.type===t.group;
  requireDesign(node.blendMode===(isGroup?constants.BlendMode.passThrough:constants.BlendMode.normal),
    'unsupported_blend','Non-default blend modes cannot be reproduced by the local renderer.',node.id);
  switch(node.type) {
    case t.group: {
      const group=node as GroupNode;
      requireDesign(!group.maskShape,'unsupported_mask','Masked native groups cannot be flattened into schema elements.',node.id);
      requireDesign(group.opacity===1,'group_compositing','Group opacity cannot be reproduced by multiplying child opacity.',node.id);
      return {...base,kind:'group',children:[...group.children].map(child=>plain(child,images))};
    }
    case t.text: {
      const text=node as TextNode;
      requireDesign(text.isStandaloneText(),'unsupported_text_layout','Threaded text is not supported; its shared content must remain linked.',node.id);
      requireDesign(text.visualEffects.length===0,'unsupported_text_effect','Text effects cannot be reproduced by the local renderer.',node.id);
      requireDesign(text.layout.type===constants.TextLayout.autoWidth||text.layout.type===constants.TextLayout.autoHeight,
        'unsupported_text_layout','Area, circular and magic-fit text cannot be imported faithfully.',node.id);
      const paragraphs=text.fullContent.paragraphStyleRanges, first=paragraphs[0];
      requireDesign(first,'unsupported_paragraph','Missing paragraph style.',node.id);
      requireDesign(paragraphs.every(p=>p.spaceBefore===0&&p.spaceAfter===0&&p.lineSpacing===first.lineSpacing&&!p.list),
        'unsupported_paragraph','Paragraph spacing, lists and mixed leading are not representable.',node.id);
      const align=text.textAlignment;
      requireDesign([constants.TextAlignment.left,constants.TextAlignment.center,constants.TextAlignment.right].includes(align),
        'unsupported_alignment','Justified or unknown text alignment is not supported.',node.id);
      return {...base,kind:'text',text:text.fullContent.text,
        layout:text.layout.type===constants.TextLayout.autoWidth?'autoWidth':'autoHeight',
        alignment:align===constants.TextAlignment.left?'left':align===constants.TextAlignment.center?'center':'right',lineSpacing:first.lineSpacing,
        runs:text.fullContent.characterStyleRanges.map(r=>({length:r.length,fontSize:r.fontSize,color:{...r.color},
          font:{postscriptName:r.font.postscriptName,family:r.font.family,style:r.font.style,availableForEditing:r.font.availableForEditing},
          letterSpacing:r.letterSpacing,underline:r.underline,baselineShift:r.baselineShift,...(r.link?{link:r.link}:{})}))};
    }
    case t.rectangle: {
      const rect=node as RectangleNode,radius=rect.getUniformCornerRadius();
      requireDesign(radius!==undefined,'unsupported_corners','Unequal corner radii cannot be represented.',node.id);
      requireDesign(!rect.stroke,'unsupported_shape_stroke','The current local renderer does not render rectangle strokes; import is refused.',node.id);
      return {...base,kind:'rectangle',fill:solidFill(rect.fill,node.id),radius};
    }
    case t.ellipse: {
      const ellipse=node as EllipseNode;
      requireDesign(!ellipse.stroke,'unsupported_shape_stroke','The current local renderer does not render ellipse strokes; import is refused.',node.id);
      return {...base,kind:'ellipse',fill:solidFill(ellipse.fill,node.id)};
    }
    case t.line: {
      const line=node as LineNode,stroke=strokeOf(line);
      requireDesign(stroke&&line.startArrowHeadType===constants.ArrowHeadType.none&&line.endArrowHeadType===constants.ArrowHeadType.none,
        'unsupported_line','Lines require a plain solid stroke without arrowheads.',node.id);
      // The engine renders a horizontal centerline. A native horizontal line may
      // still be rotated by its own or an ancestor's transform.
      requireDesign(same(line.startY,line.endY),'unsupported_line','Diagonal local line endpoints need a richer line schema; import is refused.',node.id);
      return {...base,kind:'line',fill:stroke.color,stroke};
    }
    case t.mediaContainer: {
      const container=node as MediaContainerNode,media=container.mediaRectangle,mask=container.maskShape;
      requireDesign(media.type===t.imageRectangle,'unsupported_media','Video, unknown and linked media are not supported.',node.id);
      requireDesign(mask.type===t.rectangle&&(mask as RectangleNode).getUniformCornerRadius()===0&&!mask.stroke,
        'unsupported_image_mask','Only an uncropped rectangular image mask is supported.',node.id);
      const image=media as ImageRectangleNode;
      requireDesign(typeof image.canExportMedia==='function'&&typeof image.fetchBitmapImage==='function','experimental_image_unavailable',
        'Experimental image export is unavailable. Enable experimentalApis in this development add-on; no screenshot fallback is used.',node.id);
      requireDesign(image.canExportMedia(),'image_export_denied','Adobe does not permit this image resource to be exported.',node.id);
      const ib=image.boundsInParent,mb=mask.boundsInParent;
      requireDesign(same(image.rotation,0)&&same(mask.rotation,0)&&same(ib.x,mb.x)&&same(ib.y,mb.y)&&same(ib.width,mb.width)&&same(ib.height,mb.height),
        'unsupported_image_crop','Cropped, rotated-in-mask or clipped images cannot be represented faithfully.',node.id);
      requireDesign(image.opacity===1&&mask.opacity===1,'unsupported_image_compositing','Image-internal opacity cannot be reproduced.',node.id);
      images.set(node.id,image);
      return {...base,kind:'image',assetId:`adobe-image-${Array.from(node.id).map(c=>c.codePointAt(0)!.toString(16).padStart(6,'0')).join('')}.png`};
    }
    default: throw new Error(`${node.id}: Unsupported Adobe node type ${node.type}; no element was dropped.`);
  }
}
function read(page:ActivePageNode,images:Map<string,ImageRectangleNode>):NativePage {
  requireDesign(page.artboards.length===1,'unsupported_animation','Only a single static artboard is supported.');
  const artboard=page.artboards.first;
  requireDesign(artboard,'missing_artboard','The current page has no active artboard.');
  return {pageId:page.id,artboardId:artboard.id,width:page.width,height:page.height,background:solidFill(artboard.fill,artboard.id),
    nodes:[...artboard.children].map(node=>plain(node,images))};
}
export function base64(bytes:Uint8Array):string {
  const alphabet='ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/'; let result='';
  for(let i=0;i<bytes.length;i+=3) {
    const a=bytes[i]!,b=bytes[i+1]??0,c=bytes[i+2]??0,n=(a<<16)|(b<<8)|c;
    result+=alphabet[(n>>>18)&63]!+alphabet[(n>>>12)&63]!+(i+1<bytes.length?alphabet[(n>>>6)&63]!:'=')+(i+2<bytes.length?alphabet[n&63]!:'=');
  }
  return result;
}
async function exportAssets(images:Map<string,ImageRectangleNode>,snapshot:Snapshot):Promise<AssetResource[]> {
  const assets:AssetResource[]=[];
  for(const [id,image] of images) {
      const bitmap=await image.fetchBitmapImage();
      requireDesign(typeof bitmap.data==='function','experimental_image_unavailable','Bitmap export is unavailable.',id);
      const blob=await bitmap.data();
      requireDesign(blob.type==='image/png'||blob.type==='image/jpeg','unsupported_image_encoding','Only PNG and JPEG image resources are supported.',id);
      const element=snapshot.source.elements.find(e=>e.id===id)!;
    assets.push({asset_id:String(element.asset_id),mime_type:blob.type,data_base64:base64(new Uint8Array(await blob.arrayBuffer()))});
  }
  return assets;
}
async function capture():Promise<CaptureResult> {
  const sequence=++captureSequence;
  lastSnapshot=undefined;lastAssets=[];
  const currentPage=editor.context.currentPage;
  requireDesign(currentPage,'missing_page','Open an Adobe Express design first.');
  let captured:CaptureResult|undefined;
  await editor.documentRoot.pages.visitPages([currentPage],async page=>{
    const images=new Map<string,ImageRectangleNode>(),snapshot=importPage(read(page,images));
    const assets=await exportAssets(images,snapshot);
    const after=importPage(read(page,new Map()));
    requireDesign(JSON.stringify(after.page)===JSON.stringify(snapshot.page),'stale_document','The document changed while resources were being read. Capture again.');
    snapshot.snapshot_id=`${snapshot.snapshot_id}-${fingerprint(assets)}`;
    captured={snapshot,assets};
  });
  requireDesign(captured&&sequence===captureSequence,'stale_capture','A newer capture replaced this one.');
  lastSnapshot=captured.snapshot;lastAssets=captured.assets;
  return captured;
}
async function createEditableCopy(snapshotId:string,result:Design):Promise<{page_id:string}> {
  requireDesign(lastSnapshot&&lastSnapshot.snapshot_id===snapshotId,'stale_snapshot','Capture the current design again before applying.');
  const snapshot=lastSnapshot, currentPage=editor.context.currentPage;
  requireDesign(currentPage&&currentPage.id===snapshot.page.pageId,'stale_document','Return to the captured source page or capture again.');
  const expectedAssets=lastAssets,sequence=captureSequence;
  // Validate the request before any resource loading or native mutation.
  planWriteback(snapshot,result);
  let created:{page_id:string}|undefined;
  await editor.documentRoot.pages.visitPages([currentPage],async activePage=>{
    const images=new Map<string,ImageRectangleNode>(),current=importPage(read(activePage,images));
    requireDesign(JSON.stringify(current.page)===JSON.stringify(snapshot.page),'stale_document','The source design changed. Capture it again before applying.');
    const assets=await exportAssets(images,current);
    requireDesign(JSON.stringify(assets)===JSON.stringify(expectedAssets),'stale_image_resource','An image resource changed since capture. Read the page again.');
    requireDesign(sequence===captureSequence&&lastSnapshot===snapshot,'stale_snapshot','A newer capture replaced this layout.');
    requireDesign(JSON.stringify(read(activePage,new Map()))===JSON.stringify(snapshot.page),'stale_document','The source changed while checking image resources.');
    created=writeCopy(activePage,snapshot,result);
  });
  requireDesign(created,'missing_output','Adobe did not create an editable page.');
  return created;
}
function writeCopy(currentPage:ActivePageNode,snapshot:Snapshot,result:Design):{page_id:string} {
  // Fully validate before creating any native object.
  const plan=planWriteback(snapshot,result),artboard=currentPage.artboards.first;
  requireDesign(artboard,'missing_artboard','The source artboard is no longer available.');
  const newPage=clonePageTransaction(editor.documentRoot.pages,artboard,plan,constants.ResizeBehavior.proportional,newPage=>{
    // Re-read every native leaf, retaining the mapping across native cloned IDs.
    const actual=read(newPage,new Map());
    const rebind=(native:NativeNode,original:NativeNode):void=>{
      requireDesign(native.kind===original.kind&&(native.children?.length??0)===(original.children?.length??0),'native_membership_mismatch','Adobe changed the cloned group structure.',original.id);
      native.id=original.id;if(original.assetId)native.assetId=original.assetId;
      native.children?.forEach((child,index)=>rebind(child,original.children![index]!));
    };
    requireDesign(actual.nodes.length===snapshot.page.nodes.length,'native_membership_mismatch','Adobe changed the root membership.');
    actual.nodes.forEach((node,index)=>rebind(node,snapshot.page.nodes[index]!));
    actual.artboardId=snapshot.page.artboardId;
    verifyNativeResult(result,importPage(actual,snapshot.font_pixels_per_point).source);
  });
  lastSnapshot=undefined;lastAssets=[];
  return {page_id:newPage.id};
}
runtime.exposeApi({capture,createEditableCopy});

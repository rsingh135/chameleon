import type { ActivePageNode, ArtboardNode, PageList, ResizeBehavior } from 'express-document-sdk';
import { requireDesign, UnsupportedDesignError, type WritePlan } from '../shared/adapter.js';

type SceneNode=NonNullable<ArtboardNode['children']['first']>;
/** Synchronous native transaction boundary. All model access stays within the
 * active-page callback. A validation error removes only objects created here. */
export function clonePageTransaction(pages:PageList,source:ArtboardNode,plan:WritePlan,
  behavior:ResizeBehavior.proportional,verify:(page:ActivePageNode)=>void):ActivePageNode {
  const originals=new Map([...source.children].map(node=>[node.id,node]));
  const roots=plan.transforms.map(transform=>{
    const original=originals.get(transform.rootId);
    requireDesign(original,'missing_native_root','An original root is no longer available.',transform.rootId);
    return {original,transform,center:original.localPointInNode(original.centerPointLocal,source)};
  });
  const clones:SceneNode[]=[];let page:ActivePageNode|undefined;
  try {
    for(const {original} of roots) clones.push(original.cloneInPlace());
    page=pages.addPage(plan.canvas);
    const destination=page.artboards.first;
    requireDesign(destination,'missing_artboard','Adobe did not create a destination artboard.');
    destination.fill=source.fill;
    for(let index=0;index<clones.length;index++) {
      const clone=clones[index]!,{transform,center}=roots[index]!;
      destination.children.append(clone);
      clone.resize({behavior,width:clone.boundsLocal.width*transform.scale,avoidScalingVisualDetailsIfPossible:false});
      clone.setPositionInParent({x:center.x*transform.scale+transform.translation.x,y:center.y*transform.scale+transform.translation.y},clone.centerPointLocal);
    }
    verify(page);
    return page;
  } catch(error) {
    const cleanupErrors:string[]=[];
    for(const clone of clones)try{clone.removeFromParent();}catch(cleanup){cleanupErrors.push(String(cleanup));}
    if(page)try{pages.remove(page);}catch(cleanup){cleanupErrors.push(String(cleanup));}
    if(cleanupErrors.length)throw new UnsupportedDesignError('native_rollback_failed',`Adobe could not fully remove the failed copy: ${cleanupErrors.join('; ')}. Original objects were not resized or restyled. Original error: ${String(error)}`);
    throw error;
  }
}

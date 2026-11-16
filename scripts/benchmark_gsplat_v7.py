#!/usr/bin/env python3
"""Matched gsplat + legacy device-v7 MLP: quality, rendering and Graph E2E."""
import copy
import ctypes
import json
import os
from pathlib import Path
import statistics
import sys
from time import perf_counter

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from analyze_gsplat_mixed import load_workload,save,sha,stats


def seal_extra(args,report):
    files=[Path(__file__),ROOT/'gsplat_mixed_ext/mixed_tensor_v7.cu',
           ROOT/'gsplat_mixed_ext/bindings_tensor_v7.cpp',ROOT/'tests/test_gsplat_tensor_v7_cuda.py',
           ROOT/'scripts/run_jetson_gsplat_v7.sh',ROOT/'tacker_ext/tacker_4dgs_head/__init__.py',
           ROOT/'arguments/__init__.py',ROOT/'render.py',
           Path(os.environ['GSPLAT_LEGACY_V7_LIBRARY']),Path(args.configs)]
    for source in files:
        report['source_and_checkpoint_sha256'][str(source.resolve())]=sha(source)
        try: relative=source.resolve().relative_to(ROOT)
        except ValueError: continue
        destination=args.output/'runtime_sources'/relative
        destination.parent.mkdir(parents=True,exist_ok=True)
        destination.write_bytes(source.read_bytes())
    report.update(kind='gsplat_legacy_device_v7_matched',schema_version=1,
                  mlp_implementation='unchanged deform_mlp_tensor_tail_gptb_device_v7',
                  graph_configuration={'profile':'full_mlp_overlap_graph_pb0','slots':2,
                      'lowreg_sampling':False,'fused_reduction':False,
                      'prefix':'same native HexPlane CUDA Graph as prior gsplat experiment'},
                  trial_schedule='forward/reverse candidate order, alternating per round')
    report['environment']['tacker_environment']={k:v for k,v in os.environ.items() if k.startswith('TACKER_')}


def make_renderer(args,pc,scene,pipe,background,fused,blocks=0):
    from gaussian_renderer.gsplat_mixed import GsplatMixedRenderer
    renderer=GsplatMixedRenderer(pc,copy.copy(pipe),background,args.qualification_profile,
        fused=fused,physical_blocks=blocks,kernel_variant='tensor_v7',cam_type=scene.dataset_type)
    renderer.prepare()
    return renderer


def v7_state(context,pc,renderer):
    import torch
    from gaussian_renderer.tacker_pipeline import prepare_full_mlp_task,finish_full_mlp_task
    task=prepare_full_mlp_task(context,pc,torch.cuda.current_stream(),renderer.variant,renderer.cached)
    delta=renderer.extension.mlp_forward([task.deform_features,*renderer.leaf_cached])
    return finish_full_mlp_task(task,pc,delta)


def node_resources(driver,graph):
    from sampling_prepass_priority import KernelParamsV1
    getter=driver.library.cuFuncGetAttribute
    getter.argtypes=[ctypes.POINTER(ctypes.c_int),ctypes.c_int,ctypes.c_void_p]
    occupancy=driver.library.cuOccupancyMaxActiveBlocksPerMultiprocessor
    occupancy.argtypes=[ctypes.POINTER(ctypes.c_int),ctypes.c_void_p,ctypes.c_int,ctypes.c_size_t]
    resources=[]
    for handle in driver.kernel_handles(graph.raw_cuda_graph()):
        params=KernelParamsV1()
        driver.call('cuGraphKernelNodeGetParams',ctypes.c_void_p(handle),ctypes.byref(params))
        row={}
        for label,code in (('registers',4),('static_shared_bytes',1),('local_bytes',3),('binary_arch',6)):
            value=ctypes.c_int()
            assert getter(ctypes.byref(value),code,params.func)==0
            row[label]=value.value
        active=ctypes.c_int()
        threads=params.block_x*params.block_y*params.block_z
        assert occupancy(ctypes.byref(active),params.func,threads,params.shared_bytes)==0
        row.update(threads=threads,dynamic_shared_bytes=params.shared_bytes,active_ctas_per_sm=active.value)
        resources.append(row)
    return resources


def leaf_sweep(args,pc,scene,pipe,background,views,report):
    import torch
    from gaussian_renderer import prepare_render_context,deform_for_render
    from gaussian_renderer.gsplat_mixed import prepare_gsplat
    from gsplat.cuda._wrapper import rasterize_to_pixels
    from tests.test_gsplat_tensor_v7_cuda import LegacyV7
    from sampling_prepass_priority import CudaDriver
    renderer=make_renderer(args,pc,scene,pipe,background,False)
    extension=renderer.extension
    legacy=LegacyV7()
    report['resources']={'mixed':extension.resources(),'standalone':extension.standalone_resources()}
    report['leaf_cases']=[]
    candidate_labels=None
    for view_index in (0,24,48):
        context=prepare_render_context(views[view_index],pc,pipe,background,cam_type=scene.dataset_type)
        state=deform_for_render(context,pc) if view_index==0 else v7_state(context,pc,renderer)
        prepared=prepare_gsplat(context,state)
        next_context=prepare_render_context(views[view_index+1],pc,pipe,background,cam_type=scene.dataset_type)
        grid=pc._deformation.deformation_net.grid
        query=getattr(grid,'forward_cuda_graph',grid)
        features=query(next_context.means3D,next_context.timestamp).contiguous()
        mlp=[features,*renderer.leaf_cached]
        r,w,h=prepared.inputs,prepared.width,prepared.height
        raster_tiles=((w+15)//16)*((h+15)//16)
        deform_tiles=(features.shape[0]+15)//16
        def pixels(): return rasterize_to_pixels(*r[:4],w,h,16,r[4],r[5],backgrounds=r[6])
        def independent(): return (*pixels(),*extension.mlp_forward(mlp))
        reference=(*pixels(),*legacy(mlp))
        arms={'raster_native':pixels,'mlp_legacy':lambda:legacy(mlp),
              'mlp_migrated':lambda:extension.mlp_forward(mlp),'independent':independent}
        modes={}
        for blocks in dict.fromkeys((0,raster_tiles,(deform_tiles+1)//2,raster_tiles//2)):
            modes['mixed_b'+str(blocks)]=(blocks,2)
        modes['raster_mixed_envelope']=(raster_tiles,0)
        modes['mlp_mixed_envelope']=(0,1)
        candidate_labels=[label for label,(_,mode) in modes.items() if mode==2]
        for label,(blocks,mode) in modes.items():
            arms[label]=lambda blocks=blocks,mode=mode:extension.forward(r,mlp,w,h,blocks,mode)
        stream,driver=torch.cuda.Stream(),CudaDriver()
        stream.wait_stream(torch.cuda.current_stream())
        graphs,outputs={},{}
        case={'raster_view':view_index,'deform_view':view_index+1,'timings':{},
              'logical_tiles':{'raster':raster_tiles,'deform':deform_tiles},
              'feature_sha256':__import__('hashlib').sha256(features.cpu().numpy().tobytes()).hexdigest()}
        for label,callback in arms.items():
            with torch.cuda.stream(stream):
                for _ in range(2): callback()
            stream.synchronize()
            graph=torch.cuda.CUDAGraph(keep_graph=True)
            with torch.cuda.graph(graph,stream=stream): outputs[label]=callback()
            graphs[label]=graph
            with torch.cuda.stream(stream): graph.replay()
            stream.synchronize()
            actual=outputs[label]
            if label in modes:
                mode=modes[label][1]
                pairs=[]
                if mode!=1: pairs+=list(zip(actual[:2],reference[:2]))
                if mode!=0: pairs+=list(zip(actual[3:8],reference[2:]))
            elif label=='raster_native': pairs=list(zip(actual,reference[:2]))
            elif label.startswith('mlp_'): pairs=list(zip(actual,reference[2:]))
            else: pairs=list(zip(actual,reference))
            assert all(torch.equal(a,b) for a,b in pairs),('legacy/native mismatch',view_index,label)
            nodes=driver.kernel_nodes(graph.raw_cuda_graph())
            assert len(nodes)==(2 if label=='independent' else 1),nodes
            case['timings'][label]={'rounds':[],'bitwise_equal':True,'nodes':nodes,
                                    'resources':node_resources(driver,graph)}
        for trial in range(6):
            order=list(arms) if trial%2==0 else list(reversed(arms))
            for label in order:
                graph=graphs[label]
                start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                with torch.cuda.stream(stream):
                    for _ in range(5): graph.replay()
                    start.record(stream)
                    for _ in range(20): graph.replay()
                    end.record(stream)
                end.synchronize()
                elapsed=start.elapsed_time(end)/20
                case['timings'][label]['rounds'].append(elapsed)
                print('LEAF',view_index,trial,label,elapsed,flush=True)
        for value in case['timings'].values(): value['timing']=stats(value['rounds'])
        raw=extension.forward(r,mlp,w,h,raster_tiles,2,True)
        windows=raw[8].cpu().tolist()
        fractions=[max(0,min(re,de)-max(rs,ds))/min(re-rs,de-ds)
                   for rs,re,ds,de in windows if rs and ds and re>rs and de>ds]
        assert fractions and any(x>0 for x in fractions)
        case['overlap_probe']={'paired_ctas':len(fractions),'fraction_with_overlap':sum(x>0 for x in fractions)/len(fractions),
                               'median_overlap_over_shorter':statistics.median(fractions),'blocks':raster_tiles}
        report['leaf_cases'].append(case)
        save(args.output/'partial.json',report)
        del graphs,outputs,arms,reference,raw,state,prepared,context,next_context,features,mlp,r
        torch.cuda.synchronize()
    report['leaf_average_ms']={label:statistics.mean(case['timings'][label]['timing']['median_ms']
                               for case in report['leaf_cases']) for label in report['leaf_cases'][0]['timings']}
    best=min(candidate_labels,key=lambda label:report['leaf_average_ms'][label])
    report['selected']={'variant':'tensor_v7','label':best,'blocks':int(best.split('_b')[1]),
                        'criterion':'minimum mean of three frozen-view leaf medians; formal50 independent validation'}
    context=prepare_render_context(views[24],pc,pipe,background,cam_type=scene.dataset_type)
    prepared=prepare_gsplat(context,v7_state(context,pc,renderer))
    next_context=prepare_render_context(views[25],pc,pipe,background,cam_type=scene.dataset_type)
    query=getattr(pc._deformation.deformation_net.grid,'forward_cuda_graph',pc._deformation.deformation_net.grid)
    mlp=[query(next_context.means3D,next_context.timestamp).contiguous(),*renderer.leaf_cached]
    raw=extension.forward(prepared.inputs,mlp,prepared.width,prepared.height,report['selected']['blocks'],2,True)
    fractions=[max(0,min(re,de)-max(rs,ds))/min(re-rs,de-ds)
               for rs,re,ds,de in raw[8].cpu().tolist() if rs and ds and re>rs and de>ds]
    assert fractions and any(x>0 for x in fractions)
    report['selected_overlap_probe']={'blocks':report['selected']['blocks'],'paired_ctas':len(fractions),
        'fraction_with_overlap':sum(x>0 for x in fractions)/len(fractions),
        'median_overlap_over_shorter':statistics.median(fractions),'raster_view':24,'deform_view':25}
    report['timing_scope']='leaf Graph replay; 3 real frame pairs; 6 alternating rounds, 5 warmup +20 replays per arm'


def measure(callback,count):
    import torch
    torch.cuda.synchronize()
    start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
    start.record(); before=perf_counter()
    callback()
    end.record(); torch.cuda.synchronize()
    wall=(perf_counter()-before)*1000/count
    return {'wall_ms':wall,'cuda_span_ms':start.elapsed_time(end)/count,'fps':1000/wall}


def summarize_trials(values):
    return {key:statistics.median(row[key] for row in values) for key in ('wall_ms','cuda_span_ms','fps')}


def formal(args,pc,scene,pipe,background,views,report):
    import torch
    from gaussian_renderer import prepare_render_context,render
    from gaussian_renderer.gsplat_mixed import prepare_gsplat,render_prepared
    selected=json.loads((args.output.parent/'sweep/report.json').read_text())['selected']
    report['selection']=selected
    renderers={label:make_renderer(args,pc,scene,pipe,background,fused,selected['blocks'])
               for label,fused in (('independent',False),('mixed',True))}
    for a,b in zip(renderers['independent'].leaf_cached,renderers['mixed'].leaf_cached):
        assert torch.equal(a,b)
    report['packed_tail_sha256']=__import__('hashlib').sha256(renderers['independent'].leaf_cached[4].cpu().numpy().tobytes()).hexdigest()
    def consume(renderer,selected_views):
        for _ in renderer.render_sequence(selected_views): pass
    for renderer in renderers.values():
        consume(renderer,views[:3]); consume(renderer,views[:10])
    report['end_to_end']={label:{'trials':[]} for label in renderers}
    for trial in range(10):
        order=list(renderers) if trial%2==0 else list(reversed(renderers))
        for label in order:
            renderer=renderers[label]
            row=measure(lambda:consume(renderer,views),len(views))
            counts=renderer.last_execution_counts
            assert counts['prefix_graph_replays']==49 and counts['outputs']==50,counts
            assert counts['mixed_launches']==(49 if renderer.fused else 0),counts
            assert counts['separate_leaf_pairs']==(0 if renderer.fused else 49),counts
            report['end_to_end'][label]['trials'].append(row)
            report['end_to_end'][label]['execution_counts']=dict(counts)
            print('E2E',trial,label,row,flush=True)
        save(args.output/'partial.json',report)
    for value in report['end_to_end'].values(): value['median']=summarize_trials(value['trials'])

    # Shared activated device-v7 states, generated outside timing. Mode 0 is
    # a Raster-only resource-envelope control, not the real mixed leaf cost.
    base=renderers['independent']
    contexts=[prepare_render_context(v,pc,pipe,background,cam_type=scene.dataset_type) for v in views]
    states=[v7_state(c,pc,base) for c in contexts]
    dummy=pc.get_xyz.new_empty((pc.get_xyz.shape[0],32))
    mlp=[dummy,*base.leaf_cached]
    for context,state in zip(contexts,states):
        prepared=prepare_gsplat(context,state)
        native_result=render_prepared(prepared)
        raw=base.extension.forward(prepared.inputs,mlp,prepared.width,prepared.height,selected['blocks'],0)
        mixed_result=prepared.result(raw[0])
        assert torch.equal(native_result.render,mixed_result.render)
        assert torch.equal(native_result.depth,mixed_result.depth)
    report['raster_only_outputs_bitwise_equal_views']=50
    def raster_sequence(label):
        for context,state in zip(contexts,states):
            prepared=prepare_gsplat(context,state)
            if label=='native': render_prepared(prepared)
            else:
                raw=base.extension.forward(prepared.inputs,mlp,prepared.width,prepared.height,selected['blocks'],0)
                prepared.result(raw[0])
    report['raster_only']={label:{'trials':[]} for label in ('native','mixed_envelope')}
    for label in report['raster_only']: raster_sequence(label)
    for trial in range(10):
        order=list(report['raster_only']) if trial%2==0 else list(reversed(report['raster_only']))
        for label in order:
            row=measure(lambda:raster_sequence(label),len(views))
            report['raster_only'][label]['trials'].append(row)
            print('RASTER',trial,label,row,flush=True)
    for value in report['raster_only'].values(): value['median']=summarize_trials(value['trials'])
    report['raster_only_scope']='cached shared device-v7 activated states, gsplat prepare+RGBD pixels+result layout; excludes context/deformation; mixed_envelope disables MLP and is diagnostic only'
    del contexts,states,dummy,mlp
    torch.cuda.synchronize(); torch.cuda.empty_cache()
    save(args.output/'partial.json',report)

    from lpipsPyTorch.modules.lpips import LPIPS
    from utils.image_utils import psnr
    from utils.loss_utils import ssim
    metric=LPIPS('vgg','0.1').cuda().eval()
    reference=[]; independent=[]
    report['quality']={}
    class ReplayAudit:
        def __init__(self,owner): self.owner,self.calls=owner,0
        def replay(self): self.calls+=1; return self.owner.replay()
    for label in ('gsplat_native','independent','mixed'):
        rows=[]; renderer=renderers.get(label)
        audits=[]
        if renderer:
            audits=[ReplayAudit(slot.graph) for slot in renderer.graphs]
            for slot,audit in zip(renderer.graphs,audits): slot.graph=audit
            outputs=renderer.render_sequence(views)
        else: outputs=(render(v,pc,pipe,background,cam_type=scene.dataset_type) for v in views)
        try:
            for i,result in enumerate(outputs):
                raw,depth=result['render'],result['depth']
                assert torch.isfinite(raw).all() and torch.isfinite(depth).all()
                cpu_raw,cpu_depth=raw.cpu(),depth.cpu()
                if label=='gsplat_native': reference.append((cpu_raw,cpu_depth))
                image=raw.clamp(0,1); gt=views[i].original_image[:3].cuda()
                row={'view':i,'psnr_db':float(psnr(image[None],gt[None]).mean()),
                     'ssim':float(ssim(image[None],gt[None])),
                     'lpips':float(metric(image[None],gt[None]).mean())}
                if label!='gsplat_native':
                    a,b=reference[i]
                    row.update(rgb_max_abs_vs_native=float((cpu_raw-a).abs().max()),
                               rgb_mae_vs_native=float((cpu_raw-a).abs().mean()),
                               depth_max_abs_vs_native=float((cpu_depth-b).abs().max()))
                if label=='independent': independent.append((cpu_raw,cpu_depth))
                elif label=='mixed':
                    a,b=independent[i]
                    row.update(rgb_bitwise_equal=torch.equal(cpu_raw,a),depth_bitwise_equal=torch.equal(cpu_depth,b),
                               rgb_max_abs_vs_independent=float((cpu_raw-a).abs().max()),
                               depth_max_abs_vs_independent=float((cpu_depth-b).abs().max()))
                    assert row['rgb_bitwise_equal'] and row['depth_bitwise_equal'],row
                rows.append(row)
        finally:
            if renderer:
                for slot,audit in zip(renderer.graphs,audits): slot.graph=audit.owner
        report['quality'][label]={'per_view':rows,'mean':{key:statistics.mean(r[key] for r in rows) for key in ('psnr_db','ssim','lpips')}}
        if renderer:
            assert sum(a.calls for a in audits)==49
            report['quality'][label]['actual_graph_replay_calls_per_slot']=[a.calls for a in audits]
        save(args.output/'partial.json',report)
        print('QUALITY',label,report['quality'][label]['mean'],flush=True)
    native=report['quality']['gsplat_native']['mean']
    for label in ('independent','mixed'):
        value=report['quality'][label]; means=value['mean']
        delta={'psnr_drop_db':native['psnr_db']-means['psnr_db'],'ssim_drop':native['ssim']-means['ssim'],
               'lpips_increase':means['lpips']-native['lpips']}
        limits={'psnr_drop_db':0.05,'ssim_drop':1e-4,'lpips_increase':1e-4}
        value.update(delta_vs_gsplat_native=delta,thresholds=limits,quality_gate_passed=all(delta[k]<=v for k,v in limits.items()))
    report['resources']={label:{'mixed':renderer.extension.resources(),
                        'standalone':renderer.extension.standalone_resources(),
                        'mlp_implementation':renderer.mlp_implementation,'blocks':renderer.physical_blocks,
                        'graph_slots':len(renderer.graphs)} for label,renderer in renderers.items()}
    report['timing_scope']='normal Python E2E, same 50 views, 10 alternating rounds, includes prefill/drain/context/prefix/MLP/residual/prepare/RGBD/sync; excludes loading/JIT/warmup/metrics/I/O'


def main():
    args,pc,scene,pipe,background,views,report=load_workload()
    seal_extra(args,report)
    (leaf_sweep if args.stage=='sweep' else formal)(args,pc,scene,pipe,background,views,report)
    report['inputs_unchanged']=all(sha(p)==digest for p,digest in report['source_and_checkpoint_sha256'].items())
    assert report['inputs_unchanged']
    binaries={str(Path(m.__file__).resolve()) for m in list(sys.modules.values())
              if (getattr(m,'__file__','') or '').endswith('.so')
              and any(s in m.__file__ for s in ('gsplat','tacker_4dgs','diff_gaussian'))}
    report['runtime_binary_sha256']={p:sha(p) for p in sorted(binaries)}
    report['completed']=True
    save(args.output/'report.json',report)
    print('DONE',args.output/'report.json',flush=True)


if __name__=='__main__': main()

#!/usr/bin/env python3
"""Resource controls, CTA sweep and matched Graph E2E for gsplat mixed.

Leaf timing replays captured graphs (allocation/host dispatch excluded). The
formal stage measures normal Python inference, including host dispatch, prefix,
prepare, residual activation and synchronization, with 50 consecutive views.
"""
import argparse
import copy
import ctypes
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def stats(values):
    return {'median_ms': statistics.median(values), 'mean_ms': statistics.mean(values),
            'min_ms': min(values), 'max_ms': max(values), 'samples': values}


def load_workload():
    from arguments import ModelParams, ModelHiddenParams, PipelineParams, get_combined_args
    from utils.params_utils import load_config, merge_hparams
    from utils.general_utils import safe_state
    from scene import Scene
    from gaussian_renderer import GaussianModel
    import torch
    parser = argparse.ArgumentParser(description=__doc__)
    model, hidden, pipeline = ModelParams(parser, sentinel=True), ModelHiddenParams(parser), PipelineParams(parser)
    parser.add_argument('--configs', required=True)
    parser.add_argument('--iteration', type=int, default=14000)
    parser.add_argument('--qualification-profile', type=Path, required=True)
    parser.add_argument('--stage', choices=('sweep', 'sweep_grid', 'formal'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = merge_hparams(get_combined_args(parser), load_config(parser.parse_args().configs))
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / 'report.json').exists():
        raise RuntimeError('refusing to overwrite completed report')
    safe_state(False)
    torch.set_grad_enabled(False)
    dataset = model.extract(args)
    pc = GaussianModel(dataset.sh_degree, hidden.extract(args))
    scene = Scene(dataset, pc, load_iteration=args.iteration, shuffle=False)
    assert scene.loaded_iter == args.iteration
    pc._deformation.eval()
    pipe = pipeline.extract(args)
    pipe.raster_backend, pipe.gsplat_packed = 'gsplat', False
    background = torch.tensor([1,1,1] if dataset.white_background else [0,0,0], device='cuda', dtype=torch.float32)
    views = [scene.getTestCameras()[i] for i in range(50)]
    checkpoint = Path(dataset.model_path) / 'point_cloud' / ('iteration_%d' % args.iteration)
    sources = [ROOT/'gsplat_mixed_ext'/f for f in ('mixed.cu','bindings.cpp','__init__.py')]
    if (ROOT/'gsplat_mixed_ext/mixed_grid_constant.cu').exists(): sources.append(ROOT/'gsplat_mixed_ext/mixed_grid_constant.cu')
    sources += [ROOT/'gaussian_renderer'/f for f in ('__init__.py','gsplat_mixed.py','gsplat_backend.py','tacker_pipeline.py')]
    sources += [ROOT/'scene/hexplane.py', ROOT/'tacker_ext/include/deform_mlp_device.cuh', Path(__file__), args.qualification_profile]
    sources += [p for p in checkpoint.iterdir() if p.is_file()]
    props=torch.cuda.get_device_properties(0)
    report = {'stage':args.stage, 'environment':{'torch':torch.__version__, 'cuda':torch.version.cuda,
              'gpu':torch.cuda.get_device_name(), 'python':sys.version,
              'sm_count':props.multi_processor_count,'total_memory_bytes':props.total_memory,
              'compute_capability':list(torch.cuda.get_device_capability()),
              'cache_environment':{k:v for k,v in os.environ.items() if k.endswith(('CACHE_HOME','CACHE_PATH','EXTENSIONS_DIR')) or k=='TMPDIR'}},
              'workload':{'iteration':args.iteration, 'model':dataset.model_path, 'views':list(range(50)),
                          'gaussians':pc.get_xyz.shape[0], 'resolution':[views[0].image_width,views[0].image_height]},
              'parent_qualification':str(args.qualification_profile),
              'parent_qualification_sha256':sha(args.qualification_profile),
              'source_and_checkpoint_sha256':{str(p.resolve()):sha(p) for p in sources}}
    for source in sources:
        try: relative=source.resolve().relative_to(ROOT)
        except ValueError: continue
        destination=args.output/'runtime_sources'/relative
        destination.parent.mkdir(parents=True,exist_ok=True)
        destination.write_bytes(source.read_bytes())
    (args.output/'parent.qualification.json').write_bytes(args.qualification_profile.read_bytes())
    return args, pc, scene, pipe, background, views, report


def sweep(args, pc, scene, pipe, background, views, report):
    import torch
    from gaussian_renderer import prepare_render_context, deform_for_render
    from gaussian_renderer.gsplat_mixed import prepare_gsplat
    from gaussian_renderer.tacker_pipeline import _cache_full_mlp_parameters
    from gsplat_mixed_ext import load
    from gsplat.cuda._wrapper import rasterize_to_pixels
    from tacker_4dgs_head import deform_mlp_delta_v1
    from sampling_prepass_priority import CudaDriver, KernelParamsV1
    modules = {v:load(v) for v in (('v1','v2','v3') if args.stage=='sweep_grid' else ('v1','v2'))}
    report['resources'] = {v:{'mixed':m.resources(),'standalone':m.standalone_resources()}
                            for v,m in modules.items()}
    context = prepare_render_context(views[0],pc,pipe,background,cam_type=scene.dataset_type)
    prepared = prepare_gsplat(context,deform_for_render(context,pc))
    # Raster(0), Deform(1), matching the real inter-frame pipeline.
    next_context = prepare_render_context(views[1],pc,pipe,background,cam_type=scene.dataset_type)
    features = pc._deformation.deformation_net.grid(next_context.means3D,next_context.timestamp).contiguous()
    mlp = [features,*_cache_full_mlp_parameters(pc)]
    r,w,h = prepared.inputs, prepared.width, prepared.height
    rt,dt = ((w+15)//16)*((h+15)//16),(features.shape[0]+15)//16
    report['logical_tiles'] = {'raster':rt,'deform':dt}
    def native_raster():
        return rasterize_to_pixels(*r[:4],w,h,16,r[4],r[5],backgrounds=r[6])
    def native_mlp():
        return deform_mlp_delta_v1(*mlp)
    expected = (*native_raster(),*native_mlp())
    def native_serial():
        return (*native_raster(),*native_mlp())
    auxiliary=torch.cuda.Stream()
    def native_concurrent():
        current=torch.cuda.current_stream()
        auxiliary.wait_stream(current)
        raster_result=native_raster()
        with torch.cuda.stream(auxiliary): mlp_result=native_mlp()
        current.wait_stream(auxiliary)
        return (*raster_result,*mlp_result)
    arms = {'native_raster':native_raster,'native_mlp':native_mlp,'native_serial':native_serial,
            'native_concurrent':native_concurrent}
    spec = {}
    for label,mode in (('own_serial',3),('own_raster',4),('own_mlp',5),('envelope_raster',0),('envelope_mlp',1)):
        spec[label] = ('v1',0,mode)
    spec['envelope_raster_gridmatched'] = ('v1',rt,0)
    spec['v2_envelope_raster_gridmatched'] = ('v2',rt,0)
    spec['v2_envelope_mlp'] = ('v2',0,1)
    for variant in modules:
        for blocks in dict.fromkeys((0,rt,(dt+1)//2,rt//2,1536,768,384,192)):
            spec[variant+'_b'+str(blocks)] = (variant,blocks,2)
    if args.stage=='sweep_grid':
        spec={k:v for k,v in spec.items() if (v[0]=='v3' and (v[2]!=2 or v[1] in (0,rt,rt//2,768)))
              or k in ('v1_b0','v2_b5440','own_mlp','envelope_mlp')}
        spec['v3_envelope_mlp']=('v3',0,1)
    for label,(v,b,mode) in spec.items():
        arms[label] = lambda v=v,b=b,mode=mode: modules[v].forward(r,mlp,w,h,b,mode)
    stream, driver = torch.cuda.Stream(), CudaDriver()
    stream.wait_stream(torch.cuda.current_stream())
    graphs, outputs = {}, {}
    report['leaf'] = {}
    for label, callback in arms.items():
        with torch.cuda.stream(stream):
            for _ in range(2): callback()
        stream.synchronize()
        graph = torch.cuda.CUDAGraph(keep_graph=True)
        with torch.cuda.graph(graph,stream=stream):
            outputs[label] = callback()
        graphs[label] = graph
        with torch.cuda.stream(stream): graph.replay()
        stream.synchronize()
        actual = outputs[label]
        if label not in ('native_raster','native_mlp','native_serial','native_concurrent'):
            mode = spec[label][2]
            pairs = []
            if mode not in (1,5): pairs += [(actual[0],expected[0]),(actual[1],expected[1])]
            if mode not in (0,4): pairs += list(zip(actual[3:8],expected[2:]))
        elif label == 'native_raster': pairs = list(zip(actual,expected[:2]))
        elif label == 'native_mlp': pairs = list(zip(actual,expected[2:]))
        else: pairs = list(zip(actual,expected))
        errors = [float((a-b).abs().max()) for a,b in pairs]
        for a,b in pairs: torch.testing.assert_close(a,b,atol=2e-6,rtol=1e-5)
        nodes = driver.kernel_nodes(graph.raw_cuda_graph())
        assert len(nodes) == (2 if label in ('native_serial','native_concurrent','own_serial') else 1), nodes
        resources = []
        getter = driver.library.cuFuncGetAttribute
        getter.argtypes = [ctypes.POINTER(ctypes.c_int),ctypes.c_int,ctypes.c_void_p]
        occupancy = driver.library.cuOccupancyMaxActiveBlocksPerMultiprocessor
        occupancy.argtypes = [ctypes.POINTER(ctypes.c_int),ctypes.c_void_p,ctypes.c_int,ctypes.c_size_t]
        for handle in driver.kernel_handles(graph.raw_cuda_graph()):
            params=KernelParamsV1()
            driver.call('cuGraphKernelNodeGetParams',ctypes.c_void_p(handle),ctypes.byref(params))
            row={}
            for name,code in (('registers',4),('static_shared_bytes',1),('local_bytes',3),('binary_arch',6)):
                value=ctypes.c_int()
                assert getter(ctypes.byref(value),code,params.func)==0
                row[name]=value.value
            active=ctypes.c_int()
            threads=params.block_x*params.block_y*params.block_z
            assert occupancy(ctypes.byref(active),params.func,threads,params.shared_bytes)==0
            row.update(threads=threads,dynamic_shared_bytes=params.shared_bytes,active_ctas_per_sm=active.value)
            resources.append(row)
        report['leaf'][label] = {'max_abs_errors':errors,'nodes':nodes,'rounds':[]}
        report['leaf'][label]['resources']=resources
    # Batch event timing: no Python submission gaps in these GPU-bound batches.
    for trial in range(6):
        labels = list(arms) if trial%2==0 else list(reversed(arms))
        for label in labels:
            graph = graphs[label]
            with torch.cuda.stream(stream):
                for _ in range(5): graph.replay()
                start,end = torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                start.record(stream)
                for _ in range(20): graph.replay()
                end.record(stream)
            end.synchronize()
            elapsed = start.elapsed_time(end)/20
            report['leaf'][label]['rounds'].append(elapsed)
            print('LEAF',trial,label,elapsed,flush=True)
        save(args.output/'partial.json',report)
    for value in report['leaf'].values(): value['timing'] = stats(value['rounds'])
    candidates = [label for label in spec if spec[label][2] == 2]
    best = min(candidates,key=lambda k:report['leaf'][k]['timing']['median_ms'])
    report['selected'] = {'label':best,'variant':spec[best][0],'blocks':spec[best][1],
                          'criterion':'lowest leaf median on view0; independently validated in formal50'}
    # Instrumented launches are diagnostic only; excluded from all timings.
    diagnostics = {}
    for variant,blocks in (('v1',0),(spec[best][0],spec[best][1])):
        key=variant+'_b'+str(blocks)
        if key in diagnostics: continue
        windows = {}
        for mode in (0,1,2):
            raw = modules[variant].forward(r,mlp,w,h,blocks,mode,True)
            windows[str(mode)] = raw[8].cpu().tolist()
        ratios = {}
        for name,startcol in (('raster',0),('deform',2)):
            solo = windows['0' if name=='raster' else '1']
            both = windows['2']
            samples = [(b[startcol+1]-b[startcol])/(a[startcol+1]-a[startcol])
                       for a,b in zip(solo,both) if a[startcol] and b[startcol]
                       and a[startcol+1]>a[startcol] and b[startcol+1]>b[startcol]]
            ratios[name] = {'median_both_over_solo_cycles':statistics.median(samples),
                            'mean_both_over_solo_cycles':statistics.mean(samples)}
        diagnostics[key] = {'window_cycles':windows,'subgroup_inflation_same_resource_envelope':ratios}
    save(args.output/'diagnostic_windows.json',diagnostics)
    report['diagnostic_summary'] = {k:v['subgroup_inflation_same_resource_envelope'] for k,v in diagnostics.items()}
    report['timing_scope'] = 'captured Graph leaf, inputs ready, 6 alternating rounds, 5 warmup +20 replays per arm'


def formal(args, pc, scene, pipe, background, views, report):
    import torch
    from gaussian_renderer.gsplat_mixed import GsplatMixedRenderer
    selection_path=args.output.parent/'sweep_grid/report.json'
    if not selection_path.exists(): selection_path=args.output.parent/'sweep/report.json'
    selected = json.loads(selection_path.read_text())['selected']
    report['sweep_selection'] = selected
    configs = {'independent':dict(fused=False,kernel_variant='v1'), 'mixed_v1':dict(fused=True,kernel_variant='v1'),
               'mixed_candidate':dict(fused=True,kernel_variant=selected['variant'],physical_blocks=selected['blocks'])}
    renderers = {label:GsplatMixedRenderer(pc,copy.copy(pipe),background,args.qualification_profile,
                   cam_type=scene.dataset_type,**config) for label,config in configs.items()}
    def consume(renderer,selected_views):
        for _ in renderer.render_sequence(selected_views): pass
    for renderer in renderers.values():
        renderer.prepare()
        consume(renderer,views[:10])
    torch.cuda.synchronize()
    report['end_to_end'] = {label:{'trials':[]} for label in renderers}
    for trial in range(10):
        order = list(renderers) if trial%2==0 else list(reversed(renderers))
        for label in order:
            renderer=renderers[label]
            torch.cuda.synchronize()
            start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            start.record(); before=perf_counter()
            consume(renderer,views)
            end.record(); torch.cuda.synchronize()
            elapsed=(perf_counter()-before)*1000/len(views)
            row={'wall_ms':elapsed,'cuda_span_ms':start.elapsed_time(end)/len(views),'fps':1000/elapsed}
            counts=renderer.last_execution_counts
            assert counts['prefix_graph_replays']==49 and counts['outputs']==50,counts
            assert counts['mixed_launches']==(49 if renderer.fused else 0),counts
            report['end_to_end'][label]['trials'].append(row)
            report['end_to_end'][label]['execution_counts']=dict(counts)
            print('E2E',trial,label,row,flush=True)
        save(args.output/'partial.json',report)
    for value in report['end_to_end'].values():
        value['median']={k:statistics.median(row[k] for row in value['trials']) for k in ('wall_ms','cuda_span_ms','fps')}
    # Direct image/depth equality against the same-precision independent pipe
    # is stricter than comparing aggregate PSNR/SSIM/LPIPS to ground truth.
    reference=[]
    report['quality']={}
    for label,renderer in renderers.items():
        rows=[]
        for i,result in enumerate(renderer.render_sequence(views)):
            image,depth=result['render'].cpu(),result['depth'].cpu()
            assert torch.isfinite(image).all() and torch.isfinite(depth).all()
            if label=='independent': reference.append((image,depth))
            else:
                a,b=reference[i]
                row={'view':i,'rgb_max_abs':float((image-a).abs().max()),
                     'depth_max_abs':float((depth-b).abs().max()),
                     'rgb_bitwise_equal':torch.equal(image,a),'depth_bitwise_equal':torch.equal(depth,b)}
                torch.testing.assert_close(image,a,atol=2e-6,rtol=1e-5)
                torch.testing.assert_close(depth,b,atol=2e-6,rtol=1e-5)
                rows.append(row)
        report['quality'][label]=rows
    report['resources']={k:{'kernel_variant':r.kernel_variant,'blocks':r.physical_blocks,
                            'mixed':r.extension.resources(),'graph_slots':len(r.graphs)} for k,r in renderers.items()}
    report['timing_scope']='normal Python E2E, 50 consecutive views, 10 alternating rounds; excludes setup/JIT/warmup/metrics/I/O'


def main():
    args,pc,scene,pipe,background,views,report=load_workload()
    (sweep if args.stage.startswith('sweep') else formal)(args,pc,scene,pipe,background,views,report)
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

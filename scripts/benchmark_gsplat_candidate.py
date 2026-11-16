#!/usr/bin/env python3
"""Port legacy v11 candidate pairing, with grid and same-binary controls."""
import copy
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from analyze_gsplat_mixed import load_workload,save,sha,stats
from benchmark_gsplat_v7 import seal_extra,v7_state,node_resources,measure,summarize_trials

# Predeclared comparisons; no choice based on measured FPS.
CONFIGS={
    'independent':dict(kernel_variant='tensor_v7',fused=False,physical_blocks=0),
    'current_3486':dict(kernel_variant='tensor_v7',fused=True,physical_blocks=3486),
    'grid_5440':dict(kernel_variant='tensor_v7',fused=True,physical_blocks=5440),
    'ca_off_5440':dict(kernel_variant='tensor_v7_ca',fused=True,physical_blocks=5440,candidate_pairing=False),
    'ca_on_5440':dict(kernel_variant='tensor_v7_ca',fused=True,physical_blocks=5440,candidate_pairing=True),
}


def setup_report(args,report):
    seal_extra(args,report)
    sources=[Path(__file__),ROOT/'gsplat_mixed_ext/mixed_tensor_v7_ca.cu',
             ROOT/'gsplat_mixed_ext/bindings_tensor_v7_ca.cpp',ROOT/'gaussian_renderer/gsplat_pairing.py',
             ROOT/'tests/test_gsplat_candidate_cuda.py',ROOT/'tests/test_gsplat_candidate_pairing.py',
             ROOT/'scripts/run_jetson_gsplat_candidate.sh',
             ROOT/'benchmarks/protocols/gsplat_candidate_pairing_20261027.json']
    for source in sources:
        report['source_and_checkpoint_sha256'][str(source.resolve())]=sha(source)
        dest=args.output/'runtime_sources'/source.relative_to(ROOT)
        dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(source.read_bytes())
    report.update(kind='gsplat_v11_candidate_pairing',configs=CONFIGS,
        qualification_role='parent workload and native two-slot HexPlane Graph only; new leaf separately checked',
        mapping_rule='one base Deform tile per Raster CTA; one extra tile to largest-candidate CTA per contiguous group; ties first',
        fallback_rule='ordinary grid stride unless physical==Raster tiles and 0<Deform-Raster<Raster',
        comparison_scope='same checkpoint, prefix, MLP arithmetic and precision; no Raster prepass or fused HexPlane reduction')
    report['predeclared_protocol']=json.loads(sources[-1].read_text())


def renderers(args,pc,scene,pipe,background):
    from gaussian_renderer.gsplat_mixed import GsplatMixedRenderer
    result={}
    for label,config in CONFIGS.items():
        result[label]=GsplatMixedRenderer(pc,copy.copy(pipe),background,args.qualification_profile,
            cam_type=scene.dataset_type,**config)
        result[label].prepare()
    return result


def launch(renderer,prepared,mlp,profile=False):
    arguments=(prepared.inputs,mlp,prepared.width,prepared.height,renderer.physical_blocks,2,profile)
    if renderer.kernel_variant=='tensor_v7_ca':
        return renderer.extension.forward(*arguments,renderer.candidate_pairing)
    return renderer.extension.forward(*arguments)


def audit_pairing(renderer,prepared,mlp,out_path):
    import torch
    from gaussian_renderer.gsplat_pairing import candidate_pairing_plan
    raw=launch(renderer,prepared,mlp,True)
    offsets=prepared.inputs[4].flatten().cpu().tolist()
    end=prepared.inputs[5].numel()
    counts=[(offsets[i+1] if i+1<len(offsets) else end)-a for i,a in enumerate(offsets)]
    plan=candidate_pairing_plan(counts,(mlp[0].shape[0]+15)//16,renderer.physical_blocks,renderer.candidate_pairing)
    assignments=raw[9].cpu().tolist()
    assert [row[0] for row in assignments]==[len(tiles) for tiles in plan['tiles']]
    assert [row[1] for row in assignments]==plan['extra_ranks']
    assert [bool(row[2]) for row in assignments]==[plan['active']]*len(assignments)
    assert [row[3] for row in assignments]==counts
    flat=[tile for tiles in plan['tiles'] for tile in tiles]
    assert sorted(flat)==list(range((mlp[0].shape[0]+15)//16))
    windows=raw[8].cpu().tolist()
    overlap=[max(0,min(re,de)-max(rs,ds))/min(re-rs,de-ds)
             for rs,re,ds,de in windows if rs and ds and re>rs and de>ds]
    assert overlap and any(x>0 for x in overlap)
    selected=[i for i,rank in enumerate(plan['extra_ranks']) if rank>=0]
    multiple=[i for i,tiles in enumerate(plan['tiles']) if len(tiles)>1]
    save(out_path,{'counts':counts,'assignments':assignments,'plan':plan,'clock_windows':windows})
    return {'active':plan['active'],'physical_ctas':len(assignments),'selected_extra_workers':len(selected),
            'mlp_tiles_executed':sum(row[0] for row in assignments),'all_tiles_exactly_once':True,
            'workers_with_multiple_mlp_tiles':len(multiple),
            'multiple_worker_candidate_count_mean':statistics.mean(counts[i] for i in multiple) if multiple else None,
            'selected_candidate_count_mean':statistics.mean(counts[i] for i in selected) if selected else None,
            'overall_candidate_count_mean':statistics.mean(counts),
            'overlap_fraction':sum(x>0 for x in overlap)/len(overlap),
            'median_overlap_over_shorter':statistics.median(overlap),
            'assignment_sha256':hashlib.sha256(json.dumps(assignments).encode()).hexdigest()}


def leaf(args,pc,scene,pipe,background,views,report):
    import torch
    from gaussian_renderer import prepare_render_context,deform_for_render
    from gsplat.cuda._wrapper import rasterize_to_pixels
    from gaussian_renderer.gsplat_mixed import prepare_gsplat
    from tests.test_gsplat_tensor_v7_cuda import LegacyV7
    from sampling_prepass_priority import CudaDriver
    runners=renderers(args,pc,scene,pipe,background)
    base=runners['independent'];legacy=LegacyV7()
    report['leaf_cases']=[]
    for index in (0,24,48):
        context=prepare_render_context(views[index],pc,pipe,background,cam_type=scene.dataset_type)
        state=deform_for_render(context,pc) if index==0 else v7_state(context,pc,base)
        prepared=prepare_gsplat(context,state)
        next_context=prepare_render_context(views[index+1],pc,pipe,background,cam_type=scene.dataset_type)
        grid=pc._deformation.deformation_net.grid
        features=getattr(grid,'forward_cuda_graph',grid)(next_context.means3D,next_context.timestamp).contiguous()
        mlp=[features,*base.leaf_cached]
        r=prepared.inputs
        def pixels(): return rasterize_to_pixels(*r[:4],prepared.width,prepared.height,16,r[4],r[5],backgrounds=r[6])
        def separate(): return (*pixels(),*base.extension.mlp_forward(mlp))
        expected=(*pixels(),*legacy(mlp))
        arms={'raster_standalone':pixels,'mlp_standalone':lambda:base.extension.mlp_forward(mlp),
              'independent':separate}
        for label,renderer in runners.items():
            if label!='independent': arms[label]=lambda renderer=renderer:launch(renderer,prepared,mlp)
        stream,driver=torch.cuda.Stream(),CudaDriver()
        stream.wait_stream(torch.cuda.current_stream())
        graphs,outputs={},{}
        case={'raster_view':index,'deform_view':index+1,'arms':{}}
        for label,callback in arms.items():
            with torch.cuda.stream(stream):
                for _ in range(2): callback()
            stream.synchronize()
            owner=torch.cuda.CUDAGraph(keep_graph=True)
            with torch.cuda.graph(owner,stream=stream): outputs[label]=callback()
            graphs[label]=owner
            with torch.cuda.stream(stream): owner.replay()
            stream.synchronize()
            raw=outputs[label]
            if label=='raster_standalone': pairs=list(zip(raw,expected[:2]))
            elif label=='mlp_standalone': pairs=list(zip(raw,expected[2:]))
            elif label=='independent': pairs=list(zip(raw,expected))
            else: pairs=list(zip(raw[:2],expected[:2]))+list(zip(raw[3:8],expected[2:]))
            assert all(torch.equal(a,b) for a,b in pairs),(index,label)
            nodes=driver.kernel_nodes(owner.raw_cuda_graph())
            assert len(nodes)==(2 if label=='independent' else 1),nodes
            case['arms'][label]={'rounds':[],'bitwise_equal':True,'nodes':nodes,'resources':node_resources(driver,owner)}
        for trial in range(6):
            order=list(arms) if trial%2==0 else list(reversed(arms))
            for label in order:
                a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                with torch.cuda.stream(stream):
                    for _ in range(5): graphs[label].replay()
                    a.record(stream)
                    for _ in range(20): graphs[label].replay()
                    b.record(stream)
                b.synchronize();elapsed=a.elapsed_time(b)/20
                case['arms'][label]['rounds'].append(elapsed)
                print('LEAF',index,trial,label,elapsed,flush=True)
        for value in case['arms'].values(): value['timing']=stats(value['rounds'])
        case['mapping']={label:audit_pairing(runners[label],prepared,mlp,args.output/f'mapping_{index:02d}_{label}.json')
                         for label in ('ca_off_5440','ca_on_5440')}
        assert not case['mapping']['ca_off_5440']['active'] and case['mapping']['ca_on_5440']['active']
        assert case['mapping']['ca_on_5440']['selected_extra_workers']==1531
        report['leaf_cases'].append(case);save(args.output/'partial.json',report)
        del graphs,outputs,arms,state,expected
        torch.cuda.synchronize()
    report['leaf_average_ms']={label:statistics.mean(case['arms'][label]['timing']['median_ms']
                              for case in report['leaf_cases']) for label in report['leaf_cases'][0]['arms']}
    report['timing_scope']='leaf Graph, 3 real frame pairs, 6 alternating rounds, 5 warmup +20 replays; excludes prefix/prepare/allocation/JIT'


def verify_counts(renderer,count):
    c=renderer.last_execution_counts
    assert c['outputs']==count and c['prefix_graph_replays']==count-1,c
    assert c['mixed_launches']==(count-1 if renderer.fused else 0),c
    assert c['separate_leaf_pairs']==(0 if renderer.fused else count-1),c
    assert renderer.last_fallback_reason is None
    return dict(c)


def formal(args,pc,scene,pipe,background,views,report):
    import torch
    from gaussian_renderer import prepare_render_context,render
    from gaussian_renderer.gsplat_mixed import prepare_gsplat,render_prepared
    runners=renderers(args,pc,scene,pipe,background)
    base=runners['independent']
    for renderer in runners.values():
        for a,b in zip(base.leaf_cached,renderer.leaf_cached): assert torch.equal(a,b)
    def consume(renderer):
        for _ in renderer.render_sequence(views): pass
    for renderer in runners.values():
        for _ in renderer.render_sequence(views[:10]): pass
    report['end_to_end']={label:{'trials':[]} for label in runners}
    for trial in range(10):
        order=list(runners) if trial%2==0 else list(reversed(runners))
        for label in order:
            row=measure(lambda:consume(runners[label]),len(views))
            report['end_to_end'][label]['trials'].append(row)
            report['end_to_end'][label]['execution_counts']=verify_counts(runners[label],len(views))
            print('E2E',trial,label,row,flush=True)
        save(args.output/'partial.json',report)
    for value in report['end_to_end'].values(): value['median']=summarize_trials(value['trials'])

    contexts=[prepare_render_context(view,pc,pipe,background,cam_type=scene.dataset_type) for view in views]
    states=[v7_state(context,pc,base) for context in contexts]
    dummy=pc.get_xyz.new_empty((pc.get_xyz.shape[0],32));mlp=[dummy,*base.leaf_cached]
    raster_labels=('native','current_3486','ca_off_5440','ca_on_5440')
    def raster_sequence(label):
        for context,state in zip(contexts,states):
            prepared=prepare_gsplat(context,state)
            if label=='native': render_prepared(prepared)
            else:
                renderer=runners[label]
                arguments=(prepared.inputs,mlp,prepared.width,prepared.height,renderer.physical_blocks,0,False)
                raw=renderer.extension.forward(*arguments,renderer.candidate_pairing) if renderer.kernel_variant=='tensor_v7_ca' else renderer.extension.forward(*arguments)
                prepared.result(raw[0])
    report['raster_only_controls']={label:{'trials':[]} for label in raster_labels}
    for label in raster_labels: raster_sequence(label)
    for trial in range(10):
        for label in (raster_labels if trial%2==0 else tuple(reversed(raster_labels))):
            row=measure(lambda:raster_sequence(label),len(views))
            report['raster_only_controls'][label]['trials'].append(row)
            print('RASTER',trial,label,row,flush=True)
    for value in report['raster_only_controls'].values(): value['median']=summarize_trials(value['trials'])
    report['raster_only_scope']='same cached activated states, prepare+RGBD pixels+layout; MLP disabled, resource controls are not subgroup times in Both'
    del states,contexts,dummy,mlp
    torch.cuda.synchronize();torch.cuda.empty_cache();save(args.output/'partial.json',report)

    from lpipsPyTorch.modules.lpips import LPIPS
    from utils.image_utils import psnr
    from utils.loss_utils import ssim
    metric=LPIPS('vgg','0.1').cuda().eval()
    native_ref=[];independent_ref=[];report['quality']={}
    class ReplayAudit:
        def __init__(self,owner): self.owner,self.calls=owner,0
        def replay(self): self.calls+=1;return self.owner.replay()
    for label in ('gsplat_native',*runners):
        renderer=runners.get(label);audits=[];rows=[]
        if renderer:
            audits=[ReplayAudit(slot.graph) for slot in renderer.graphs]
            for slot,audit in zip(renderer.graphs,audits): slot.graph=audit
            outputs=renderer.render_sequence(views)
        else: outputs=(render(view,pc,pipe,background,cam_type=scene.dataset_type) for view in views)
        try:
            for i,result in enumerate(outputs):
                raw,depth=result['render'],result['depth']
                assert torch.isfinite(raw).all() and torch.isfinite(depth).all()
                cpu,dep=raw.cpu(),depth.cpu()
                if label=='gsplat_native': native_ref.append((cpu,dep))
                if label=='independent': independent_ref.append((cpu,dep))
                image=raw.clamp(0,1);gt=views[i].original_image[:3].cuda()
                row={'view':i,'psnr_db':float(psnr(image[None],gt[None]).mean()),
                     'ssim':float(ssim(image[None],gt[None])),'lpips':float(metric(image[None],gt[None]).mean())}
                for name,reference in (('native',native_ref),('independent',independent_ref)):
                    if name=='independent' and label=='gsplat_native': continue
                    a,b=reference[i]
                    row.update({f'rgb_mae_vs_{name}':float((cpu-a).abs().mean()),
                                f'rgb_max_abs_vs_{name}':float((cpu-a).abs().max()),
                                f'depth_mae_vs_{name}':float((dep-b).abs().mean()),
                                f'depth_max_abs_vs_{name}':float((dep-b).abs().max())})
                    if name=='independent':
                        assert torch.equal(cpu,a) and torch.equal(dep,b),(label,i)
                rows.append(row)
        finally:
            if renderer:
                for slot,audit in zip(renderer.graphs,audits): slot.graph=audit.owner
        value={'per_view':rows,'mean':{k:statistics.mean(row[k] for row in rows) for k in rows[0] if k!='view'}}
        if renderer:
            assert sum(a.calls for a in audits)==49
            value['actual_graph_replays_per_slot']=[a.calls for a in audits]
        report['quality'][label]=value
        print('QUALITY',label,{k:value['mean'][k] for k in ('psnr_db','ssim','lpips')},flush=True)
        save(args.output/'partial.json',report)
    for label in runners:
        value=report['quality'][label];reference=report['quality']['gsplat_native']['mean'];q=value['mean']
        delta={'psnr_drop_db':reference['psnr_db']-q['psnr_db'],'ssim_drop':reference['ssim']-q['ssim'],'lpips_increase':q['lpips']-reference['lpips']}
        limits={'psnr_drop_db':0.05,'ssim_drop':1e-4,'lpips_increase':1e-4}
        value.update(delta_vs_native=delta,thresholds=limits,quality_gate_passed=all(delta[k]<=v for k,v in limits.items()))
    report['resources']={label:{'mixed':r.extension.resources(),'standalone':r.extension.standalone_resources(),
                               'graph_slots':len(r.graphs)} for label,r in runners.items()}
    report['timing_scope']='normal Python E2E, same checkpoint/native two-slot Graph, 50 views, 10 alternating rounds; includes prefill/drain/context/prefix/MLP/activation/prepare/RGBD/sync, excludes loading/JIT/warmup/metrics/I/O'


def main():
    args,pc,scene,pipe,background,views,report=load_workload()
    setup_report(args,report)
    (leaf if args.stage=='sweep' else formal)(args,pc,scene,pipe,background,views,report)
    report['inputs_unchanged']=all(sha(p)==h for p,h in report['source_and_checkpoint_sha256'].items())
    assert report['inputs_unchanged']
    binaries={str(Path(module.__file__).resolve()) for module in list(sys.modules.values())
              if (getattr(module,'__file__','') or '').endswith('.so')
              and any(name in module.__file__ for name in ('gsplat','tacker_4dgs','diff_gaussian'))}
    report['runtime_binary_sha256']={p:sha(p) for p in sorted(binaries)}
    report['completed']=True;save(args.output/'report.json',report)
    print('DONE',args.output/'report.json',flush=True)


if __name__=='__main__': main()

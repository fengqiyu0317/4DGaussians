"""CPU audit reference for legacy v11 candidate-aware Deform tile assignment.

This is not called on the timed rendering path. The CUDA implementation reads
gsplat's tile offsets directly and needs no preprocessing kernel.
"""


def candidate_pairing_plan(candidate_counts, deform_tiles, physical_blocks=None, enabled=True):
    counts=tuple(int(x) for x in candidate_counts)
    raster_tiles=len(counts)
    blocks=raster_tiles if physical_blocks is None else int(physical_blocks)
    if not counts or any(x<0 for x in counts) or deform_tiles<0 or blocks<=0:
        raise ValueError('invalid candidate counts, tile count or physical grid')
    extra=deform_tiles-raster_tiles
    paired=bool(enabled and blocks==raster_tiles and 0<extra<raster_tiles)
    if not paired:
        return {'active':False,'tiles':[list(range(b,deform_tiles,blocks)) for b in range(blocks)],
                'extra_ranks':[-1]*blocks}
    tiles=[[b] for b in range(blocks)]
    ranks=[-1]*blocks
    # Group-forward reference, independent of the CUDA CTA-to-group formula.
    for group in range(extra):
        begin,end=group*raster_tiles//extra,(group+1)*raster_tiles//extra
        winner=max(range(begin,end),key=lambda b:counts[b])
        ranks[winner]=group
        tiles[winner].append(raster_tiles+group)
    return {'active':True,'tiles':tiles,'extra_ranks':ranks}

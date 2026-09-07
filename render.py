#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
import imageio
import numpy as np
import torch
from scene import Scene
import os
import cv2
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render, TackerRenderer, TwoStreamRenderer
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args, ModelHiddenParams
from gaussian_renderer import GaussianModel
from time import time
import threading
import concurrent.futures
def multithread_write(image_list, path):
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=None)
    def write_image(image, count, path):
        try:
            torchvision.utils.save_image(image, os.path.join(path, '{0:05d}'.format(count) + ".png"))
            return count, True
        except:
            return count, False
        
    tasks = []
    for index, image in enumerate(image_list):
        tasks.append(executor.submit(write_image, image, index, path))
    executor.shutdown()
    for index, status in enumerate(tasks):
        if status == False:
            write_image(image_list[index], index, path)
    
to8b = lambda x : (255*np.clip(x.cpu().numpy(),0,1)).astype(np.uint8)


def _create_sequence_renderer(
    gaussians,
    pipeline,
    background,
    cam_type,
    execution_mode,
    tacker_profile,
    workload_name,
    iteration,
):
    if execution_mode == "two_stream":
        return TwoStreamRenderer(
            gaussians,
            pipeline,
            background,
            stage="fine",
            cam_type=cam_type,
        )
    if execution_mode == "tacker":
        return TackerRenderer(
            gaussians,
            pipeline,
            background,
            stage="fine",
            cam_type=cam_type,
            profile_path=tacker_profile,
            workload_name=workload_name,
            iteration=iteration,
        )
    if execution_mode != "serial":
        raise ValueError("unknown execution mode: {}".format(execution_mode))
    return None


def _render_outputs(
    views,
    gaussians,
    pipeline,
    background,
    cam_type,
    execution_mode,
    tacker_profile,
    workload_name,
    iteration,
    sequence_renderer=None,
):
    if execution_mode == "serial":
        for view in views:
            yield render(
                view,
                gaussians,
                pipeline,
                background,
                cam_type=cam_type,
            )
        return

    renderer = sequence_renderer
    if renderer is None:
        renderer = _create_sequence_renderer(
            gaussians,
            pipeline,
            background,
            cam_type,
            execution_mode,
            tacker_profile,
            workload_name,
            iteration,
        )

    iterator = renderer.render_sequence(views)
    try:
        for _ in range(len(views)):
            try:
                yield next(iterator)
            except StopIteration:
                raise RuntimeError(
                    "{} renderer returned fewer frames than requested".format(
                        execution_mode
                    )
                )
        try:
            next(iterator)
        except StopIteration:
            pass
        else:
            raise RuntimeError(
                "{} renderer returned more frames than requested".format(
                    execution_mode
                )
            )
    finally:
        iterator.close()
        renderer.synchronize()

    if execution_mode == "two_stream" and renderer.last_fallback_reason is not None:
        print("two_stream fell back to serial: {}".format(renderer.last_fallback_reason))
    if execution_mode == "tacker" and renderer.last_fallback_reason is not None:
        print(
            "tacker fell back to {}: {}".format(
                renderer.actual_execution_mode,
                renderer.last_fallback_reason,
            )
        )


def render_set(
    model_path,
    name,
    iteration,
    views,
    gaussians,
    pipeline,
    background,
    cam_type,
    execution_mode="serial",
    tacker_profile=None,
    workload_name=None,
):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    gt_list = []
    render_list = []
    print("point nums:",gaussians._xyz.shape[0])

    # Construct streams/events, validate the profile, and finish immutable head
    # parameter conversion before the FPS interval.  The output generator is
    # lazy, so merely calling iter(_render_outputs(...)) is not sufficient.
    sequence_renderer = _create_sequence_renderer(
        gaussians,
        pipeline,
        background,
        cam_type,
        execution_mode,
        tacker_profile,
        workload_name,
        iteration,
    )
    if sequence_renderer is not None:
        prepare = getattr(sequence_renderer, "prepare", None)
        if callable(prepare):
            prepare()
    torch.cuda.synchronize()
    time1 = time()
    outputs = iter(_render_outputs(
        views,
        gaussians,
        pipeline,
        background,
        cam_type,
        execution_mode,
        tacker_profile,
        workload_name,
        iteration,
        sequence_renderer=sequence_renderer,
    ))
    try:
        for view in tqdm(views, total=len(views), desc="Rendering progress"):
            try:
                output = next(outputs)
            except StopIteration:
                raise RuntimeError(
                    "{} renderer returned fewer frames than requested".format(
                        execution_mode
                    )
                )
            rendering = output["render"]
            render_list.append(rendering)
            if name in ["train", "test"]:
                if cam_type != "PanopticSports":
                    gt = view.original_image[0:3, :, :]
                else:
                    gt  = view['image'].cuda()
                gt_list.append(gt)
        try:
            next(outputs)
        except StopIteration:
            pass
        else:
            raise RuntimeError(
                "{} renderer returned more frames than requested".format(
                    execution_mode
                )
            )
    finally:
        outputs.close()

    torch.cuda.synchronize()
    time2 = time()
    print("Execution mode:", execution_mode)
    print("FPS:", len(views) / (time2 - time1) if len(views) else 0.0)

    # CPU transfers and encoding are deliberately outside the render timing
    # and after the complete stream pipeline has been submitted.
    render_images = [to8b(image).transpose(1, 2, 0) for image in render_list]

    multithread_write(gt_list, gts_path)

    multithread_write(render_list, render_path)

    
    imageio.mimwrite(os.path.join(model_path, name, "ours_{}".format(iteration), 'video_rgb.mp4'), render_images, fps=30)


def render_sets(
    dataset : ModelParams,
    hyperparam,
    iteration : int,
    pipeline : PipelineParams,
    skip_train : bool,
    skip_test : bool,
    skip_video: bool,
    execution_mode="serial",
    tacker_profile=None,
    workload_name=None,
):
    if execution_mode == "tacker":
        if tacker_profile is None:
            raise ValueError("tacker mode requires --tacker-profile")
        if workload_name is None:
            raise ValueError("tacker mode requires --workload-name")
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree, hyperparam)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
        gaussians._deformation.eval()
        cam_type=scene.dataset_type
        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
            render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, cam_type, execution_mode, tacker_profile, workload_name)

        if not skip_test:
            render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background, cam_type, execution_mode, tacker_profile, workload_name)
        if not skip_video:
            render_set(dataset.model_path, "video", scene.loaded_iter, scene.getVideoCameras(), gaussians, pipeline, background, cam_type, execution_mode, tacker_profile, workload_name)
if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    hyperparam = ModelHiddenParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--skip_video", action="store_true")
    parser.add_argument("--configs", type=str)
    parser.add_argument(
        "--execution-mode",
        "--execution_mode",
        dest="execution_mode",
        choices=("serial", "two_stream", "tacker"),
        default="serial",
    )
    parser.add_argument("--tacker-profile", type=str)
    parser.add_argument("--workload-name", type=str)
    args = get_combined_args(parser)
    print("Rendering " , args.model_path)
    if args.configs:
        from utils.params_utils import load_config, merge_hparams
        config = load_config(args.configs)
        args = merge_hparams(args, config)
    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(
        model.extract(args),
        hyperparam.extract(args),
        args.iteration,
        pipeline.extract(args),
        args.skip_train,
        args.skip_test,
        args.skip_video,
        args.execution_mode,
        args.tacker_profile,
        args.workload_name,
    )

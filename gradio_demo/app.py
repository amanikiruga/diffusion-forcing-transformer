from typing import List, Literal
from pathlib import Path
from functools import partial
import gradio as gr
import numpy as np
import torch
from torchvision.datasets.utils import download_and_extract_archive
from einops import repeat
from omegaconf import OmegaConf
from algorithms.dfot import DFoTVideoPose
from history_guidance import HistoryGuidance
from utils.ckpt_utils import download_pretrained
from datasets.video.utils.io import read_video
from export import export_to_video, export_to_gif, export_images_to_gif
from camera_pose import extend_poses, CameraPose
from scipy.spatial.transform import Rotation, Slerp

DATASET_URL = "https://huggingface.co/kiwhansong/DFoT/resolve/main/datasets/RealEstate10K_Tiny.tar.gz"
DATASET_DIR = Path("data/real-estate-10k-tiny")
LONG_LENGTH = 10  # seconds
NAVIGATION_FPS = 3

if not DATASET_DIR.exists():
    DATASET_DIR.mkdir(parents=True)
    download_and_extract_archive(
        DATASET_URL,
        DATASET_DIR.parent,
        remove_finished=True,
    )


metadata = torch.load(DATASET_DIR / "metadata" / "test.pt", weights_only=False)
video_list = [
    read_video(path).permute(0, 3, 1, 2) / 255.0 for path in metadata["video_paths"]
]
poses_list = [
    torch.cat(
        [
            poses[:, :4],
            poses[:, 6:],
        ],
        dim=-1,
    ).to(torch.float32)
    for poses in (
        torch.load(DATASET_DIR / "test_poses" / f"{path.stem}.pt")
        for path in metadata["video_paths"]
    )
]

first_frame_list = [
    (video[0] * 255).permute(1, 2, 0).numpy().clip(0, 255).astype("uint8")
    for video in video_list
]
gif_paths = []
for idx, video, path in zip(
    range(len(video_list)), video_list, metadata["video_paths"]
):
    indices = torch.linspace(0, video.size(0) - 1, 16, dtype=torch.long)
    gif_paths.append(export_to_gif(video[indices], fps=8))


# pylint: disable-next=no-value-for-parameter
dfot = DFoTVideoPose.load_from_checkpoint(
    checkpoint_path=download_pretrained("pretrained:DFoT_RE10K.ckpt"),
    map_location="cpu",
    cfg=OmegaConf.load("config.yaml"),
).eval()
dfot.to("cuda")


def prepare_long_gt_video(idx: int):
    video = video_list[idx]
    indices = torch.linspace(0, video.size(0) - 1, 200, dtype=torch.long)
    return export_to_video(video[indices], fps=200 // LONG_LENGTH)


def prepare_short_gt_video(idx: int):
    video = video_list[idx]
    indices = torch.linspace(0, video.size(0) - 1, 8, dtype=torch.long)
    video = (
        (video[indices].permute(0, 2, 3, 1) * 255).clamp(0, 255).to(torch.uint8).numpy()
    )
    return [video[i] for i in range(video.shape[0])]


def video_to_gif_and_images(video, indices):
    masked_video = [
        image if i in indices else np.zeros_like(image) for i, image in enumerate(video)
    ]
    return [(export_images_to_gif(masked_video), "GIF")] + [
        (image, f"t={i}" if i in indices else "")
        for i, image in enumerate(masked_video)
    ]


def get_duration_single_image_to_long_video(idx: int, guidance_scale: float, fps: int, progress:gr.Progress):
    return 30 * fps


@torch.autocast("cuda")
@torch.no_grad()
def single_image_to_long_video(
    idx: int, guidance_scale: float, fps: int, progress=gr.Progress(track_tqdm=True)
):
    video = video_list[idx]
    poses = poses_list[idx]
    indices = torch.linspace(0, video.size(0) - 1, LONG_LENGTH * fps, dtype=torch.long)
    xs = video[indices].unsqueeze(0).to("cuda")
    conditions = poses[indices].unsqueeze(0).to("cuda")
    dfot.cfg.tasks.prediction.history_guidance.guidance_scale = guidance_scale
    dfot.cfg.tasks.prediction.keyframe_density = 12 / (fps * LONG_LENGTH)
    # dfot.cfg.tasks.interpolation.history_guidance.guidance_scale = guidance_scale
    gen_video = dfot._unnormalize_x(
        dfot._predict_videos(
            dfot._normalize_x(xs),
            conditions,
        )
    )
    return export_to_video(gen_video[0].detach().cpu(), fps=fps)


@torch.autocast("cuda")
@torch.no_grad()
def any_images_to_short_video(
    scene_idx: int,
    image_indices: List[int],
    guidance_scale: float,
):
    video = video_list[scene_idx]
    poses = poses_list[scene_idx]
    indices = torch.linspace(0, video.size(0) - 1, 8, dtype=torch.long)
    xs = video[indices].unsqueeze(0).to("cuda")
    conditions = poses[indices].unsqueeze(0).to("cuda")
    pbar = CustomProgressBar(
        gr.Progress(track_tqdm=True).tqdm(
            iterable=None,
            desc="Sampling with DFoT",
            total=dfot.sampling_timesteps,
        )
    )
    gen_video = dfot._unnormalize_x(
        dfot._sample_sequence(
            batch_size=1,
            context=dfot._normalize_x(xs),
            context_mask=torch.tensor([i in image_indices for i in range(8)])
            .unsqueeze(0)
            .to("cuda"),
            conditions=conditions,
            history_guidance=HistoryGuidance.vanilla(
                guidance_scale=guidance_scale,
                visualize=False,
            ),
            pbar=pbar,
        )[0]
    )
    gen_video = (
        (gen_video[0].detach().cpu().permute(0, 2, 3, 1) * 255)
        .clamp(0, 255)
        .to(torch.uint8)
        .numpy()
    )
    return video_to_gif_and_images([image for image in gen_video], list(range(8)))


class CustomProgressBar:
    def __init__(self, pbar):
        self.pbar = pbar

    def set_postfix(self, **kwargs):
        pass

    def __getattr__(self, attr):
        return getattr(self.pbar, attr)

def get_duration_navigate_video(video: torch.Tensor,
    poses: torch.Tensor,
    x_angle: float,
    y_angle: float,
    distance: float
):
    if abs(x_angle) < 30 and abs(y_angle) < 30 and distance < 150:
        return 45
    return 30

@torch.autocast("cuda")
@torch.no_grad()
def navigate_video(
    video: torch.Tensor,
    poses: torch.Tensor,
    x_angle: float,
    y_angle: float,
    distance: float,
):
    n_context_frames = min(len(video), 4)
    n_prediction_frames = 8 - n_context_frames
    pbar = CustomProgressBar(
        gr.Progress(track_tqdm=True).tqdm(
            iterable=None,
            desc=f"Predicting next {n_prediction_frames} frames with DFoT",
            total=dfot.sampling_timesteps,
        )
    )
    xs = dfot._normalize_x(video.clone().unsqueeze(0).to("cuda"))
    conditions = poses.clone().unsqueeze(0).to("cuda")
    conditions = extend_poses(
        conditions,
        n=n_prediction_frames,
        x_angle=x_angle,
        y_angle=y_angle,
        distance=distance,
    )
    context_mask = (
        torch.cat(
            [
                torch.ones(1, n_context_frames) * (1 if n_context_frames == 1 else 2),
                torch.zeros(1, n_prediction_frames),
            ],
            dim=-1,
        )
        .long()
        .to("cuda")
    )
    next_video = (
        dfot._unnormalize_x(
            dfot._sample_sequence(
                batch_size=1,
                context=torch.cat(
                    [
                        xs[:, -n_context_frames:],
                        torch.zeros(
                            1,
                            n_prediction_frames,
                            *xs.shape[2:],
                            device=xs.device,
                            dtype=xs.dtype,
                        ),
                    ],
                    dim=1,
                ),
                context_mask=context_mask,
                conditions=conditions[:, -8:],
                history_guidance=HistoryGuidance.smart(
                    x_angle=x_angle,
                    y_angle=y_angle,
                    distance=distance,
                    visualize=False,
                ),
                pbar=pbar,
            )[0]
        )[0][n_context_frames:]
        .detach()
        .cpu()
    )
    gen_video = torch.cat([video, next_video], dim=0)
    poses = conditions[0].detach().cpu()

    images = (gen_video.permute(0, 2, 3, 1) * 255).clamp(0, 255).to(torch.uint8).numpy()

    return (
        gen_video,
        poses,
        images[-1],
        export_to_video(gen_video, fps=NAVIGATION_FPS),
        [(image, f"t={i}") for i, image in enumerate(images)],
    )


def undo_navigation(
    video: torch.Tensor,
    poses: torch.Tensor,
):
    if len(video) > 8:
        video = video[:-4]
        poses = poses[:-4]
    elif len(video) == 8:
        video = video[:1]
        poses = poses[:1]
    else:
        gr.Warning("You have no moves left to undo!")
    images = (video.permute(0, 2, 3, 1) * 255).clamp(0, 255).to(torch.uint8).numpy()
    return (
        video,
        poses,
        images[-1],
        export_to_video(video, fps=NAVIGATION_FPS),
        [(image, f"t={i}") for i, image in enumerate(images)],
    )


def _interpolate_conditions(conditions, indices):
    """
    Interpolate conditions to fill out missing frames

    Aegs:
        conditions (Tensor): conditions (B, T, C)
        indices (Tensor): indices of keyframes (T')
    """
    assert indices[0].item() == 0
    assert indices[-1].item() == conditions.shape[1] - 1

    indices = indices.cpu().numpy()
    batch_size, n_tokens, _ = conditions.shape
    t = np.linspace(0, n_tokens - 1, n_tokens)

    key_conditions = conditions[:, indices]
    poses = CameraPose.from_vectors(key_conditions)
    extrinsics = poses.extrinsics().cpu().numpy()
    ps = extrinsics[..., :3, 3]
    rs = extrinsics[..., :3, :3].reshape(batch_size, -1, 3, 3)

    interp_extrinsics = np.zeros((batch_size, n_tokens, 3, 4))
    for i in range(batch_size):
        slerp = Slerp(indices, Rotation.from_matrix(rs[i]))
        interp_extrinsics[i, :, :3, :3] = slerp(t).as_matrix()
        for j in range(3):
            interp_extrinsics[i, :, j, 3] = np.interp(t, indices, ps[i, :, j])
    interp_extrinsics = torch.from_numpy(interp_extrinsics.astype(np.float32))
    interp_extrinsics = interp_extrinsics.to(conditions.device).flatten(2)
    conditions = repeat(key_conditions[:, 0, :4], "b c -> b t c", t=n_tokens)
    conditions = torch.cat([conditions.clone(), interp_extrinsics], dim=-1)

    return conditions


def _interpolate_between(
    xs: torch.Tensor,
    conditions: torch.Tensor,
    interpolation_factor: int,
    progress=gr.Progress(track_tqdm=True),
):
    l = xs.shape[1]
    final_l = (l - 1) * interpolation_factor + 1
    x_shape = xs.shape[2:]
    context = torch.zeros(
        (
            1,
            final_l,
            *x_shape,
        ),
        device=xs.device,
        dtype=xs.dtype,
    )
    long_conditions = torch.zeros(
        (1, final_l, *conditions.shape[2:]),
        device=conditions.device,
        dtype=conditions.dtype,
    )
    context_mask = torch.zeros(
        (1, final_l),
        device=xs.device,
        dtype=torch.bool,
    )
    context_indices = torch.arange(
        0, final_l, interpolation_factor, device=conditions.device
    )
    context[:, context_indices] = xs
    long_conditions[:, context_indices] = conditions
    context_mask[:, ::interpolation_factor] = True
    long_conditions = _interpolate_conditions(
        long_conditions,
        context_indices,
    )

    xs = dfot._interpolate_videos(
        context,
        context_mask,
        conditions=long_conditions,
    )
    return xs, long_conditions


def get_duration_smooth_navigation(
    video: torch.Tensor, poses: torch.Tensor, interpolation_factor: int, progress: gr.Progress
):
    length = (len(video) - 1) * interpolation_factor + 1
    return 2 * length


@torch.autocast("cuda")
@torch.no_grad()
def smooth_navigation(
    video: torch.Tensor,
    poses: torch.Tensor,
    interpolation_factor: int,
    progress=gr.Progress(track_tqdm=True),
):
    if len(video) < 8:
        gr.Warning("Navigate first before applying temporal super-resolution!")
    else:
        video, poses = _interpolate_between(
            dfot._normalize_x(video.clone().unsqueeze(0).to("cuda")),
            poses.clone().unsqueeze(0).to("cuda"),
            interpolation_factor,
        )
        video = dfot._unnormalize_x(video)[0].detach().cpu()
        poses = poses[0].detach().cpu()
    images = (video.permute(0, 2, 3, 1) * 255).clamp(0, 255).to(torch.uint8).numpy()
    return (
        video,
        poses,
        images[-1],
        export_to_video(video, fps=NAVIGATION_FPS * interpolation_factor),
        [(image, f"t={i}") for i, image in enumerate(images)],
    )

def render_demo1(s: Literal["Selection", "Generation"], idx: int, demo1_stage: gr.State, demo1_selected_index: gr.State):
    gr.Markdown(
        f"""
        ## Demo 1: Single Image → Long {LONG_LENGTH}-second Video
        > #### _Diffusion Forcing Transformer can generate long videos via sliding window rollouts and temporal super-resolution._
    """,
    elem_classes=["task-title"]
    )
    match s:
        case "Selection":
            with gr.Group():
                demo1_image_gallery = gr.Gallery(
                    height=300,
                    value=first_frame_list,
                    label="Select an Image to Animate",
                    columns=[8],
                    selected_index=idx,
                    allow_preview=False,
                    preview=False,
                )

                @demo1_image_gallery.select(
                    inputs=None, outputs=[demo1_stage, demo1_selected_index]
                )
                def move_to_generation(selection: gr.SelectData):
                    return "Generation", selection.index

        case "Generation":
            with gr.Row():
                gr.Image(
                    value=first_frame_list[idx],
                    label="Input Image",
                    width=256,
                    height=256,
                )
                gr.Video(
                    value=prepare_long_gt_video(idx),
                    label="Ground Truth Video",
                    width=256,
                    height=256,
                    autoplay=True,
                    loop=True,
                )
                demo1_video = gr.Video(
                    label="Generated Video",
                    width=256,
                    height=256,
                    autoplay=True,
                    loop=True,
                    show_share_button=True,
                    show_download_button=True,
                )

            gr.Markdown("### Generation Controls ↓")
            demo1_guidance_scale = gr.Slider(
                minimum=1,
                maximum=6,
                value=4,
                step=0.5,
                label="History Guidance Scale",
                info="Without history guidance: 1.0; Recommended: 4.0",
                interactive=True,
            )
            demo1_fps = gr.Slider(
                minimum=4,
                maximum=20,
                value=4,
                step=1,
                label="FPS",
                info=f"A {LONG_LENGTH}-second video will be generated at this FPS; Decrease for faster generation; Increase for a smoother video",
                interactive=True,
            )
            gr.Button("Generate Video", variant="primary").click(
                fn=single_image_to_long_video,
                inputs=[
                    demo1_selected_index,
                    demo1_guidance_scale,
                    demo1_fps,
                ],
                outputs=demo1_video,
            )

def render_demo2(s: Literal["Scene", "Image", "Generation"], scene_idx: int, image_indices: List[int], demo2_stage: gr.State, demo2_selected_scene_index: gr.State, demo2_selected_image_indices: gr.State):
    gr.Markdown(
        """
        ## Demo 2: Any Number of Images → Short 2-second Video
        > #### _Diffusion Forcing Transformer is a flexible model that can generate videos given variable number of context frames._
    """,
    elem_classes=["task-title"]
    )

    match s:
        case "Scene":
            with gr.Group():
                demo2_scene_gallery = gr.Gallery(
                    height=300,
                    value=gif_paths,
                    label="Select a Scene to Generate Video",
                    columns=[8],
                    selected_index=scene_idx,
                    allow_preview=False,
                    preview=False,
                )

                @demo2_scene_gallery.select(
                    inputs=None, outputs=[demo2_stage, demo2_selected_scene_index]
                )
                def move_to_image_selection(selection: gr.SelectData):
                    return "Image", selection.index

        case "Image":
            with gr.Group():
                demo2_image_gallery = gr.Gallery(
                    height=150,
                    value=[
                        (image, f"t={i}")
                        for i, image in enumerate(
                            prepare_short_gt_video(scene_idx)
                        )
                    ],
                    label="Select Input Images",
                    columns=[8],
                )

                demo2_selector = gr.CheckboxGroup(
                    label="Select Any Number of Input Images",
                    info="Image-to-Video: Select t=0; Interpolation: Select t=0 and t=7",
                    choices=[(f"t={i}", i) for i in range(8)],
                    value=[],
                )
                demo2_image_select_button = gr.Button(
                    "Next Step", variant="primary"
                )

                @demo2_image_select_button.click(
                    inputs=[demo2_selector],
                    outputs=[demo2_stage, demo2_selected_image_indices],
                )
                def generate_video(selected_indices):
                    if len(selected_indices) == 0:
                        gr.Warning("Select at least one image!")
                        return "Image", []
                    else:
                        return "Generation", selected_indices

        case "Generation":
            with gr.Group():
                gt_video = prepare_short_gt_video(scene_idx)

                demo2_input_image_gallery = gr.Gallery(
                    height=150,
                    value=video_to_gif_and_images(gt_video, image_indices),
                    label="Input Images",
                    columns=[9],
                )
                demo2_generated_gallery = gr.Gallery(
                    height=150,
                    value=[],
                    label="Generated Video",
                    columns=[9],
                )

                demo2_ground_truth_gallery = gr.Gallery(
                    height=150,
                    value=video_to_gif_and_images(gt_video, list(range(8))),
                    label="Ground Truth Video",
                    columns=[9],
                )
            gr.Markdown("### Generation Controls ↓")
            demo2_guidance_scale = gr.Slider(
                minimum=1,
                maximum=6,
                value=4,
                step=0.5,
                label="History Guidance Scale",
                info="Without history guidance: 1.0; Recommended: 4.0",
                interactive=True,
            )
            gr.Button("Generate Video", variant="primary").click(
                fn=any_images_to_short_video,
                inputs=[
                    demo2_selected_scene_index,
                    demo2_selected_image_indices,
                    demo2_guidance_scale,
                ],
                outputs=demo2_generated_gallery,
            )

def render_demo3(
    s: Literal["Selection", "Generation"],
    idx: int,
    demo3_stage: gr.State,
    demo3_selected_index: gr.State,
    demo3_current_video: gr.State,
    demo3_current_poses: gr.State
):
    gr.Markdown(
        """
        ## Demo 3: Single Image → Extremely Long Video _(Navigate with Your Camera Movements!)_
        > #### _History Guidance significantly improves quality and temporal consistency, enabling stable rollouts for extremely long videos._
    """,
    elem_classes=["task-title"]
    )
    match s:
        case "Selection":
            with gr.Group():
                demo3_image_gallery = gr.Gallery(
                    height=300,
                    value=first_frame_list,
                    label="Select an Image to Start Navigation",
                    columns=[8],
                    selected_index=idx,
                    allow_preview=False,
                    preview=False,
                )

                @demo3_image_gallery.select(
                    inputs=None, outputs=[demo3_stage, demo3_selected_index, demo3_current_video, demo3_current_poses]
                )
                def move_to_generation(selection: gr.SelectData):
                    idx = selection.index
                    return (
                        "Generation",
                        idx,
                        video_list[idx][:1],
                        poses_list[idx][:1],
                    )

        case "Generation":
            with gr.Row():
                with gr.Column(scale=3):
                    with gr.Row():
                        demo3_current_view = gr.Image(
                            value=first_frame_list[idx],
                            label="Current View",
                            width=256,
                            height=256,
                        )
                        demo3_video = gr.Video(
                            label="Generated Video",
                            width=256,
                            height=256,
                            autoplay=True,
                            loop=True,
                            show_share_button=True,
                            show_download_button=True,
                        )

                    demo3_generated_gallery = gr.Gallery(
                        value=[],
                        label="Generated Frames",
                        columns=[6],
                    )

                with gr.Column():
                    gr.Markdown("### Navigation Controls ↓")
                    with gr.Accordion("Instructions", open=False):
                        gr.Markdown("""
                            - **The model will predict the next few frames based on your camera movements. Repeat the process to continue navigating through the scene.**
                            - **At the end of your navigation, apply temporal super-resolution to increase the FPS,** also utilizing the DFoT model.
                            - The most suitable history guidance scheme will be automatically selected based on your camera movements.    
                        """)
                    with gr.Tab("Basic", elem_id="basic-controls-tab"):
                        with gr.Group():
                            gr.Markdown("_**Select a direction to move:**_")
                            with gr.Row(elem_id="basic-controls"):
                                gr.Button(
                                    "↰-60°\nVeer",
                                    size="sm",
                                    min_width=0,
                                    variant="primary",
                                ).click(
                                    fn=partial(
                                        navigate_video,
                                        x_angle=0,
                                        y_angle=-60,
                                        distance=0,
                                    ),
                                    inputs=[
                                        demo3_current_video,
                                        demo3_current_poses,
                                    ],
                                    outputs=[
                                        demo3_current_video,
                                        demo3_current_poses,
                                        demo3_current_view,
                                        demo3_video,
                                        demo3_generated_gallery,
                                    ],
                                )

                                gr.Button(
                                    "↖-30°\nTurn",
                                    size="sm",
                                    min_width=0,
                                    variant="primary",
                                ).click(
                                    fn=partial(
                                        navigate_video,
                                        x_angle=0,
                                        y_angle=-30,
                                        distance=50,
                                    ),
                                    inputs=[
                                        demo3_current_video,
                                        demo3_current_poses,
                                    ],
                                    outputs=[
                                        demo3_current_video,
                                        demo3_current_poses,
                                        demo3_current_view,
                                        demo3_video,
                                        demo3_generated_gallery,
                                    ],
                                )

                                gr.Button(
                                    "↑0°\nAhead",
                                    size="sm",
                                    min_width=0,
                                    variant="primary",
                                ).click(
                                    fn=partial(
                                        navigate_video,
                                        x_angle=0,
                                        y_angle=0,
                                        distance=100,
                                    ),
                                    inputs=[
                                        demo3_current_video,
                                        demo3_current_poses,
                                    ],
                                    outputs=[
                                        demo3_current_video,
                                        demo3_current_poses,
                                        demo3_current_view,
                                        demo3_video,
                                        demo3_generated_gallery,
                                    ],
                                )
                                gr.Button(
                                    "↗30°\nTurn",
                                    size="sm",
                                    min_width=0,
                                    variant="primary",
                                ).click(
                                    fn=partial(
                                        navigate_video,
                                        x_angle=0,
                                        y_angle=30,
                                        distance=50,
                                    ),
                                    inputs=[
                                        demo3_current_video,
                                        demo3_current_poses,
                                    ],
                                    outputs=[
                                        demo3_current_video,
                                        demo3_current_poses,
                                        demo3_current_view,
                                        demo3_video,
                                        demo3_generated_gallery,
                                    ],
                                )
                                gr.Button(
                                    "↱\n60° Veer",
                                    size="sm",
                                    min_width=0,
                                    variant="primary",
                                ).click(
                                    fn=partial(
                                        navigate_video,
                                        x_angle=0,
                                        y_angle=60,
                                        distance=0,
                                    ),
                                    inputs=[
                                        demo3_current_video,
                                        demo3_current_poses,
                                    ],
                                    outputs=[
                                        demo3_current_video,
                                        demo3_current_poses,
                                        demo3_current_view,
                                        demo3_video,
                                        demo3_generated_gallery,
                                    ],
                                )
                    with gr.Tab("Advanced", elem_id="advanced-controls-tab"):
                        with gr.Group():
                            gr.Markdown("_**Select angles and distance:**_")

                            demo3_y_angle = gr.Slider(
                                minimum=-90,
                                maximum=90,
                                value=0,
                                step=10,
                                label="Horizontal Angle",
                                interactive=True,
                            )
                            demo3_x_angle = gr.Slider(
                                minimum=-40,
                                maximum=40,
                                value=0,
                                step=10,
                                label="Vertical Angle",
                                interactive=True,
                            )
                            demo3_distance = gr.Slider(
                                minimum=0,
                                maximum=200,
                                value=100,
                                step=10,
                                label="Distance",
                                interactive=True,
                            )

                            gr.Button(
                                "Generate Next Move", variant="primary"
                            ).click(
                                fn=navigate_video,
                                inputs=[
                                    demo3_current_video,
                                    demo3_current_poses,
                                    demo3_x_angle,
                                    demo3_y_angle,
                                    demo3_distance,
                                ],
                                outputs=[
                                    demo3_current_video,
                                    demo3_current_poses,
                                    demo3_current_view,
                                    demo3_video,
                                    demo3_generated_gallery,
                                ],
                            )
                    gr.Markdown("---")
                    with gr.Group():
                        gr.Markdown("_You can always undo your last move:_")
                        gr.Button("Undo Last Move", variant="huggingface").click(
                            fn=undo_navigation,
                            inputs=[demo3_current_video, demo3_current_poses],
                            outputs=[
                                demo3_current_video,
                                demo3_current_poses,
                                demo3_current_view,
                                demo3_video,
                                demo3_generated_gallery,
                            ],
                        )
                    with gr.Group():
                        gr.Markdown(
                            "_At the end, apply temporal super-resolution to obtain a smoother video:_"
                        )
                        demo3_interpolation_factor = gr.Slider(
                            minimum=2,
                            maximum=10,
                            value=2,
                            step=1,
                            label="By a Factor of",
                            interactive=True,
                        )
                        gr.Button("Smooth Out Video", variant="huggingface").click(
                            fn=smooth_navigation,
                            inputs=[
                                demo3_current_video,
                                demo3_current_poses,
                                demo3_interpolation_factor,
                            ],
                            outputs=[
                                demo3_current_video,
                                demo3_current_poses,
                                demo3_current_view,
                                demo3_video,
                                demo3_generated_gallery,
                            ],
                        )
        
    

# Create the Gradio Blocks
with gr.Blocks(theme=gr.themes.Base(primary_hue="teal")) as demo:
    gr.HTML(
        """
    <style>
    [data-tab-id="task-1"], [data-tab-id="task-2"], [data-tab-id="task-3"] {
        font-size: 16px !important;
        font-weight: bold;
    }
    #page-title h1 {
        color: #0D9488 !important;
    }
    .task-title h2 {
        color: #F59E0C !important;
    }
    .header-button-row {
        gap: 4px !important;
    }
    .header-button-row div {
        width: 131.0px !important;
    }

    .header-button-column {
        width: 131.0px !important;
        gap: 5px !important;
    }
    .header-button a {
        border: 1px solid #e4e4e7;
    }
    .header-button .button-icon {
        margin-right: 8px;
    }
    .demo-button-column .gap {
        gap: 5px !important;
    }
    #basic-controls {
        column-gap: 0px;
    }
    #basic-controls-tab {
        padding: 0px;
    }
    #advanced-controls-tab {
        padding: 0px;
    }
    #selected-demo-button {
        color: #F59E0C;
        text-decoration: underline;
    }
    .demo-button {
        text-align: left !important;
        display: block !important;
    }
    </style>
    """
    )

    demo_idx = gr.State(value=1)

    with gr.Sidebar():
        gr.Markdown("# Diffusion Forcing Transformer with History Guidance", elem_id="page-title")
        gr.Markdown(
            "### Official Interactive Demo for [_History-Guided Video Diffusion_](https://arxiv.org/abs/2502.06764)"
        )
        gr.Markdown("---")
        gr.Markdown("#### Links ↓")
        with gr.Row(elem_classes=["header-button-row"]):
            with gr.Column(elem_classes=["header-button-column"], min_width=0):
                gr.Button(
                    value="Website",
                    link="https://boyuan.space/history-guidance",
                    icon="https://simpleicons.org/icons/googlechrome.svg",
                    elem_classes=["header-button"],
                    size="md",
                    min_width=0,
                )
                gr.Button(
                    value="Paper",
                    link="https://arxiv.org/abs/2502.06764",
                    icon="https://simpleicons.org/icons/arxiv.svg",
                    elem_classes=["header-button"],
                    size="md",
                    min_width=0,
                )
            with gr.Column(elem_classes=["header-button-column"], min_width=0):
                gr.Button(
                    value="Code",
                    link="https://github.com/kwsong0113/diffusion-forcing-transformer",
                    icon="https://simpleicons.org/icons/github.svg",
                    elem_classes=["header-button"],
                    size="md",
                    min_width=0,
                )
                gr.Button(
                    value="Weights",
                    link="https://huggingface.co/kiwhansong/DFoT",
                    icon="https://simpleicons.org/icons/huggingface.svg",
                    elem_classes=["header-button"],
                    size="md",
                    min_width=0,
                )
        gr.Markdown("---")
        gr.Markdown("#### Choose a Demo ↓")
        with gr.Column(elem_classes=["demo-button-column"]):
            @gr.render(inputs=[demo_idx])
            def render_demo_tabs(idx):
                demo_tab_button1 = gr.Button(
                    "1: Image → Long Video",
                    size="md", elem_classes=["demo-button"], **{"elem_id": "selected-demo-button"} if idx == 1 else {}
                ).click(
                    fn=lambda: 1,
                    outputs=demo_idx
                )
                demo_tab_button2 = gr.Button(
                    "2: Any # of Images → Short Video",
                    size="md", elem_classes=["demo-button"], **{"elem_id": "selected-demo-button"} if idx == 2 else {}
                ).click(
                    fn=lambda: 2,
                    outputs=demo_idx
                )
                demo_tab_button3 = gr.Button(
                    "3: Image → Extremely Long Video",
                    size="md", elem_classes=["demo-button"],  **{"elem_id": "selected-demo-button"} if idx == 3 else {}
                ).click(
                    fn=lambda: 3,
                    outputs=demo_idx
                )
        gr.Markdown("---")
        gr.Markdown("#### Troubleshooting ↓")
        with gr.Group():
            with gr.Accordion("Error or Unexpected Results?", open=False):
                gr.Markdown("Please try again after refreshing the page and ensure you do not click the same button multiple times.")
            with gr.Accordion("Too Slow or No GPU Allocation?", open=False):
                gr.Markdown(
                    "Consider running the demo locally (click the dots in the top-right corner). Alternatively, you can subscribe to Hugging Face Pro for an increased GPU quota."
                )

    demo1_stage = gr.State(value="Selection")
    demo1_selected_index = gr.State(value=None)
    demo2_stage = gr.State(value="Scene")
    demo2_selected_scene_index = gr.State(value=None)
    demo2_selected_image_indices = gr.State(value=[])
    demo3_stage = gr.State(value="Selection")
    demo3_selected_index = gr.State(value=None)
    demo3_current_video = gr.State(value=None)
    demo3_current_poses = gr.State(value=None)

    @gr.render(inputs=[demo_idx, demo1_stage, demo1_selected_index, demo2_stage, demo2_selected_scene_index, demo2_selected_image_indices, demo3_stage, demo3_selected_index])
    def render_demo(
        _demo_idx, _demo1_stage, _demo1_selected_index, _demo2_stage, _demo2_selected_scene_index, _demo2_selected_image_indices, _demo3_stage, _demo3_selected_index
    ):
        match _demo_idx:
            case 1:
                render_demo1(_demo1_stage, _demo1_selected_index, demo1_stage, demo1_selected_index)
            case 2:
                render_demo2(_demo2_stage, _demo2_selected_scene_index, _demo2_selected_image_indices,
                    demo2_stage, demo2_selected_scene_index, demo2_selected_image_indices)
            case 3:
                render_demo3(_demo3_stage, _demo3_selected_index, demo3_stage, demo3_selected_index, demo3_current_video, demo3_current_poses)
                

if __name__ == "__main__":
    # demo.launch()
    demo.queue().launch(debug=True, share=False) 
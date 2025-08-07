import trimesh
import torch
import time
from typing import Callable


def bbox(vertices: torch.Tensor) -> torch.Tensor:
    min_, max_ = vertices.min(dim=0).values, vertices.max(dim=0).values
    return torch.stack([min_, max_], dim=0)


def normalize(vertices: torch.Tensor) -> torch.Tensor:
    """
    uniform scale and center to unit cube
    """
    min_, max_ = bbox(vertices)
    scale = (max_ - min_).max() / 2
    center_ = (min_ + max_) / 2
    return (vertices - center_) / scale


def _combine_scene(mesh: trimesh.Scene) -> tuple[torch.Tensor, torch.Tensor]:
    geom = list(mesh.geometry.values())
    vertex_counts = torch.cat([torch.tensor([len(m.vertices)]) for m in geom])
    offsets = [vertex_counts[:i].sum().item() for i in range(len(geom))]
    vertices = torch.cat([torch.tensor(m.vertices) for m in geom])
    faces = torch.cat(
        [torch.tensor(m.faces) + offset for m, offset in zip(geom, offsets)]
    )

    return vertices, faces


def load_obj(path: str) -> tuple[torch.Tensor, torch.Tensor]:
    scene = trimesh.load(path)

    vertices, faces = _combine_scene(scene)
    vertices = normalize(vertices)

    mesh = trimesh.Trimesh(vertices, faces)
    mesh.process(validate=True)

    vertices = torch.tensor(mesh.vertices).float()
    faces = torch.tensor(mesh.faces).long()

    return vertices, faces


def save_obj(path: str, vertices: torch.Tensor, faces: torch.Tensor):
    mesh = trimesh.Trimesh(vertices.cpu().numpy(), faces.cpu().numpy())
    mesh.export(path)


class Timer:
    def __init__(self, print_time=True, label=None, print_fn: Callable = print):
        self.print_time = print_time
        self.label = label if label is not None else "Time taken:"
        self.print_fn = print_fn

    def __enter__(self) -> "Timer":
        self.start = time.perf_counter()

        return self

    def __exit__(self, *args, **kwargs):
        self.end = time.perf_counter()
        self.elapsed = self.end - self.start
        if self.print_time:
            self.print_fn(f"{self.label} {self.elapsed:.6f} seconds")

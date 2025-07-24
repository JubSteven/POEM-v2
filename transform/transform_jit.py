import torch


@torch.jit.script
def assemble_T(tsl: torch.Tensor, rotmat: torch.Tensor):
    # tsl [..., 3]
    # rotmat [..., 3, 3]
    leading_shape = list(tsl.shape[:-1])
    # leading_dim = len(leading_shape)

    res = torch.zeros(leading_shape + [4, 4]).to(tsl)
    res[..., 3, 3] = 1.0
    res[..., :3, 3] = tsl
    res[..., :3, :3] = rotmat
    return res


@torch.jit.script
def inv_transf(transf: torch.Tensor):
    leading_shape = transf.shape[:-2]
    leading_dim = len(leading_shape)

    R_inv = torch.transpose(transf[..., :3, :3], leading_dim, leading_dim + 1)
    t_inv = -R_inv @ transf[..., :3, 3:]
    res = torch.zeros_like(transf)
    res[..., :3, :3] = R_inv
    res[..., :3, 3:] = t_inv
    res[..., 3, 3] = 1
    return res


@torch.jit.script
def transf_point_array(transf: torch.Tensor, point: torch.Tensor):
    # transf: [..., 4, 4]
    # point: [..., X, 3]
    leading_shape = point.shape[:-2]
    leading_dim = len(leading_shape)

    res = (
        torch.transpose(
            torch.matmul(
                transf[..., :3, :3],
                torch.transpose(point, leading_dim, leading_dim + 1),
            ),
            leading_dim,
            leading_dim + 1,
        )  # [..., X, 3]
        + transf[..., :3, 3][..., None, :]  # [..., 1, 3]
    )
    return res


@torch.jit.script
def project_point_array(cam_intr: torch.Tensor, point: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    leading_shape = point.shape[:-2]
    leading_dim = len(leading_shape)

    hom_2d = torch.transpose(
        torch.matmul(
            cam_intr,
            torch.transpose(point, leading_dim, leading_dim + 1),
        ),
        leading_dim,
        leading_dim + 1,
    )  # [..., X, 3]
    xy = hom_2d[..., 0:2]
    z = hom_2d[..., 2:]
    z[torch.abs(z) < eps] = eps
    uv = xy / z
    return uv


@torch.jit.script
def inv_rotmat(rotmat: torch.Tensor):
    leading_shape = rotmat.shape[:-2]
    leading_dim = len(leading_shape)

    rotmat_inv = torch.transpose(rotmat[..., :3, :3], leading_dim, leading_dim + 1)
    return rotmat_inv


@torch.jit.script
def rotate_point_array(rotmat: torch.Tensor, point: torch.Tensor) -> torch.Tensor:
    # rotmat: [..., 3, 3]
    # point: [..., X, 3]
    leading_shape = point.shape[:-2]
    leading_dim = len(leading_shape)

    res = torch.transpose(
        torch.matmul(
            rotmat,
            torch.transpose(point, leading_dim, leading_dim + 1),
        ),
        leading_dim,
        leading_dim + 1,
    )
    return res

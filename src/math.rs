/// Creates perspective projection as expected by WebGPU
///
/// Convention: Camera at origin, looks along -z in a right-handed coordinate system
/// Normalized Device Coordinates: Cube with corner points (-1, -1, 0) and (1, 1, 1)
#[expect(clippy::deprecated_cfg_attr)]
pub fn perspective_projection(
    fov_deg: f32,
    aspect_ratio: f32,
    near: f32,
    far: f32,
) -> cgmath::Matrix4<f32> {
    let fov_rad = fov_deg * (std::f32::consts::PI / 180.);
    let c = 1. / (fov_rad / 2.).tan();
    let a = aspect_ratio;
    let f = far;
    let n = near;

    let c0r0 = c / a;
    let c0r1 = 0.;
    let c0r2 = 0.;
    let c0r3 = 0.;

    let c1r0 = 0.;
    let c1r1 = c;
    let c1r2 = 0.;
    let c1r3 = 0.;

    let c2r0 = 0.;
    let c2r1 = 0.;
    let c2r2 = -f / (f - n);
    let c2r3 = -1.;

    let c3r0 = 0.;
    let c3r1 = 0.;
    let c3r2 = -(f * n) / (f - n);
    let c3r3 = 0.;

    #[cfg_attr(rustfmt, rustfmt_skip)]
    cgmath::Matrix4::new(
        c0r0, c0r1, c0r2, c0r3,
        c1r0, c1r1, c1r2, c1r3,
        c2r0, c2r1, c2r2, c2r3,
        c3r0, c3r1, c3r2, c3r3,
    )
}

use nalgebra as na;

pub fn rotation_towards(dir: na::Vector3<f32>, up: na::Vector3<f32>) -> na::Matrix3<f32> {
    let right = dir.cross(&up).normalize();
    let up = right.cross(&dir);
    na::Matrix3::from_columns(&[right, up, -dir])
}

pub fn view_matrix(pos: na::Vector3<f32>, rot: na::Matrix3<f32>) -> na::Matrix4<f32> {
    let translation = -rot.transpose() * pos;
    let rotation = rot.transpose();
    let mut view = na::Matrix4::<f32>::identity();
    view.fixed_view_mut::<3, 3>(0, 0).copy_from(&rotation);
    view.fixed_view_mut::<3, 1>(0, 3).copy_from(&translation);
    view
}

/// Creates perspective projection as expected by WebGPU
///
/// Convention: Camera at origin, looks along -z in a right-handed coordinate system
/// Normalized Device Coordinates: Cube with corner points (-1, -1, 0) and (1, 1, 1)
#[expect(clippy::deprecated_cfg_attr)]
pub fn perspective_projection(
    fovy_deg: f32,
    aspect_ratio: f32,
    near: f32,
    far: f32,
) -> nalgebra::Matrix4<f32> {
    let fovy_rad = fovy_deg * (std::f32::consts::PI / 180.);
    let c = 1. / (fovy_rad / 2.).tan();
    let a = aspect_ratio;
    let f = far;
    let n = near;

    let r0c0 = c / a;
    let r1c0 = 0.;
    let r2c0 = 0.;
    let r3c0 = 0.;

    let r0c1 = 0.;
    let r1c1 = c;
    let r2c1 = 0.;
    let r3c1 = 0.;

    let r0c2 = 0.;
    let r1c2 = 0.;
    let r2c2 = -f / (f - n);
    let r3c2 = -1.;

    let r0c3 = 0.;
    let r1c3 = 0.;
    let r2c3 = -(f * n) / (f - n);
    let r3c3 = 0.;

    #[cfg_attr(rustfmt, rustfmt_skip)]
    nalgebra::Matrix4::new(
        r0c0, r0c1, r0c2, r0c3,
        r1c0, r1c1, r1c2, r1c3,
        r2c0, r2c1, r2c2, r2c3,
        r3c0, r3c1, r3c2, r3c3,
    )
}

pub fn orthogonal_basis_for_normal(normal: &na::Unit<na::Vector3<f32>>) -> na::Matrix3<f32> {
    let unit = if (**normal - *na::Vector3::x_axis()).magnitude() > 0.1 {
        na::Vector3::x_axis()
    } else {
        na::Vector3::y_axis()
    };
    let tangent = na::Unit::new_normalize(normal.cross(&unit));
    let bitangent = na::Unit::new_normalize(normal.cross(&tangent));

    na::Matrix3::from_columns(&[**normal, *tangent, *bitangent])
}

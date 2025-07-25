
@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> @builtin(position) vec4<f32> {
    if vertex_index == 0 {
        return vec4<f32>(-1., -1., 0., 1.);
    }
    if vertex_index == 1 {
        return vec4<f32>(1., -1., 0., 1.);
    }
    if vertex_index == 2 {
        return vec4<f32>(1., 1., 0., 1.);
    }
    if vertex_index == 3 {
        return vec4<f32>(-1., -1., 0., 1.);
    }
    if vertex_index == 4 {
        return vec4<f32>(1., 1., 0., 1.);
    }
    // if vertex_index == 5 {
        return vec4<f32>(-1., 1., 0., 1.);
    // }
}

@group(0) @binding(0)
var t_shadow: texture_depth_2d;
@group(0) @binding(1)
var s_shadow: sampler;

@fragment
fn fs_main(@builtin(position) coords: vec4<f32>) -> @location(0) vec4<f32> {
    let size: f32 = 1024;
    let uv = coords.xy / size;
    let depth = textureSample(t_shadow, s_shadow, uv);
    return vec4<f32>(vec3<f32>(depth), 1.);
}

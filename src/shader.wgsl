
struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) color: vec3<f32>,
    @location(2) tex_coords: vec2<f32>,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) vert_pos: vec3<f32>,
    @location(1) color: vec3<f32>,
    @location(2) tex_coords: vec2<f32>,
};

struct CameraUniform {
    view_proj: mat4x4<f32>,
};

// group: first param in set_bind_group()
// binding: from BindGroupLayout
@group(0) @binding(0)
var t_diffuse: texture_2d<f32>;
@group(0) @binding(1)
var s_diffuse: sampler;

@group(1) @binding(0)
var<uniform> camera: CameraUniform;

@vertex
fn vs_main(model: VertexInput) -> VertexOutput {
    // var: mutable, but need type specified
    // let: immutable, but type is inferred
    var out: VertexOutput;
    out.clip_position = camera.view_proj * vec4<f32>(model.position, 1.0);
    out.vert_pos = model.position;
    out.color = model.color;
    out.tex_coords = model.tex_coords;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // @builtin(position) is in framebuffer space, i.e. pixel coordinates (origin top left)
    // with a window 800x600, clip_position would be between 0-800 and 0-600
    return vec4<f32>(in.color, 1.0);
}

@fragment
fn fs_tex(in: VertexOutput) -> @location(0) vec4<f32> {
    return textureSample(t_diffuse, s_diffuse, in.tex_coords);
}


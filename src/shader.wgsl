
struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) color: vec3<f32>,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) vert_pos: vec3<f32>,
    @location(1) color: vec3<f32>,
};

@vertex
fn vs_main(model: VertexInput) -> VertexOutput {
    // var: mutable, but need type specified
    // let: immutable, but type is inferred
    var out: VertexOutput;
    out.clip_position = vec4<f32>(model.position, 1.0);
    out.vert_pos = model.position;
    out.color = model.color;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // @builtin(position) is in framebuffer space, i.e. pixel coordinates (origin top left)
    // with a window 800x600, clip_position would be between 0-800 and 0-600
    return vec4<f32>(in.color, 1.0);
}


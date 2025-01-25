
struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) vert_pos: vec3<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) in_vertex_index: u32) -> VertexOutput {
    // var: mutable, but need type specified
    // let: immutable, but type is inferred
    var out: VertexOutput;
    // calculate vertices from index
    let x = f32(1 - i32(in_vertex_index)) * 0.5;
    let y = f32(i32(in_vertex_index & 1) * 2 - 1) * 0.5;
    out.clip_position = vec4<f32>(x, y, 0.0, 1.0);
    out.vert_pos = out.clip_position.xyz;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // @builtin(position) is in framebuffer space, i.e. pixel coordinates (origin top left)
    // with a window 800x600, clip_position would be between 0-800 and 0-600
    return vec4<f32>(0.3, 0.2, 0.1, 1.0);
}

@fragment
fn fs_rgb(in: VertexOutput) -> @location(0) vec4<f32> {
    return vec4<f32>(in.vert_pos.xy, 0.0, 1.0);
}



struct VertexInput {
    @location(0) position: vec3<f32>,
}

struct InstanceInput {
    @location(5) model_matrix_0: vec4<f32>,
    @location(6) model_matrix_1: vec4<f32>,
    @location(7) model_matrix_2: vec4<f32>,
    @location(8) model_matrix_3: vec4<f32>,

    @location(9) normal_matrix_0: vec3<f32>,
    @location(10) normal_matrix_1: vec3<f32>,
    @location(11) normal_matrix_2: vec3<f32>,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_position: vec3<f32>,
}

struct CubeView {
    proj: mat4x4<f32>
}

struct Light {
    position: vec3<f32>,
    far_plane: f32,
    color: vec3<f32>,
}

@group(0) @binding(0)
var<uniform> cube_view: CubeView;
@group(1) @binding(0)
var<uniform> light: Light;

@vertex
fn vs_main(model: VertexInput, instance: InstanceInput) -> VertexOutput {
    var out: VertexOutput;
    let model_matrix = mat4x4<f32>(
        instance.model_matrix_0,
        instance.model_matrix_1,
        instance.model_matrix_2,
        instance.model_matrix_3,
    );

    let world_position = model_matrix * vec4<f32>(model.position, 1.0);
    out.world_position = world_position.xyz / world_position.w;
    out.clip_position = cube_view.proj * world_position;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) f32 {
    let dist = length(in.world_position - light.position);
    return dist / light.far_plane;
}


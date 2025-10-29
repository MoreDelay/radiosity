struct VertexInput {
    @location(0) position: vec3<f32>,
    @if(TEX_COORD)
    @location(1) tex_coords: vec2<f32>,
    @location(2) normal: vec3<f32>,
    @location(3) tangent: vec3<f32>,
    @location(4) bitangent: vec3<f32>,
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
    @if(TEX_COORD)
    @location(1) tex_coords: vec2<f32>,
    @location(2) tangent_position: vec3<f32>,
    @location(3) tangent_light_position: vec3<f32>,
    @location(4) tangent_view_position: vec3<f32>,
}

struct PhongInput {
    specular_color: vec3<f32>,
    specular_exponent: f32,
    diffuse_color: vec3<f32>,
    ambient_color: vec3<f32>,
}

struct Camera {
    view_pos: vec4<f32>,
    view_proj: mat4x4<f32>,
}

struct Light {
    position: vec3<f32>,
    far_plane: f32,
    color: vec3<f32>,
}

// order from least frequently changed to most frequently changed
// and try to reuse order in many pipelines for fewer switching
// https://toji.dev/webgpu-best-practices/bind-groups#reusing-pipeline-layouts
@group(0) @binding(0)
var<uniform> camera: Camera;

@group(1) @binding(0)
var<uniform> light: Light;

@group(2) @binding(0)
var<uniform> phong: PhongInput;

@group(3) @binding(0)
var t_shadow: texture_cube<f32>;
@group(3) @binding(1)
var s_shadow: sampler;

@if(COLOR_MAP)
@group(constants::COLOR_MAP_SLOT) @binding(0)
var t_diffuse: texture_2d<f32>;
@if(COLOR_MAP)
@group(constants::COLOR_MAP_SLOT) @binding(1)
var s_diffuse: sampler;

@if(NORMAL_MAP)
@group(constants::NORMAL_MAP_SLOT) @binding(0)
var t_normal: texture_2d<f32>;
@if(NORMAL_MAP)
@group(constants::NORMAL_MAP_SLOT) @binding(1)
var s_normal: sampler;


@vertex
fn vs_main(model: VertexInput, instance: InstanceInput) -> VertexOutput {
    // var: mutable, but need type specified
    // let: immutable, but type is inferred
    var out: VertexOutput;
    let model_matrix = mat4x4<f32>(
        instance.model_matrix_0,
        instance.model_matrix_1,
        instance.model_matrix_2,
        instance.model_matrix_3,
    );
    let normal_matrix = mat3x3<f32>(
        instance.normal_matrix_0,
        instance.normal_matrix_1,
        instance.normal_matrix_2,
    );

    let world_normal = normalize(normal_matrix * model.normal);
    let world_tangent = normalize(normal_matrix * model.tangent);
    let world_bitangent = normalize(normal_matrix * model.bitangent);
    let tangent_matrix = transpose(mat3x3<f32>(
        world_tangent,
        world_bitangent,
        world_normal
    ));

    let world_position = model_matrix * vec4<f32>(model.position, 1.0);
    out.clip_position = camera.view_proj * world_position;
    out.world_position = world_position.xyz;

    @if(TEX_COORD)
    {
        out.tex_coords = model.tex_coords;
    }

    out.tangent_position = tangent_matrix * world_position.xyz;
    out.tangent_view_position = tangent_matrix * camera.view_pos.xyz;
    out.tangent_light_position = tangent_matrix * light.position;
    return out;
}

fn is_in_light(world_position: vec3<f32>) -> f32 {
    const bias: f32 = 0.005;
    let dir = world_position - light.position;
    let cube_dir = vec3<f32>(dir.xy, -dir.z);
    let dist = length(cube_dir);
    let depth = dist / light.far_plane - bias;
    let sample = textureSample(t_shadow, s_shadow, cube_dir).r;
    return select(1., 0., depth > sample);
}

fn face_index(dir: vec3<f32>) -> u32 {
    let a = abs(dir);
    if a.x >= a.y && a.x >= a.z {
        return select(1u, 0u, dir.x > 0.0); // 0 = +X, 1 = -X
    }
    if a.y >= a.x && a.y >= a.z {
        return select(3u, 2u, dir.y > 0.0); // 2 = +Y, 3 = -Y
    }
    return select(5u, 4u, dir.z > 0.0);     // 4 = +Z, 5 = -Z
}

@if(NORMAL_MAP)
fn normal_from_map(tex_coords: vec2<f32>) -> vec3<f32> {
    let object_normal: vec4<f32> = textureSample(t_normal, s_normal, tex_coords);
    let tangent_normal = object_normal.xyz * 2.0 - 1.0;
    return tangent_normal;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {

    @if(COLOR_MAP && TEX_COORD)
    let object_color: vec4<f32> = textureSample(t_diffuse, s_diffuse, in.tex_coords);
    @else
    let object_color: vec4<f32> = vec4(1., 1., 1., 1.);

    @if(NORMAL_MAP && TEX_COORD)
    let tangent_normal: vec3<f32> = normal_from_map(in.tex_coords);
    @else
    let tangent_normal: vec3<f32> = vec3(0., 0., 1.);

    let ambient_strength = 0.1;
    let ambient_color = phong.ambient_color * light.color * ambient_strength;

    let light_dir = normalize(in.tangent_light_position - in.tangent_position);
    let diffuse_strength = max(dot(tangent_normal, light_dir), 0.0);
    let diffuse_color = phong.diffuse_color * light.color * diffuse_strength;

    let view_dir = normalize(in.tangent_view_position - in.tangent_position);
    let half_dir = normalize(view_dir + light_dir);
    let specular_strength = pow(max(dot(tangent_normal, half_dir), 0.0), phong.specular_exponent);

    let specular_color = specular_strength * phong.specular_color * light.color;

    let in_light = is_in_light(in.world_position);

    let result = ambient_color + (diffuse_color /* + specular_color */) * in_light;
    return vec4<f32>(result, 1.);
}



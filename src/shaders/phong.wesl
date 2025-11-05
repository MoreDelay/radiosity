struct VertexInput {
    @location(0) position: vec3<f32>,
    @if(HAS_TEX_COORD)
    @location(1) tex_coords: vec2<f32>,
    @if(HAS_NORMAL)
    @location(2) normal: vec3<f32>,
    @if(HAS_TANGENT)
    @location(3) tangent: vec3<f32>,
    @if(HAS_BI_TANGENT)
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
    @if(HAS_TEX_COORD)
    @location(1) tex_coords: vec2<f32>,
    @if(HAS_NORMAL && HAS_TANGENT && HAS_BI_TANGENT)
    @location(2) tangent_position: vec3<f32>,
    @if(HAS_NORMAL && HAS_TANGENT && HAS_BI_TANGENT)
    @location(3) tangent_light_position: vec3<f32>,
    @if(HAS_NORMAL && HAS_TANGENT && HAS_BI_TANGENT)
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

@if(USE_SHADOW_MAP)
@group(constants::SHADOW_MAP_SLOT) @binding(0)
var t_shadow: texture_cube<f32>;
@if(USE_SHADOW_MAP)
@group(constants::SHADOW_MAP_SLOT) @binding(1)
var s_shadow: sampler;

@if(USE_COLOR_MAP)
@group(constants::COLOR_MAP_SLOT) @binding(0)
var t_diffuse: texture_2d<f32>;
@if(USE_COLOR_MAP)
@group(constants::COLOR_MAP_SLOT) @binding(1)
var s_diffuse: sampler;

@if(USE_NORMAL_MAP)
@group(constants::NORMAL_MAP_SLOT) @binding(0)
var t_normal: texture_2d<f32>;
@if(USE_NORMAL_MAP)
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


    let world_position = model_matrix * vec4<f32>(model.position, 1.0);
    out.clip_position = camera.view_proj * world_position;
    out.world_position = world_position.xyz;

    @if(HAS_TEX_COORD)
    {
        out.tex_coords = model.tex_coords;
    }

    @if(HAS_NORMAL && HAS_TANGENT && HAS_BI_TANGENT)
    {
        let world_normal = normalize(normal_matrix * model.normal);
        let world_tangent = normalize(normal_matrix * model.tangent);
        let world_bitangent = normalize(normal_matrix * model.bitangent);
        let tangent_matrix = transpose(mat3x3<f32>(
            world_tangent,
            world_bitangent,
            world_normal
        ));

        out.tangent_position = tangent_matrix * world_position.xyz;
        out.tangent_view_position = tangent_matrix * camera.view_pos.xyz;
        out.tangent_light_position = tangent_matrix * light.position;
    }
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

@if(USE_NORMAL_MAP)
fn normal_from_map(tex_coords: vec2<f32>) -> vec3<f32> {
    let object_normal = textureSample(t_normal, s_normal, tex_coords);
    let tangent_normal = object_normal.xyz * 2.0 - 1.0;
    return tangent_normal;
}

fn phong_ambient(color: vec3<f32>) -> vec3<f32> {
    let ambient_strength = 0.1;
    let ambient_color = color * /* phong.ambient_color * */ light.color * ambient_strength;

    return ambient_color;
}

fn phong_diffuse(
    pos: vec3<f32>,
    light_pos: vec3<f32>,
    normal: vec3<f32>,
    color: vec3<f32>,
) -> vec3<f32> {
    let light_dir = normalize(light_pos - pos);
    let diffuse_strength = max(dot(normal, light_dir), 0.0);
    let diffuse_color = color * phong.diffuse_color * light.color * diffuse_strength;

    return diffuse_color;
}

fn phong_specular(
    pos: vec3<f32>,
    light_pos: vec3<f32>,
    cam_pos: vec3<f32>,
    normal: vec3<f32>
) -> vec3<f32> {
    let view_dir = normalize(cam_pos - pos);
    let light_dir = normalize(light_pos - pos);
    let half_dir = normalize(view_dir + light_dir);
    let specular_strength = pow(max(dot(normal, half_dir), 0.0), phong.specular_exponent);
    let specular_color = specular_strength * phong.specular_color * light.color;

    return specular_color;
}

fn phong_model(in: VertexOutput, normal: vec3<f32>, color: vec3<f32>) -> vec3<f32> {
    let pos = in.tangent_position;
    let light_pos = in.tangent_light_position;
    let cam_pos = in.tangent_view_position;

    let ambient_color = phong_ambient(color);
    let diffuse_color = phong_diffuse(pos, light_pos, normal, color);
    let specular_color = phong_specular(pos, light_pos, cam_pos, normal);

    @if(USE_SHADOW_MAP)
    let in_light = is_in_light(in.world_position);
    @if(!USE_SHADOW_MAP) // @else
    let in_light = 1.;

    return ambient_color + (diffuse_color + specular_color) * in_light;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // get textured color
    @if(USE_COLOR_MAP && HAS_TEX_COORD)
    let object_color = textureSample(t_diffuse, s_diffuse, in.tex_coords);
    @if(!(USE_COLOR_MAP && HAS_TEX_COORD)) // @else
    let object_color = vec4(phong.diffuse_color, 1.);

    // get normal in tangent space
    @if(USE_NORMAL_MAP && HAS_TEX_COORD)
    let tangent_normal = normal_from_map(in.tex_coords);
    @if(!(USE_NORMAL_MAP && HAS_TEX_COORD)) // @else
    let tangent_normal = vec3(0., 0., 1.);

    // compute phong color
    @if(HAS_NORMAL && HAS_TANGENT && HAS_BI_TANGENT)
    let result = phong_model(in, tangent_normal, object_color.xyz);
    @if(!(HAS_NORMAL && HAS_TANGENT && HAS_BI_TANGENT)) // @else
    let result = phong.ambient_color;

    return vec4<f32>(result, 1.);
}



use std::{fs::File, io::BufReader, path::Path};

use anyhow::Context;
use cgmath::{EuclideanSpace, Zero};
use image::ImageReader;

use crate::render::layout::{
    GpuTransfer, GpuTransferRef, InstanceBufferRaw, InstanceRaw, TextureRaw, TriangleBufferRaw,
    VertexRaw,
};

pub const NUM_INSTANCES_PER_ROW: u32 = 10;

#[derive(Copy, Clone, Debug)]
pub struct Instance {
    pub position: cgmath::Vector3<f32>,
    pub rotation: cgmath::Quaternion<f32>,
}

#[derive(Copy, Clone, Debug)]
pub struct Vertex {
    pub position: cgmath::Point3<f32>,
    pub tex_coords: cgmath::Point2<f32>,
    pub normal: cgmath::Vector3<f32>,
    pub tangent: cgmath::Vector3<f32>,
    pub bitangent: cgmath::Vector3<f32>,
}

#[derive(Copy, Clone, Debug)]
pub struct Triplet(u32, u32, u32);

#[derive(Clone, Debug)]
pub struct ColorTexture(pub image::ImageBuffer<image::Rgba<u8>, Vec<u8>>);

#[derive(Clone, Debug)]
pub struct NormalTexture(pub image::ImageBuffer<image::Rgba<u8>, Vec<u8>>);

pub struct Model {
    pub mesh: Mesh,
    pub material: Material,
}

pub struct Material {
    #[expect(unused)]
    pub name: String,
    pub color_texture: Option<ColorTexture>,
    pub normal_texture: Option<NormalTexture>,
}

pub struct Mesh {
    pub vertices: Vec<Vertex>,
    pub triangles: Vec<Triplet>,
    #[expect(unused)]
    pub material_id: usize,
}

impl Material {
    pub fn load(root: &Path, mat: &tobj::Material) -> anyhow::Result<Self> {
        let diffuse_texture = mat
            .diffuse_texture
            .as_ref()
            .ok_or(anyhow::anyhow!("no diffuse texture"))?;
        let diffuse_texture = root.join(diffuse_texture);
        let diffuse_image = ImageReader::open(diffuse_texture)?.decode()?;

        let normal_texture = mat
            .normal_texture
            .as_ref()
            .ok_or(anyhow::anyhow!("no normal texture"))?;
        let normal_texture = root.join(normal_texture);
        let normal_image = ImageReader::open(normal_texture)?.decode()?;

        let color_texture = Some(ColorTexture(diffuse_image.into()));
        let normal_texture = Some(NormalTexture(normal_image.into()));
        Ok(Self {
            name: mat.name.to_string(),
            color_texture,
            normal_texture,
        })
    }
}

impl Model {
    pub fn load(file_name: &Path) -> anyhow::Result<Self> {
        let root = file_name.parent().expect("texture should not be root");
        let obj_file = File::open(file_name).context("could not find obj")?;
        let mut obj_reader = BufReader::new(obj_file);

        let (models, materials) = tobj::load_obj_buf(
            &mut obj_reader,
            &tobj::LoadOptions {
                single_index: true,
                triangulate: true,
                ..Default::default()
            },
            move |path| {
                let path = root.join(path);
                let mtl_file = File::open(path).unwrap();
                let mut mtl_reader = BufReader::new(mtl_file);
                tobj::load_mtl_buf(&mut mtl_reader)
            },
        )?;

        let material = materials?
            .first()
            .map(|mat| Material::load(root, mat))
            .ok_or(anyhow::anyhow!("no material"))??;

        let mesh = models
            .first()
            .map(Mesh::new)
            .ok_or(anyhow::anyhow!("no mesh"))?;

        Ok(Self { mesh, material })
    }
}

impl Mesh {
    fn new(model: &tobj::Model) -> Self {
        assert!(
            model.mesh.positions.len() % 3 == 0,
            "expect only triangle meshes"
        );

        let mut vertices = (0..model.mesh.positions.len() / 3)
            .map(|i| {
                let position = cgmath::Point3 {
                    x: model.mesh.positions[i * 3],
                    y: model.mesh.positions[i * 3 + 1],
                    z: model.mesh.positions[i * 3 + 2],
                };
                let tex_coords = if model.mesh.texcoords.is_empty() {
                    cgmath::Point2::origin()
                } else {
                    cgmath::Point2 {
                        x: model.mesh.texcoords[i * 2],
                        y: 1.0 - model.mesh.texcoords[i * 2 + 1],
                    }
                };
                let normal = if model.mesh.normals.is_empty() {
                    cgmath::Vector3 {
                        x: 0.0,
                        y: 0.0,
                        z: 0.0,
                    }
                } else {
                    cgmath::Vector3 {
                        x: model.mesh.normals[i * 3],
                        y: model.mesh.normals[i * 3 + 1],
                        z: model.mesh.normals[i * 3 + 2],
                    }
                };
                // calculated below
                let tangent = cgmath::Vector3::zero();
                let bitangent = cgmath::Vector3::zero();
                Vertex {
                    position,
                    tex_coords,
                    normal,
                    tangent,
                    bitangent,
                }
            })
            .collect::<Vec<_>>();

        // compute tangents (similar to computing normals)
        assert!(
            model.mesh.indices.len() % 3 == 0,
            "mesh should only contain triangles"
        );
        let triangles = model
            .mesh
            .indices
            .chunks(3)
            .flat_map(<&[u32; 3]>::try_from)
            .map(|&[a, b, c]| Triplet(a, b, c))
            .collect::<Vec<_>>();
        let mut triangles_included = vec![0; vertices.len()];

        for &Triplet(i, j, k) in triangles.iter() {
            let v0 = vertices[i as usize];
            let v1 = vertices[j as usize];
            let v2 = vertices[k as usize];

            let pos0 = v0.position.to_vec();
            let pos1 = v1.position.to_vec();
            let pos2 = v2.position.to_vec();

            let uv0 = v0.tex_coords.to_vec();
            let uv1 = v1.tex_coords.to_vec();
            let uv2 = v2.tex_coords.to_vec();

            let delta_pos1 = pos1 - pos0;
            let delta_pos2 = pos2 - pos0;

            let delta_uv1 = uv1 - uv0;
            let delta_uv2 = uv2 - uv0;

            let r = 1.0 / (delta_uv1.x * delta_uv2.y - delta_uv1.y * delta_uv2.x);
            let tangent = (delta_pos1 * delta_uv2.y - delta_pos2 * delta_uv1.y) * r;
            let bitangent = (delta_pos2 * delta_uv1.x - delta_pos1 * delta_uv2.x) * -r;

            vertices[i as usize].tangent = tangent + vertices[i as usize].tangent;
            vertices[j as usize].tangent = tangent + vertices[j as usize].tangent;
            vertices[k as usize].tangent = tangent + vertices[k as usize].tangent;
            vertices[i as usize].bitangent = bitangent + vertices[i as usize].bitangent;
            vertices[j as usize].bitangent = bitangent + vertices[j as usize].bitangent;
            vertices[k as usize].bitangent = bitangent + vertices[k as usize].bitangent;

            triangles_included[i as usize] += 1;
            triangles_included[j as usize] += 1;
            triangles_included[k as usize] += 1;
        }

        for (i, n) in triangles_included.into_iter().enumerate() {
            let denom = 1.0 / n as f32;
            let v = &mut vertices[i];
            v.tangent *= denom;
            v.bitangent *= denom;
        }
        let material_id = model.mesh.material_id.unwrap_or(0);

        Mesh {
            vertices,
            triangles,
            material_id,
        }
    }
}

// Safety: Triangles consist of 3 indices, and we check that all indices stay within vertex slice.
unsafe impl GpuTransfer for Mesh {
    type Raw = TriangleBufferRaw;

    fn to_raw(&self) -> Self::Raw {
        assert!(self.vertices.len() <= <u32>::MAX as usize);
        let upper_bound = self.vertices.len() as u32;
        let indices = self
            .triangles
            .iter()
            .flat_map(|&Triplet(i, j, k)| {
                assert!(i < upper_bound);
                assert!(j < upper_bound);
                assert!(k < upper_bound);
                [i, j, k]
            })
            .collect();
        let data = self.vertices.iter().map(|v| v.to_raw()).collect();
        TriangleBufferRaw {
            vertices: data,
            indices,
        }
    }
}

// Safety: No additional expectations beyond those of `InstanceRaw`.
unsafe impl GpuTransfer for Vec<Instance> {
    type Raw = InstanceBufferRaw;

    fn to_raw(&self) -> Self::Raw {
        let data = self.iter().map(|i| i.to_raw()).collect();
        InstanceBufferRaw { instances: data }
    }
}

// Safety: set size to the packed dimensions of the image, format matches with stored RGBA pixels.
unsafe impl<'a> GpuTransferRef<'a> for ColorTexture {
    type Raw = TextureRaw<'a>;

    fn to_raw(&'a self) -> TextureRaw<'a> {
        let Self(image) = self;
        let data = &image;
        let dimensions = image.dimensions();
        let size = wgpu::Extent3d {
            width: dimensions.0,
            height: dimensions.1,
            depth_or_array_layers: 1,
        };
        let format = wgpu::TextureFormat::Rgba8UnormSrgb;
        TextureRaw { data, format, size }
    }
}

// Safety: set size to the packed dimensions of the image, format matches with stored RGBA pixels
unsafe impl<'a> GpuTransferRef<'a> for NormalTexture {
    type Raw = TextureRaw<'a>;

    fn to_raw(&'a self) -> TextureRaw<'a> {
        let Self(image) = self;
        let data = &image;
        let dimensions = image.dimensions();
        let size = wgpu::Extent3d {
            width: dimensions.0,
            height: dimensions.1,
            depth_or_array_layers: 1,
        };
        let format = wgpu::TextureFormat::Rgba8Unorm;
        TextureRaw { data, format, size }
    }
}

unsafe impl GpuTransfer for Vertex {
    type Raw = VertexRaw;
    fn to_raw(&self) -> Self::Raw {
        Self::Raw {
            position: self.position.into(),
            tex_coords: self.tex_coords.into(),
            normal: self.normal.into(),
            tangent: self.tangent.into(),
            bitangent: self.bitangent.into(),
        }
    }
}

// Safety: InstanceRaw restricts model to a row-major matrix with bottom right != 0
// and normal to the the same as top left 3x3 matrix of model
unsafe impl GpuTransfer for Instance {
    type Raw = InstanceRaw;
    fn to_raw(&self) -> Self::Raw {
        let model = (cgmath::Matrix4::from_translation(self.position)
            * cgmath::Matrix4::from(self.rotation))
        .into();
        let normal = cgmath::Matrix3::from(self.rotation).into();
        Self::Raw { model, normal }
    }
}

impl From<Triplet> for (u32, u32, u32) {
    fn from(Triplet(i, j, k): Triplet) -> Self {
        (i, j, k)
    }
}

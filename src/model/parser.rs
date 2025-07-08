use std::fs::File;
use std::io::BufReader;
use std::io::prelude::*;
use std::path::Path;
use std::path::PathBuf;

use nom::Offset;
use nom::branch::alt;
use nom::bytes::streaming::take_while1;
use nom::bytes::streaming::{tag, take_until};
use nom::character::streaming::{i32 as parse_i32, line_ending, space1, u32 as parse_u32};
use nom::combinator::{eof, opt, verify};
use nom::multi::{fill, many_m_n, many1};
use nom::number::streaming::float;
use nom::sequence::{delimited, preceded, separated_pair};
use nom::{IResult, Parser};
use thiserror::Error;

#[derive(Debug, PartialEq, Clone, Copy)]
pub struct ObjV {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub w: f32,
}

#[derive(Debug, PartialEq, Clone, Copy)]
pub struct ObjVt {
    pub u: f32,
    pub v: f32,
    pub w: f32,
}

#[derive(Debug, PartialEq, Clone, Copy)]
pub struct ObjVn {
    pub i: f32,
    pub j: f32,
    pub k: f32,
}

#[derive(Debug, PartialEq)]
pub struct ObjVertexTriplet {
    pub index_vertex: usize,
    pub index_texture: Option<usize>,
    pub index_normal: Option<usize>,
}

#[derive(Debug, PartialEq)]
pub struct ObjF {
    pub triplets: Vec<ObjVertexTriplet>,
}

#[derive(Debug, PartialEq, Clone)]
pub struct ObjMtllib {
    pub filepath: PathBuf,
}

#[derive(Debug, PartialEq, Clone)]
pub struct ObjUsemtl {
    pub name: String,
}

#[derive(Debug, PartialEq, Clone, Copy)]
pub struct ObjMaterialSwitch {
    pub material_index: usize,
    pub first_face: usize,
}

pub struct ParsedObj {
    pub vertices: Vec<ObjV>,
    pub texture_coords: Vec<ObjVt>,
    pub normals: Vec<ObjVn>,
    pub faces: Vec<ObjF>,
    pub material_switches: Vec<ObjMaterialSwitch>,
}

pub enum ObjLine {
    Empty,
    GeometricVertex(ObjV),
    TextureCoordinates(ObjVt),
    VertexNormal(ObjVn),
    FaceIndex(ObjF),
    MtlLib(ObjMtllib),
    UseMtl(ObjUsemtl),
}

#[derive(Error, Debug)]
pub enum ParseError {
    #[error("could not read file")]
    IoError(#[from] std::io::Error),
    #[error("file is ill-formed at line {0}")]
    WrongFormat(usize),
}

fn end_of_line(input: &str) -> IResult<&str, ()> {
    let mut parser = alt((line_ending, eof));
    let (input, _) = parser.parse(input)?;
    Ok((input, ()))
}

fn obj_whitespace(input: &str) -> IResult<&str, ()> {
    // allow for line continuation
    let mut parser = many1(alt((space1, preceded(tag("\\"), line_ending))));
    let (input, _whitespace) = parser.parse(input)?;
    Ok((input, ()))
}

pub fn obj_comment(input: &str) -> IResult<&str, &str> {
    let mut parser = preceded(tag("#"), take_until("\n"));
    let (input, comment) = parser.parse(input)?;
    Ok((input, comment.trim()))
}

pub fn obj_blank_line(input: &str) -> IResult<&str, ()> {
    let mut parser = delimited(opt(obj_whitespace), opt(obj_comment), end_of_line);
    let (input, _comment) = parser.parse(input)?;
    Ok((input, ()))
}

pub fn obj_ignore(input: &str) -> IResult<&str, ()> {
    let mut parser = preceded(alt((tag("usemtl"), tag("o"), tag("s"))), take_until("\n"));
    let (input, _) = parser.parse(input)?;
    Ok((input, ()))
}

pub fn obj_geometric_vertex(input: &str) -> IResult<&str, ObjV> {
    let parse_numbers = many_m_n(3, 4, preceded(obj_whitespace, float));
    let mut parser = delimited(tag("v"), parse_numbers, obj_blank_line);
    let (input, mut vertex) = parser.parse(input)?;

    vertex.resize(4, 1.);
    let vertex = if let [x, y, z, w] = vertex[..] {
        ObjV { x, y, z, w }
    } else {
        unreachable!()
    };

    Ok((input, vertex))
}

pub fn obj_texture_coordinates(input: &str) -> IResult<&str, ObjVt> {
    let correct_tex_coord = verify(float, |&num| (0. ..=1.).contains(&num));
    let parse_numbers = many_m_n(1, 3, preceded(obj_whitespace, correct_tex_coord));
    let mut parser = delimited(tag("vt"), parse_numbers, obj_blank_line);
    let (input, mut vertex) = parser.parse(input)?;

    vertex.resize(3, 0.);
    let vertex = if let [u, v, w] = vertex[..] {
        ObjVt { u, v, w }
    } else {
        unreachable!()
    };

    Ok((input, vertex))
}

pub fn obj_vertex_normal(input: &str) -> IResult<&str, ObjVn> {
    let mut normal = [0.; 3];
    // use closure to satisfy Fn requirement, gets changed to FnMut in nom v8.0.0
    let parse_numbers = fill(|s| preceded(obj_whitespace, float).parse(s), &mut normal);
    let mut parser = delimited(tag("vn"), parse_numbers, obj_blank_line);
    let (input, ()) = parser.parse(input)?;
    drop(parser); // release mut ref to normal

    let [i, j, k] = normal;
    let normal = ObjVn { i, j, k };
    Ok((input, normal))
}

pub fn obj_face_element(
    n_vertices: usize,
    n_textures: usize,
    n_normals: usize,
) -> impl Fn(&str) -> IResult<&str, ObjF>
where
{
    move |input: &str| {
        // issue with lifetime annotation on closures: see RFC 3216
        fn coerce<F>(closure: F) -> F
        where
            F: for<'a> Fn(&'a str) -> IResult<&'a str, usize>,
        {
            closure
        }

        // obj uses 1-based indexing, convert to 0-based
        fn checked_index(len: usize) -> impl FnMut(&str) -> IResult<&str, usize> {
            coerce(move |i: &str| -> IResult<&str, usize> {
                let mut parser = verify(parse_i32, |&v| v != 0 && v.unsigned_abs() as usize <= len);
                let (input, vertex) = parser.parse(i)?;
                let mag = vertex.unsigned_abs() as usize;
                let vertex = if vertex < 0 { len - mag } else { mag - 1 };
                Ok((input, vertex))
            })
        }

        let vxx = checked_index(n_vertices).map(|v| ObjVertexTriplet {
            index_vertex: v,
            index_texture: None,
            index_normal: None,
        });
        let vtx = separated_pair(
            checked_index(n_vertices),
            tag("/"),
            checked_index(n_textures),
        )
        .map(|(v, t)| ObjVertexTriplet {
            index_vertex: v,
            index_texture: Some(t),
            index_normal: None,
        });

        let vxn = separated_pair(
            checked_index(n_vertices),
            tag("//"),
            checked_index(n_normals),
        )
        .map(|(v, n)| ObjVertexTriplet {
            index_vertex: v,
            index_texture: None,
            index_normal: Some(n),
        });

        let vtn = separated_pair(
            checked_index(n_vertices),
            tag("/"),
            separated_pair(
                checked_index(n_textures),
                tag("/"),
                checked_index(n_normals),
            ),
        )
        .map(|(v, (t, n))| ObjVertexTriplet {
            index_vertex: v,
            index_texture: Some(t),
            index_normal: Some(n),
        });

        let parse_face = alt((
            many_m_n(3, 512, preceded(obj_whitespace, vxx)),
            many_m_n(3, 512, preceded(obj_whitespace, vtx)),
            many_m_n(3, 512, preceded(obj_whitespace, vxn)),
            many_m_n(3, 512, preceded(obj_whitespace, vtn)),
        ));
        let mut parser = delimited(tag("f"), parse_face, obj_blank_line);
        let (input, triplets) = parser.parse(input)?;

        let face = ObjF { triplets };
        Ok((input, face))
    }
}

pub fn obj_material_library(root_dir: PathBuf) -> impl Fn(&str) -> IResult<&str, ObjMtllib> {
    move |input: &str| {
        let parser = take_while1(|c: char| !c.is_whitespace());
        let parser = verify(parser, |s: &str| s.ends_with(".mtl"));
        let parser = preceded(obj_whitespace, parser);
        let mut parser = delimited(tag("mtllib"), parser, obj_blank_line);

        let (input, filepath) = parser.parse(input)?;
        let filepath = root_dir.join(filepath);
        let mtllib = ObjMtllib { filepath };
        Ok((input, mtllib))
    }
}

pub fn obj_material_use(input: &str) -> IResult<&str, ObjUsemtl> {
    let parser = take_while1(|c: char| !c.is_whitespace());
    let parser = preceded(obj_whitespace, parser);
    let mut parser = delimited(tag("usemtl"), parser, obj_blank_line);

    let (input, name) = parser.parse(input)?;
    let name = name.to_string();
    let usemtl = ObjUsemtl { name };
    Ok((input, usemtl))
}

pub fn parse_obj(path: &Path) -> Result<(ParsedObj, Vec<ParsedMtl>), ParseError> {
    let abs_path = path.canonicalize()?;
    assert!(abs_path.is_file());
    let abs_dir = abs_path.parent().unwrap();

    let file = File::open(path)?;
    let mut buffer = String::new();
    let mut reader = BufReader::new(file);

    let mut vertices = Vec::new();
    let mut texture_coords = Vec::new();
    let mut normals = Vec::new();
    let mut faces = Vec::new();
    let mut material_switches = Vec::new();

    let mut materials = Vec::new();

    let mut line_index = 0;
    while reader.read_line(&mut buffer)? > 0 {
        line_index += 1;
        loop {
            let mut parser = alt((
                obj_blank_line.map(|_| ObjLine::Empty),
                obj_geometric_vertex.map(ObjLine::GeometricVertex),
                obj_texture_coordinates.map(ObjLine::TextureCoordinates),
                obj_vertex_normal.map(ObjLine::VertexNormal),
                obj_face_element(vertices.len(), texture_coords.len(), normals.len())
                    .map(ObjLine::FaceIndex),
                obj_material_library(abs_dir.to_path_buf()).map(ObjLine::MtlLib),
                obj_material_use.map(ObjLine::UseMtl),
                obj_ignore.map(|_| ObjLine::Empty),
            ));
            let (rest, line) = match parser.parse(&buffer) {
                Ok((rest, line)) => (rest, line),
                Err(nom::Err::Incomplete(_)) => break,
                Err(_) => return Err(ParseError::WrongFormat(line_index)),
            };
            drop(parser);
            let offset = buffer.offset(rest);
            buffer = buffer.split_off(offset);

            match line {
                ObjLine::Empty => (),
                ObjLine::GeometricVertex(obj_v) => vertices.push(obj_v),
                ObjLine::TextureCoordinates(obj_vt) => texture_coords.push(obj_vt),
                ObjLine::VertexNormal(obj_vn) => normals.push(obj_vn),
                ObjLine::FaceIndex(obj_f) => faces.push(obj_f),
                ObjLine::MtlLib(ObjMtllib { filepath }) => materials.extend(parse_mtl(&filepath)?),
                ObjLine::UseMtl(ObjUsemtl { name }) => {
                    let material_index = materials
                        .iter()
                        .enumerate()
                        .find_map(|(i, mtl)| if mtl.name == name { Some(i) } else { None });
                    let Some(material_index) = material_index else {
                        return Err(ParseError::WrongFormat(line_index));
                    };

                    let first_face = faces.len(); // the next face that gets pushes
                    material_switches.push(ObjMaterialSwitch {
                        material_index,
                        first_face,
                    });
                }
            };
        }
    }

    let parsed_obj = ParsedObj {
        vertices,
        texture_coords,
        normals,
        faces,
        material_switches,
    };
    Ok((parsed_obj, materials))
}

#[allow(unused)]
pub struct MtlNs(pub f32);
#[allow(unused)]
pub struct MtlKa(pub f32, pub f32, pub f32);
#[allow(unused)]
pub struct MtlKd(pub f32, pub f32, pub f32);
#[expect(unused)]
pub struct MtlKs(pub f32, pub f32, pub f32);
#[expect(unused)]
pub struct MtlKe(pub f32, pub f32, pub f32);
#[expect(unused)]
pub struct MtlNi(pub f32);
#[expect(unused)]
pub struct MtlD(pub f32);
#[expect(unused)]
pub struct MtlIllum(pub u32);
pub struct MtlMapBump(pub PathBuf);
pub struct MtlMapKd(pub PathBuf);

pub struct ParsedMtl {
    pub name: String,
    pub ns: Option<MtlNs>,
    pub ka: Option<MtlKa>,
    pub kd: Option<MtlKd>,
    pub ks: Option<MtlKs>,
    pub ke: Option<MtlKe>,
    pub ni: Option<MtlNi>,
    pub d: Option<MtlD>,
    pub illum: Option<MtlIllum>,
    pub map_bump: Option<MtlMapBump>,
    pub map_kd: Option<MtlMapKd>,
}

impl ParsedMtl {
    fn new(name: String) -> ParsedMtl {
        ParsedMtl {
            name,
            ns: None,
            ka: None,
            kd: None,
            ks: None,
            ke: None,
            ni: None,
            d: None,
            illum: None,
            map_bump: None,
            map_kd: None,
        }
    }
}

enum MtlLine {
    Empty,
    New(String),
    Ns(MtlNs),
    Ka(MtlKa),
    Kd(MtlKd),
    Ks(MtlKs),
    Ke(MtlKe),
    Ni(MtlNi),
    D(MtlD),
    Illum(MtlIllum),
    MapBump(MtlMapBump),
    MapKd(MtlMapKd),
}

pub fn parse_mtl(path: &Path) -> Result<Vec<ParsedMtl>, ParseError> {
    let abs_path = path.canonicalize()?;
    assert!(abs_path.is_file());
    let abs_dir = abs_path.parent().unwrap();

    let file = File::open(&abs_path)?;
    let mut buffer = String::new();
    let mut reader = BufReader::new(file);

    let mut all_materials = vec![];
    let mut current_mtl = None;

    let mut line_index = 0;
    while reader.read_line(&mut buffer)? > 0 {
        line_index += 1;
        loop {
            let newmtl = take_while1(|c: char| !c.is_whitespace());
            let newmtl = preceded(obj_whitespace, newmtl);
            let newmtl = preceded(tag("newmtl"), newmtl).map(String::from);
            let ns = preceded(obj_whitespace, float);
            let ns = preceded(tag("Ns"), ns).map(MtlNs);
            let ka = (
                preceded(obj_whitespace, float),
                preceded(obj_whitespace, float),
                preceded(obj_whitespace, float),
            );
            let ka = preceded(tag("Ka"), ka).map(|(r, g, b)| MtlKa(r, g, b));
            let kd = (
                preceded(obj_whitespace, float),
                preceded(obj_whitespace, float),
                preceded(obj_whitespace, float),
            );
            let kd = preceded(tag("Kd"), kd).map(|(r, g, b)| MtlKd(r, g, b));
            let ks = (
                preceded(obj_whitespace, float),
                preceded(obj_whitespace, float),
                preceded(obj_whitespace, float),
            );
            let ks = preceded(tag("Ks"), ks).map(|(r, g, b)| MtlKs(r, g, b));
            let ke = (
                preceded(obj_whitespace, float),
                preceded(obj_whitespace, float),
                preceded(obj_whitespace, float),
            );
            let ke = preceded(tag("Ke"), ke).map(|(r, g, b)| MtlKe(r, g, b));
            let ni = preceded(obj_whitespace, float);
            let ni = preceded(tag("Ni"), ni).map(MtlNi);
            let d = preceded(obj_whitespace, float);
            let d = preceded(tag("d"), d).map(MtlD);
            let illum = preceded(obj_whitespace, parse_u32);
            let illum = preceded(tag("illum"), illum).map(MtlIllum);
            let map_bump = take_while1(|c: char| !c.is_whitespace());
            let map_bump = preceded(obj_whitespace, map_bump);
            let map_bump = preceded(tag("map_Bump"), map_bump)
                .map(|s| abs_dir.join(s))
                .map(MtlMapBump);
            let map_kd = take_while1(|c: char| !c.is_whitespace());
            let map_kd = preceded(obj_whitespace, map_kd);
            let map_kd = preceded(tag("map_Kd"), map_kd)
                .map(|s| abs_dir.join(s))
                .map(MtlMapKd);

            let mut parser = alt((
                obj_blank_line.map(|_| MtlLine::Empty),
                newmtl.map(MtlLine::New),
                ns.map(MtlLine::Ns),
                ka.map(MtlLine::Ka),
                kd.map(MtlLine::Kd),
                ks.map(MtlLine::Ks),
                ke.map(MtlLine::Ke),
                ni.map(MtlLine::Ni),
                d.map(MtlLine::D),
                illum.map(MtlLine::Illum),
                map_bump.map(MtlLine::MapBump),
                map_kd.map(MtlLine::MapKd),
            ));
            let (rest, line) = match parser.parse(&buffer) {
                Ok((rest, line)) => (rest, line),
                Err(nom::Err::Incomplete(_)) => break,
                Err(_) => return Err(ParseError::WrongFormat(line_index)),
            };
            drop(parser);
            let offset = buffer.offset(rest);
            buffer = buffer.split_off(offset);

            match line {
                MtlLine::Empty => (),
                MtlLine::New(name) => {
                    if let Some(last_mtl) = current_mtl {
                        all_materials.push(last_mtl);
                    }
                    current_mtl = Some(ParsedMtl::new(name))
                }
                MtlLine::Ns(mtl_ns) => match current_mtl.as_mut() {
                    Some(mtl) if mtl.ns.is_none() => mtl.ns = Some(mtl_ns),
                    _ => return Err(ParseError::WrongFormat(line_index)),
                },
                MtlLine::Ka(mtl_ka) => match current_mtl.as_mut() {
                    Some(mtl) if mtl.ka.is_none() => mtl.ka = Some(mtl_ka),
                    _ => return Err(ParseError::WrongFormat(line_index)),
                },
                MtlLine::Kd(mtl_kd) => match current_mtl.as_mut() {
                    Some(mtl) if mtl.kd.is_none() => mtl.kd = Some(mtl_kd),
                    _ => return Err(ParseError::WrongFormat(line_index)),
                },
                MtlLine::Ks(mtl_ks) => match current_mtl.as_mut() {
                    Some(mtl) if mtl.ks.is_none() => mtl.ks = Some(mtl_ks),
                    _ => return Err(ParseError::WrongFormat(line_index)),
                },
                MtlLine::Ke(mtl_ke) => match current_mtl.as_mut() {
                    Some(mtl) if mtl.ke.is_none() => mtl.ke = Some(mtl_ke),
                    _ => return Err(ParseError::WrongFormat(line_index)),
                },
                MtlLine::Ni(mtl_ni) => match current_mtl.as_mut() {
                    Some(mtl) if mtl.ni.is_none() => mtl.ni = Some(mtl_ni),
                    _ => return Err(ParseError::WrongFormat(line_index)),
                },
                MtlLine::D(mtl_d) => match current_mtl.as_mut() {
                    Some(mtl) if mtl.d.is_none() => mtl.d = Some(mtl_d),
                    _ => return Err(ParseError::WrongFormat(line_index)),
                },
                MtlLine::Illum(mtl_illum) => match current_mtl.as_mut() {
                    Some(mtl) if mtl.illum.is_none() => mtl.illum = Some(mtl_illum),
                    _ => return Err(ParseError::WrongFormat(line_index)),
                },
                MtlLine::MapBump(mtl_map_bump) => match current_mtl.as_mut() {
                    Some(mtl) if mtl.map_bump.is_none() => mtl.map_bump = Some(mtl_map_bump),
                    _ => return Err(ParseError::WrongFormat(line_index)),
                },
                MtlLine::MapKd(mtl_map_kd) => match current_mtl.as_mut() {
                    Some(mtl) if mtl.map_kd.is_none() => mtl.map_kd = Some(mtl_map_kd),
                    _ => return Err(ParseError::WrongFormat(line_index)),
                },
            };
        }
    }

    if let Some(mtl) = current_mtl {
        all_materials.push(mtl);
    }

    Ok(all_materials)
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::*;

    #[test]
    fn test_obj_blank_line() {
        let input = "# blank line\n# second blank line\n";
        let (rest, _) = obj_blank_line(input).unwrap();
        assert_eq!(rest, "# second blank line\n");
        let (rest, _) = obj_blank_line(rest).unwrap();
        assert_eq!(rest, "");
    }

    #[test]
    fn test_obj_geometric_vertex() {
        let input = "v 1.1   2.2   3.3 # this is a comment\n";
        let (rest, vec) = obj_geometric_vertex(input).unwrap();
        let expected = ObjV {
            x: 1.1,
            y: 2.2,
            z: 3.3,
            w: 1.0,
        };
        assert_eq!(rest, "");
        assert_eq!(vec, expected);
    }

    #[test]
    fn test_obj_texture_coordinates() {
        let input = "vt 1.0   0.2   0.3 # this is a comment\n";
        let (rest, vec) = obj_texture_coordinates(input).unwrap();
        let expected = ObjVt {
            u: 1.0,
            v: 0.2,
            w: 0.3,
        };
        assert_eq!(rest, "");
        assert_eq!(vec, expected);
    }

    #[test]
    fn test_obj_texture_coordinates_fail() {
        let input = "vt 1.1   2.2   3.3 # this is a comment\n";
        let out = obj_texture_coordinates(input);
        assert!(out.is_err());
    }

    #[test]
    fn test_obj_vertex_normal() {
        let input = "vn 1.1   2.2   3.3 # this is a comment\n";
        let (rest, vec) = obj_vertex_normal(input).unwrap();
        let expected = ObjVn {
            i: 1.1,
            j: 2.2,
            k: 3.3,
        };
        assert_eq!(rest, "");
        assert_eq!(vec, expected);
    }

    #[test]
    fn test_obj_face_element() {
        let input = "f -4/1/1   2/-3/2   3/3/-2 # this is a comment\n";
        let (rest, vec) = obj_face_element(4, 4, 4)(input).unwrap();
        let expected = ObjF {
            triplets: vec![
                ObjVertexTriplet {
                    index_vertex: 0,
                    index_texture: Some(0),
                    index_normal: Some(0),
                },
                ObjVertexTriplet {
                    index_vertex: 1,
                    index_texture: Some(1),
                    index_normal: Some(1),
                },
                ObjVertexTriplet {
                    index_vertex: 2,
                    index_texture: Some(2),
                    index_normal: Some(2),
                },
            ],
        };
        assert_eq!(rest, "");
        assert_eq!(vec, expected);
    }

    #[test]
    fn test_obj_face_element_fail() {
        let input = "f 1//1   2//2   3/3/3 # this is a comment\n";
        let out = obj_face_element(4, 4, 4)(input);
        assert!(out.is_err());
    }

    #[ignore]
    #[test]
    fn test_parse_obj() {
        let path = PathBuf::from("./resources/viking_room.obj");
        let (obj, _mtl) = parse_obj(&path).unwrap();
        assert!(obj.vertices.len() == 4675);
        assert!(obj.texture_coords.len() == 4675);
        assert!(obj.normals.len() == 2868);
        assert!(obj.faces.len() == 3828);
    }

    #[test]
    fn test_parse_mtl() {
        let path = PathBuf::from("./resources/cube/cube.mtl");
        let mtl = parse_mtl(&path).unwrap();
        let mtl = mtl.into_iter().next().unwrap();
        assert!(mtl.name == "Material.001");
        assert!(matches!(mtl.ns, Some(MtlNs(323.999994))));
        assert!(matches!(mtl.kd, Some(MtlKd(0.8, 0.8, 0.8))));
        let MtlMapBump(file) = mtl.map_bump.unwrap();
        assert!(file.file_name().unwrap() == "cube-normal.png");
    }
}

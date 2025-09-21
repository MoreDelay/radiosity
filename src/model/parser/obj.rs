use std::{
    fs::File,
    io::{BufRead, BufReader},
    path::{Path, PathBuf},
};

use nom::{
    IResult, Offset, Parser,
    branch::alt,
    bytes::streaming::{tag, take_until, take_while1},
    character::streaming::{i32 as parse_i32, line_ending, space1},
    combinator::opt,
    multi::{fill, many_m_n, many1, many1_count},
    number::streaming::float,
    sequence::{delimited, preceded},
};

use thiserror::Error;

use super::end_of_line;

const MAX_VERTICES_PER_FACE: usize = 16;

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

#[derive(Debug, PartialEq)]
pub struct ObjG {
    pub names: Vec<String>,
}

#[derive(Debug, PartialEq, Clone)]
pub struct ObjMtllib {
    pub name: String,
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

pub struct ObjGroupSwitch {
    pub names: Vec<String>,
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
    V(ObjV),
    Vt(Result<ObjVt, ObjSpecError>),
    Vn(ObjVn),
    F(Result<ObjF, ObjSpecError>),
    G(ObjG),
    MtlLib(ObjMtllib),
    UseMtl(ObjUsemtl),
}

#[derive(Error, Debug)]
pub enum ObjError {
    #[error("could not read file")]
    IoError(#[from] std::io::Error),
    #[error("OBJ file {0} is ill-formed at line {1}")]
    WrongFormat(PathBuf, usize),
    #[error("Obj file {0} on line {1}, value is out of spec")]
    OutOfSpec(PathBuf, usize, #[source] ObjSpecError),
    #[error("issue with MTL file")]
    FromMtl(#[source] super::mtl::MtlError),
}

#[derive(Error, Debug)]
pub enum ObjSpecError {
    #[error("invalid list index, must be non-zero and expect between -{0} and {0}, got {1}")]
    InvalidListIndex(usize, i32),
    #[error(
        "a face must have at least 3 vertices, supported at most {MAX_VERTICES_PER_FACE}, found {0}"
    )]
    BadFace(usize),
}

fn obj_whitespace(input: &str) -> IResult<&str, ()> {
    // allow for line continuation
    let mut parser = many1_count(alt((space1, preceded(tag("\\"), line_ending))));
    let (rest, _ws_count) = parser.parse(input)?;
    Ok((rest, ()))
}

fn obj_comment(input: &str) -> IResult<&str, &str> {
    let mut parser = preceded(tag("#"), take_until("\n"));
    let (rest, comment) = parser.parse(input)?;
    Ok((rest, comment.trim()))
}

fn obj_blank_line(input: &str) -> IResult<&str, ()> {
    let mut parser = delimited(opt(obj_whitespace), opt(obj_comment), end_of_line);
    let (rest, _comment) = parser.parse(input)?;
    Ok((rest, ()))
}

fn obj_ignore(input: &str) -> IResult<&str, ()> {
    let mut parser = preceded(alt((tag("o"), tag("s"))), take_until("\n"));
    let (rest, _) = parser.parse(input)?;
    Ok((rest, ()))
}

fn obj_v(input: &str) -> IResult<&str, ObjV> {
    let parser = (
        preceded(obj_whitespace, float),
        preceded(obj_whitespace, float),
        preceded(obj_whitespace, float),
        opt(preceded(obj_whitespace, float)),
    );
    let mut parser = delimited(tag("v"), parser, obj_blank_line);
    let (rest, (x, y, z, w)) = parser.parse(input)?;

    let w = w.unwrap_or(1.);
    let vertex = ObjV { x, y, z, w };
    Ok((rest, vertex))
}

fn obj_vt(input: &str) -> IResult<&str, Result<ObjVt, ObjSpecError>> {
    let parser = (
        preceded(obj_whitespace, float),
        opt(preceded(obj_whitespace, float)),
        opt(preceded(obj_whitespace, float)),
    );
    let mut parser = delimited(tag("vt"), parser, obj_blank_line);
    let (rest, (u, v, w)) = parser.parse(input)?;

    let v = v.unwrap_or(0.);
    let w = w.unwrap_or(0.);

    let vertex = ObjVt { u, v, w };

    Ok((rest, Ok(vertex)))
}

fn obj_vn(input: &str) -> IResult<&str, ObjVn> {
    let mut normal = [0.; 3];
    let parse_numbers = fill(preceded(obj_whitespace, float), &mut normal);
    let mut parser = delimited(tag("vn"), parse_numbers, obj_blank_line);
    let (rest, ()) = parser.parse(input)?;
    drop(parser); // release mut ref to normal

    let [i, j, k] = normal;
    let normal = ObjVn { i, j, k };
    Ok((rest, normal))
}

fn obj_list_index(len: usize) -> impl FnMut(&str) -> IResult<&str, Result<usize, ObjSpecError>> {
    move |i: &str| {
        let (rest, index) = parse_i32.parse(i)?;
        // obj uses 1-based indexing, convert here to 0-based
        let valid = index != 0 && index.unsigned_abs() as usize <= len;
        if !valid {
            return Ok((i, Err(ObjSpecError::InvalidListIndex(len, index))));
        }
        let abs = index.unsigned_abs() as usize;
        let index = if index < 0 { len - abs } else { abs - 1 };
        Ok((rest, Ok(index)))
    }
}

fn obj_face_vxx(
    n_vertices: usize,
) -> impl FnMut(&str) -> IResult<&str, Result<ObjVertexTriplet, ObjSpecError>> {
    move |i: &str| {
        obj_list_index(n_vertices)
            .map(|v| {
                Ok(ObjVertexTriplet {
                    index_vertex: v?,
                    index_texture: None,
                    index_normal: None,
                })
            })
            .parse(i)
    }
}

fn obj_face_vtx(
    n_vertices: usize,
    n_textures: usize,
) -> impl FnMut(&str) -> IResult<&str, Result<ObjVertexTriplet, ObjSpecError>> {
    move |i: &str| {
        (
            obj_list_index(n_vertices),
            tag("/"),
            obj_list_index(n_textures),
        )
            .map(|(v, _, t)| {
                Ok(ObjVertexTriplet {
                    index_vertex: v?,
                    index_texture: Some(t?),
                    index_normal: None,
                })
            })
            .parse(i)
    }
}

fn obj_face_vxn(
    n_vertices: usize,
    n_normals: usize,
) -> impl FnMut(&str) -> IResult<&str, Result<ObjVertexTriplet, ObjSpecError>> {
    move |i: &str| {
        (
            obj_list_index(n_vertices),
            tag("//"),
            obj_list_index(n_normals),
        )
            .map(|(v, _, n)| {
                Ok(ObjVertexTriplet {
                    index_vertex: v?,
                    index_texture: None,
                    index_normal: Some(n?),
                })
            })
            .parse(i)
    }
}

fn obj_face_vtn(
    n_vertices: usize,
    n_textures: usize,
    n_normals: usize,
) -> impl FnMut(&str) -> IResult<&str, Result<ObjVertexTriplet, ObjSpecError>> {
    move |i: &str| {
        (
            obj_list_index(n_vertices),
            tag("/"),
            obj_list_index(n_textures),
            tag("/"),
            obj_list_index(n_normals),
        )
            .map(|(v, _, t, _, n)| {
                Ok(ObjVertexTriplet {
                    index_vertex: v?,
                    index_texture: Some(t?),
                    index_normal: Some(n?),
                })
            })
            .parse(i)
    }
}

fn obj_f(
    n_v: usize,
    n_t: usize,
    n_n: usize,
) -> impl Fn(&str) -> IResult<&str, Result<ObjF, ObjSpecError>>
where
{
    const MAX: usize = MAX_VERTICES_PER_FACE;

    move |input: &str| {
        let counter = alt((
            preceded(
                (obj_whitespace, obj_face_vxx(n_v)),
                many1_count(preceded(obj_whitespace, obj_face_vxx(n_v))).map(|v| v + 1),
            ),
            preceded(
                (obj_whitespace, obj_face_vtx(n_v, n_t)),
                many1_count(preceded(obj_whitespace, obj_face_vtx(n_v, n_t))).map(|v| v + 1),
            ),
            preceded(
                (obj_whitespace, obj_face_vxn(n_v, n_n)),
                many1_count(preceded(obj_whitespace, obj_face_vxn(n_v, n_n))).map(|v| v + 1),
            ),
            preceded(
                (obj_whitespace, obj_face_vtn(n_v, n_t, n_n)),
                many1_count(preceded(obj_whitespace, obj_face_vtn(n_v, n_t, n_n))).map(|v| v + 1),
            ),
        ));
        let mut counter = delimited(tag("f"), counter, obj_blank_line);
        let (_, count) = counter.parse(input)?;
        let valid = (3..=MAX).contains(&count);
        if !valid {
            return Ok((input, Err(ObjSpecError::BadFace(count))));
        }
        let parser = alt((
            many_m_n(3, MAX, preceded(obj_whitespace, obj_face_vxx(n_v))),
            many_m_n(3, MAX, preceded(obj_whitespace, obj_face_vtx(n_v, n_t))),
            many_m_n(3, MAX, preceded(obj_whitespace, obj_face_vxn(n_v, n_n))),
            many_m_n(
                3,
                MAX,
                preceded(obj_whitespace, obj_face_vtn(n_v, n_t, n_n)),
            ),
        ));
        let mut parser = delimited(tag("f"), parser, obj_blank_line);
        let (rest, triplets) = parser.parse(input)?;

        let face = match triplets.into_iter().collect::<Result<_, _>>() {
            Ok(triplets) => ObjF { triplets },
            Err(e) => return Ok((input, Err(e))),
        };
        Ok((rest, Ok(face)))
    }
}

fn obj_mtllib(input: &str) -> IResult<&str, ObjMtllib> {
    let parser = take_while1(|c: char| !c.is_whitespace());
    let parser = preceded(obj_whitespace, parser);
    let mut parser = delimited(tag("mtllib"), parser, obj_blank_line);

    let (input, name) = parser.parse(input)?;
    let name = name.to_string();
    let mtllib = ObjMtllib { name };
    Ok((input, mtllib))
}

fn obj_g(input: &str) -> IResult<&str, ObjG> {
    let parser = take_while1(|c: char| !c.is_whitespace());
    let parser = preceded(obj_whitespace, parser);
    let parser = many1(parser);
    let mut parser = delimited(tag("g"), parser, obj_blank_line);

    let (input, names) = parser.parse(input)?;
    let names = names.into_iter().map(String::from).collect();
    let group = ObjG { names };
    Ok((input, group))
}

fn obj_usemtl(input: &str) -> IResult<&str, ObjUsemtl> {
    let parser = take_while1(|c: char| !c.is_whitespace());
    let parser = preceded(obj_whitespace, parser);
    let mut parser = delimited(tag("usemtl"), parser, obj_blank_line);

    let (input, name) = parser.parse(input)?;
    let name = name.to_string();
    let usemtl = ObjUsemtl { name };
    Ok((input, usemtl))
}

fn obj_line(
    n_vertices: usize,
    n_textures: usize,
    n_normals: usize,
) -> impl FnMut(&str) -> IResult<&str, ObjLine> {
    move |i: &str| {
        alt((
            obj_blank_line.map(|_| ObjLine::Empty),
            obj_v.map(ObjLine::V),
            obj_vt.map(ObjLine::Vt),
            obj_vn.map(ObjLine::Vn),
            obj_f(n_vertices, n_textures, n_normals).map(ObjLine::F),
            obj_mtllib.map(ObjLine::MtlLib),
            obj_g.map(ObjLine::G),
            obj_usemtl.map(ObjLine::UseMtl),
            obj_ignore.map(|_| ObjLine::Empty),
        ))
        .parse(i)
    }
}

pub fn load_obj(
    path: &Path,
    mtl_manager: &mut impl super::MtlManager,
) -> Result<ParsedObj, ObjError> {
    let file = File::open(path)?;
    let mut buffer = String::new();
    let mut reader = BufReader::new(file);

    let mut vertices = Vec::new();
    let mut texture_coords = Vec::new();
    let mut normals = Vec::new();
    let mut faces = Vec::new();
    let mut material_switches = Vec::new();
    let mut group_switches = Vec::new();

    // let mut materials = Vec::new();
    let mut material_names = Vec::new();

    let mut line_index = 0;
    while reader.read_line(&mut buffer)? > 0 {
        line_index += 1;
        loop {
            let n_v = vertices.len();
            let n_t = texture_coords.len();
            let n_n = normals.len();
            let (rest, line) = match obj_line(n_v, n_t, n_n).parse(&buffer) {
                Ok((rest, line)) => (rest, line),
                Err(nom::Err::Incomplete(_)) => break,
                _ => return Err(ObjError::WrongFormat(path.to_path_buf(), line_index)),
            };
            let offset = buffer.offset(rest);
            buffer = buffer.split_off(offset);

            let map_spec_err = |e| ObjError::OutOfSpec(path.to_path_buf(), line_index, e);

            match line {
                ObjLine::Empty => (),
                ObjLine::V(obj_v) => vertices.push(obj_v),
                ObjLine::Vt(obj_vt) => texture_coords.push(obj_vt.map_err(map_spec_err)?),
                ObjLine::Vn(obj_vn) => normals.push(obj_vn),
                ObjLine::F(obj_f) => faces.push(obj_f.map_err(map_spec_err)?),
                ObjLine::MtlLib(ObjMtllib { name }) => {
                    let found_names = mtl_manager
                        .request_mtl_load(&name)
                        .map_err(ObjError::FromMtl)?;
                    material_names.extend(found_names);
                }
                ObjLine::G(ObjG { names }) => {
                    let first_face = faces.len();
                    group_switches.push(ObjGroupSwitch { names, first_face });
                }
                ObjLine::UseMtl(ObjUsemtl { name }) => {
                    let valid_name = material_names.iter().any(|old| &name == old);
                    if !valid_name {
                        return Err(ObjError::WrongFormat(path.to_path_buf(), line_index));
                    }
                    let material_index = mtl_manager
                        .request_mtl_index(&name)
                        .expect("we must have stored this material in the manager before");

                    let first_face = faces.len(); // the next face that gets pushes
                    material_switches.push(ObjMaterialSwitch {
                        material_index,
                        first_face,
                    });
                }
            };
        }
    }

    if !normals.is_empty() {
        assert_eq!(vertices.len(), normals.len());
    }

    let parsed_obj = ParsedObj {
        vertices,
        texture_coords,
        normals,
        faces,
        material_switches,
    };
    Ok(parsed_obj)
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use crate::model::parser::SimpleMtlManager;

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
        let (rest, vec) = obj_v(input).unwrap();
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
        let (rest, vec) = obj_vt(input).unwrap();
        let vec = vec.unwrap();
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
        let out = obj_vt(input);
        assert!(out.is_err());
    }

    #[test]
    fn test_obj_vertex_normal() {
        let input = "vn 1.1   2.2   3.3 # this is a comment\n";
        let (rest, vec) = obj_vn(input).unwrap();
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
        let (rest, vec) = obj_f(4, 4, 4).parse(input).unwrap();
        let vec = vec.unwrap();
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
        let out = obj_f(4, 4, 4)(input);
        assert!(out.is_err());
    }

    #[ignore]
    #[test]
    fn test_parse_obj() {
        let path = PathBuf::from("./resources/cube/cube.obj");
        let mut mtl_manager = SimpleMtlManager::new(PathBuf::from("./resources/cube"));
        let obj = load_obj(&path, &mut mtl_manager).unwrap();
        assert!(obj.vertices.len() == 216);
        assert!(obj.texture_coords.len() == 277);
        assert!(obj.normals.len() == 216);
        assert!(obj.faces.len() == 218);
    }
}

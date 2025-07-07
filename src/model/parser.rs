use std::fs::File;
use std::io::BufReader;
use std::io::prelude::*;
use std::path::Path;

use nom::Offset;
use nom::branch::alt;
use nom::bytes::streaming::{tag, take_until};
use nom::character::streaming::{i64 as index, line_ending, space1};
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
    pub vertex: usize,
    pub texture: Option<usize>,
    pub normal: Option<usize>,
}

#[derive(Debug, PartialEq)]
pub struct ObjF {
    pub triplets: Vec<ObjVertexTriplet>,
}

pub struct ParsedObj {
    pub vertices: Vec<ObjV>,
    #[expect(dead_code)]
    pub texture_coords: Vec<ObjVt>,
    pub normals: Vec<ObjVn>,
    pub faces: Vec<ObjF>,
}

pub enum ObjLine {
    Empty,
    GeometricVertex(ObjV),
    TextureCoordinates(ObjVt),
    VertexNormal(ObjVn),
    FaceIndex(ObjF),
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
    let mut parser = preceded(
        alt((tag("usemtl"), tag("mtllib"), tag("o"), tag("s"))),
        take_until("\n"),
    );
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
    let mut normal = vec![0.; 3];
    // use closure to satisfy Fn requirement, gets changed to FnMut in nom v8.0.0
    let parse_numbers = fill(|s| preceded(obj_whitespace, float).parse(s), &mut normal);
    let mut parser = delimited(tag("vn"), parse_numbers, obj_blank_line);
    let (input, ()) = parser.parse(input)?;
    drop(parser); // release mut ref to normal

    let normal = if let [i, j, k] = normal[..] {
        ObjVn { i, j, k }
    } else {
        unreachable!()
    };
    Ok((input, normal))
}

pub fn obj_face_element(
    n_vertices: usize,
    n_textures: usize,
    n_normals: usize,
) -> impl FnMut(&str) -> IResult<&str, ObjF>
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
                let mut parser = verify(index, |&v| v != 0 && v.unsigned_abs() as usize <= len);
                let (input, vertex) = parser.parse(i)?;
                let mag = vertex.unsigned_abs() as usize;
                let vertex = if vertex < 0 { len - mag } else { mag - 1 };
                Ok((input, vertex))
            })
        }

        let vxx = checked_index(n_vertices).map(|v| ObjVertexTriplet {
            vertex: v,
            texture: None,
            normal: None,
        });
        let vtx = separated_pair(
            checked_index(n_vertices),
            tag("/"),
            checked_index(n_textures),
        )
        .map(|(v, t)| ObjVertexTriplet {
            vertex: v,
            texture: Some(t),
            normal: None,
        });

        let vxn = separated_pair(
            checked_index(n_vertices),
            tag("//"),
            checked_index(n_normals),
        )
        .map(|(v, n)| ObjVertexTriplet {
            vertex: v,
            texture: None,
            normal: Some(n),
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
            vertex: v,
            texture: Some(t),
            normal: Some(n),
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

pub fn parse_obj(path: &Path) -> Result<ParsedObj, ParseError> {
    let file = File::open(path)?;
    let mut buffer = String::new();
    let mut reader = BufReader::new(file);

    let mut vertices = Vec::new();
    let mut textures = Vec::new();
    let mut normals = Vec::new();
    let mut faces = Vec::new();

    let mut line_index = 0;
    while reader.read_line(&mut buffer)? > 0 {
        line_index += 1;
        loop {
            let mut parser = alt((
                obj_blank_line.map(|_| ObjLine::Empty),
                obj_geometric_vertex.map(ObjLine::GeometricVertex),
                obj_texture_coordinates.map(ObjLine::TextureCoordinates),
                obj_vertex_normal.map(ObjLine::VertexNormal),
                obj_face_element(vertices.len(), textures.len(), normals.len())
                    .map(ObjLine::FaceIndex),
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
                ObjLine::TextureCoordinates(obj_vt) => textures.push(obj_vt),
                ObjLine::VertexNormal(obj_vn) => normals.push(obj_vn),
                ObjLine::FaceIndex(obj_f) => faces.push(obj_f),
            };
        }
    }

    Ok(ParsedObj {
        vertices,
        texture_coords: textures,
        normals,
        faces,
    })
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
                    vertex: 0,
                    texture: Some(0),
                    normal: Some(0),
                },
                ObjVertexTriplet {
                    vertex: 1,
                    texture: Some(1),
                    normal: Some(1),
                },
                ObjVertexTriplet {
                    vertex: 2,
                    texture: Some(2),
                    normal: Some(2),
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
        let obj = parse_obj(&path).unwrap();
        assert!(obj.vertices.len() == 4675);
        assert!(obj.texture_coords.len() == 4675);
        assert!(obj.normals.len() == 2868);
        assert!(obj.faces.len() == 3828);
    }
}

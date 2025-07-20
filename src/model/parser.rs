use std::path::PathBuf;

use nom::branch::alt;
use nom::character::streaming::line_ending;
use nom::combinator::eof;
use nom::{IResult, Parser};

pub mod mtl;
pub mod obj;

pub trait MtlManager {
    /// Request to load the provided MTL library. The argument is just the provided name within the
    /// file which typically is concatenated with the parent directory of the original obj file.
    fn request_mtl_load(&mut self, name: &str) -> Result<Vec<String>, mtl::MtlError>;

    /// Given the name of a material, this call should return the index to the material that
    /// uniquely identifies the corresponding material for this manager.
    fn request_mtl_index(&self, name: &str) -> Option<usize>;
}

pub struct SimpleMtlManager {
    root_dir: PathBuf,
    materials: Vec<mtl::ParsedMtl>,
    names: Vec<String>,
}

#[expect(unused)]
impl SimpleMtlManager {
    pub fn new(root_dir: PathBuf) -> Self {
        Self {
            root_dir,
            materials: Vec::new(),
            names: Vec::new(),
        }
    }

    pub fn len(&self) -> usize {
        self.materials.len()
    }

    pub fn get(&self, index: usize) -> Option<&mtl::ParsedMtl> {
        self.materials.get(index)
    }
}

impl MtlManager for SimpleMtlManager {
    fn request_mtl_load(&mut self, name: &str) -> Result<Vec<String>, mtl::MtlError> {
        let path = self.root_dir.join(name);
        let loaded_mtls = mtl::parse_mtl(&path)?;
        let mut mtls_in_this_lib = Vec::new();

        for mtl in loaded_mtls {
            mtls_in_this_lib.push(mtl.name.clone());

            let already_loaded = self.names.iter().any(|n| n == &mtl.name);
            if already_loaded {
                continue;
            }
            self.names.push(mtl.name.clone());
            self.materials.push(mtl);
        }

        Ok(mtls_in_this_lib)
    }

    fn request_mtl_index(&self, name: &str) -> Option<usize> {
        self.names
            .iter()
            .enumerate()
            .find_map(|(i, n)| if n == name { Some(i) } else { None })
    }
}

fn end_of_line(input: &str) -> IResult<&str, ()> {
    let mut parser = alt((line_ending, eof));
    let (input, _) = parser.parse(input)?;
    Ok((input, ()))
}

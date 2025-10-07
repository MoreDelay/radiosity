#[derive(Debug, Copy, Clone)]
pub struct Color {
    pub r: f32,
    pub g: f32,
    pub b: f32,
}

impl From<Color> for [f32; 3] {
    fn from(val: Color) -> Self {
        let Color { r, g, b, .. } = val;
        [r, g, b]
    }
}

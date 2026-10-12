// SPDX-License-Identifier: MIT

//! Film grain, which the AV1 decoder adds to the pictures it outputs: the
//! driver gives it the scaling functions and the grain templates (section
//! 7.18.3 of the AV1 specification), the decoder takes the blocks of grain it
//! adds from them itself.
//!
//! The decoder takes them as NVIDIA's driver gives them
//! (`AV1FilmGrainMemory`): the 3 scaling functions, of 256 entries, then the
//! grain templates, cropped to 64x64 luma and 32x32 chroma samples (4:2:0).

use crate::ffi::{av1_fgs_cfg_t, StdVideoAV1FilmGrain};

use super::av1_gaussian::GAUSSIAN_SEQUENCE;

/// Bytes of the film grain buffer.
pub const FILM_GRAIN_B: usize = 3 * 256 + 2 * (64 * 64 + 2 * 32 * 32);

const SCALING_Y: usize = 0;
const SCALING_CB: usize = 256;
const SCALING_CR: usize = 512;
const GRAIN_Y: usize = 768;
const GRAIN_CB: usize = GRAIN_Y + 2 * 64 * 64;
const GRAIN_CR: usize = GRAIN_CB + 2 * 32 * 32;

/// The luma grain template, and the chroma ones of 4:2:0.
const LUMA_W: usize = 82;
const LUMA_H: usize = 73;
const CHROMA_W: usize = 44;
const CHROMA_H: usize = 38;

/// Round2() of the AV1 specification.
fn round2(x: i32, n: u32) -> i32 {
    if n == 0 {
        x
    } else {
        (x + (1 << (n - 1))) >> n
    }
}

/// The pseudo-random generator of the specification (get_random_number()).
struct Random(u16);

impl Random {
    fn next(&mut self, bits: u32) -> usize {
        let r = self.0;
        let bit = (r ^ (r >> 1) ^ (r >> 3) ^ (r >> 12)) & 1;
        self.0 = (r >> 1) | (bit << 15);
        usize::from(self.0 >> (16 - bits)) & ((1 << bits) - 1)
    }
}

/// The scaling function of `points` (x, y) (section 7.18.3.4), piecewise
/// linear between them, as libaom computes it.
fn scaling_function(points: &[(u8, u8)]) -> [u8; 256] {
    let mut lut = [0u8; 256];
    let (Some(&(x0, y0)), Some(&(xn, yn))) = (points.first(), points.last())
    else {
        return lut;
    };
    lut[..usize::from(x0)].fill(y0);
    for p in points.windows(2) {
        let (xa, ya) = (i32::from(p[0].0), i32::from(p[0].1));
        let (xb, yb) = (i32::from(p[1].0), i32::from(p[1].1));
        let dx = xb - xa;
        let delta = (yb - ya) * ((65536 + (dx >> 1)) / dx);
        for x in 0..dx {
            let v = ya + ((x * delta + 32768) >> 16);
            lut[(xa + x) as usize] = v as u8;
        }
    }
    lut[usize::from(xn)..].fill(yn);
    lut
}

/// Writes the grain templates `grain` of `w` x `h` samples, cropped from
/// (`x0`, `y0`) to `size` x `size`, to `dst`.
fn crop(grain: &[i32], w: usize, x0: usize, size: usize, dst: &mut [u8]) {
    for y in 0..size {
        for x in 0..size {
            let v = grain[(x0 + y) * w + x0 + x] as i16;
            let i = 2 * (y * size + x);
            dst[i..i + 2].copy_from_slice(&v.to_le_bytes());
        }
    }
}

/// Sets the film grain of `fg` up for the decoder: its parameters in `cfg`,
/// its scaling functions and grain templates in `mem`, of
/// [`FILM_GRAIN_B`] bytes, for a picture of `bit_depth` bits.
pub fn film_grain(
    fg: &StdVideoAV1FilmGrain,
    bit_depth: u8,
    cfg: &mut av1_fgs_cfg_t,
    mem: &mut [u8],
) {
    let flags = &fg.flags;
    let from_luma = flags.chroma_scaling_from_luma() != 0;
    let num_y = usize::from(fg.num_y_points);
    let num_cb = usize::from(fg.num_cb_points);
    let num_cr = usize::from(fg.num_cr_points);

    cfg.set_apply_grain(1);
    cfg.set_overlap_flag(flags.overlap_flag() as u16);
    cfg.set_clip_to_restricted_range(flags.clip_to_restricted_range() as u16);
    cfg.set_chroma_scaling_from_luma(from_luma.into());
    cfg.set_num_y_points_b((num_y > 0).into());
    cfg.set_num_cb_points_b((num_cb > 0).into());
    cfg.set_num_cr_points_b((num_cr > 0).into());
    cfg.set_scaling_shift((fg.grain_scaling_minus_8 + 8).into());
    cfg.sw_random_seed = fg.grain_seed;
    if from_luma {
        cfg.cb_luma_mult = 64;
        cfg.cr_luma_mult = 64;
    } else {
        cfg.cb_mult = (i32::from(fg.cb_mult) - 128) as i8;
        cfg.cb_luma_mult = (i32::from(fg.cb_luma_mult) - 128) as i8;
        cfg.cb_offset = (i32::from(fg.cb_offset) - 256) as i16;
        cfg.cr_mult = (i32::from(fg.cr_mult) - 128) as i8;
        cfg.cr_luma_mult = (i32::from(fg.cr_luma_mult) - 128) as i8;
        cfg.cr_offset = (i32::from(fg.cr_offset) - 256) as i16;
    }

    // The scaling functions.
    let points = |xs: &[u8], ys: &[u8], n: usize| -> Vec<(u8, u8)> {
        xs.iter().zip(ys).take(n).map(|(&x, &y)| (x, y)).collect()
    };
    let y_lut = scaling_function(&points(
        &fg.point_y_value,
        &fg.point_y_scaling,
        num_y,
    ));
    let (cb_lut, cr_lut) = if from_luma {
        (y_lut, y_lut)
    } else {
        (
            scaling_function(&points(
                &fg.point_cb_value,
                &fg.point_cb_scaling,
                num_cb,
            )),
            scaling_function(&points(
                &fg.point_cr_value,
                &fg.point_cr_scaling,
                num_cr,
            )),
        )
    };
    mem[SCALING_Y..SCALING_Y + 256].copy_from_slice(&y_lut);
    mem[SCALING_CB..SCALING_CB + 256].copy_from_slice(&cb_lut);
    mem[SCALING_CR..SCALING_CR + 256].copy_from_slice(&cr_lut);

    // The grain templates (section 7.18.3.3).
    let bd = u32::from(bit_depth);
    let grain_min = -(128 << (bd - 8));
    let grain_max = (128 << (bd - 8)) - 1;
    let shift = 12 - bd + u32::from(fg.grain_scale_shift);
    let ar_shift = u32::from(fg.ar_coeff_shift_minus_6) + 6;
    let lag = i32::from(fg.ar_coeff_lag);

    let gaussian = |random: &mut Random| {
        round2(i32::from(GAUSSIAN_SEQUENCE[random.next(11)]), shift)
    };

    let mut luma = vec![0i32; LUMA_W * LUMA_H];
    if num_y > 0 {
        let mut random = Random(fg.grain_seed);
        for g in luma.iter_mut() {
            *g = gaussian(&mut random);
        }
        for y in 3..LUMA_H {
            for x in 3..LUMA_W - 3 {
                let mut sum = 0;
                let mut pos = 0;
                'rows: for dy in -lag..=0 {
                    for dx in -lag..=lag {
                        if dy == 0 && dx == 0 {
                            break 'rows;
                        }
                        let c =
                            i32::from(fg.ar_coeffs_y_plus_128[pos] as u8) - 128;
                        let i = (y as i32 + dy) as usize * LUMA_W
                            + (x as i32 + dx) as usize;
                        sum += luma[i] * c;
                        pos += 1;
                    }
                }
                let i = y * LUMA_W + x;
                luma[i] = (luma[i] + round2(sum, ar_shift))
                    .clamp(grain_min, grain_max);
            }
        }
    }

    let chroma_grain = |seed: u16, n: usize, coeffs: &[i8]| -> Vec<i32> {
        let mut grain = vec![0i32; CHROMA_W * CHROMA_H];
        if n == 0 && !from_luma {
            return grain;
        }
        let mut random = Random(seed);
        for g in grain.iter_mut() {
            *g = gaussian(&mut random);
        }
        for y in 3..CHROMA_H {
            for x in 3..CHROMA_W - 3 {
                let mut sum = 0;
                let mut pos = 0;
                'rows: for dy in -lag..=0 {
                    for dx in -lag..=lag {
                        let c = i32::from(coeffs[pos] as u8) - 128;
                        if dy == 0 && dx == 0 {
                            // The co-located luma grain, averaged over the
                            // 2x2 luma samples of the chroma sample.
                            if num_y > 0 {
                                let ly = ((y - 3) << 1) + 3;
                                let lx = ((x - 3) << 1) + 3;
                                let l = luma[ly * LUMA_W + lx]
                                    + luma[ly * LUMA_W + lx + 1]
                                    + luma[(ly + 1) * LUMA_W + lx]
                                    + luma[(ly + 1) * LUMA_W + lx + 1];
                                sum += round2(l, 2) * c;
                            }
                            break 'rows;
                        }
                        let i = (y as i32 + dy) as usize * CHROMA_W
                            + (x as i32 + dx) as usize;
                        sum += grain[i] * c;
                        pos += 1;
                    }
                }
                let i = y * CHROMA_W + x;
                grain[i] = (grain[i] + round2(sum, ar_shift))
                    .clamp(grain_min, grain_max);
            }
        }
        grain
    };
    let cb =
        chroma_grain(fg.grain_seed ^ 0xb524, num_cb, &fg.ar_coeffs_cb_plus_128);
    let cr =
        chroma_grain(fg.grain_seed ^ 0x49d8, num_cr, &fg.ar_coeffs_cr_plus_128);

    crop(&luma, LUMA_W, 9, 64, &mut mem[GRAIN_Y..GRAIN_CB]);
    crop(&cb, CHROMA_W, 6, 32, &mut mem[GRAIN_CB..GRAIN_CR]);
    crop(&cr, CHROMA_W, 6, 32, &mut mem[GRAIN_CR..FILM_GRAIN_B]);
}

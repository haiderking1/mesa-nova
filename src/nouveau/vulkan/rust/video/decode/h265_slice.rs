// SPDX-License-Identifier: MIT

//! What NVK reads of the slice segment headers of H.265 pictures: how many
//! bits their long-term reference picture set takes, which the decoder needs
//! to skip them, and which Vulkan doesn't give.

/// The bytes of a NAL unit's payload NVK reads at most: the slice segment
/// header up to its long-term reference picture set is far shorter.
const MAX_HEADER_B: usize = 1024;

/// A reader of the bits of a NAL unit's RBSP.
struct BitReader {
    rbsp: Vec<u8>,
    pos: usize,
}

impl BitReader {
    /// The RBSP of the NAL unit payload `nal`, after its header: without the
    /// emulation prevention bytes, which follow two zero bytes.
    fn new(nal: &[u8]) -> Self {
        let mut rbsp = Vec::with_capacity(nal.len().min(MAX_HEADER_B));
        let mut zeros = 0;
        for &b in nal.iter().take(MAX_HEADER_B) {
            if zeros >= 2 && b == 3 {
                zeros = 0;
                continue;
            }
            zeros = if b == 0 { zeros + 1 } else { 0 };
            rbsp.push(b);
        }
        BitReader { rbsp, pos: 0 }
    }

    fn u(&mut self, bits: u32) -> Option<u32> {
        let mut v = 0u32;
        for _ in 0..bits {
            let byte = *self.rbsp.get(self.pos / 8)?;
            let bit = (byte >> (7 - self.pos % 8)) & 1;
            v = (v << 1) | u32::from(bit);
            self.pos += 1;
        }
        Some(v)
    }

    fn flag(&mut self) -> Option<bool> {
        self.u(1).map(|b| b != 0)
    }

    fn skip(&mut self, bits: u32) -> Option<()> {
        let pos = self.pos + bits as usize;
        (pos <= self.rbsp.len() * 8).then(|| self.pos = pos)
    }

    /// An Exp-Golomb coded unsigned value, ue(v).
    fn ue(&mut self) -> Option<u32> {
        let mut zeros = 0;
        while !self.flag()? {
            zeros += 1;
            if zeros > 31 {
                return None;
            }
        }
        Some((1u32 << zeros) - 1 + self.u(zeros)?)
    }
}

/// `Ceil(Log2(n))`: the bits of an index of `n` values.
pub(super) fn index_bits(n: u32) -> u32 {
    n.next_power_of_two().trailing_zeros()
}

/// What parsing a slice segment header up to its long-term reference
/// picture set takes of the SPS, the PPS and the picture.
pub(super) struct SliceHeaderInfo {
    pub num_extra_slice_header_bits: u32,
    pub output_flag_present: bool,
    pub separate_colour_plane: bool,
    pub log2_max_pic_order_cnt_lsb: u32,
    pub num_short_term_ref_pic_sets: u32,
    /// The bits of the short-term reference picture set the slice segment
    /// header codes itself, which Vulkan gives.
    pub short_term_ref_pic_set_bits: u32,
    pub long_term_ref_pics_present: bool,
    pub num_long_term_ref_pics_sps: u32,
}

/// The bits the long-term reference picture set of the slice segment header
/// of the NAL unit `nal`, from its start code prefix on, takes (7.3.6.1 of
/// the H.265 specification); None if it isn't the header of the first slice
/// segment of a picture, or if `nal` ends before.
pub(super) fn long_term_ref_pics_bits(
    nal: &[u8],
    info: &SliceHeaderInfo,
) -> Option<u32> {
    // The start code prefix, of a zero byte or more, and 1.
    let start = nal.iter().position(|&b| b != 0)?;
    if start < 2 || nal[start] != 1 || nal.len() < start + 3 {
        return None;
    }
    let nal_unit_type = (nal[start + 1] >> 1) & 0x3f;
    // Only the slice segments of pictures, VCL NAL units, have the header.
    if nal_unit_type >= 32 {
        return None;
    }
    let idr = nal_unit_type == 19 || nal_unit_type == 20;
    let irap = (16..=23).contains(&nal_unit_type);

    let mut r = BitReader::new(&nal[start + 3..]);
    let first_slice_segment_in_pic = r.flag()?;
    if irap {
        // no_output_of_prior_pics_flag
        r.skip(1)?;
    }
    // slice_pic_parameter_set_id
    r.ue()?;
    if !first_slice_segment_in_pic {
        // Only the first slice segment of a picture is looked at, which isn't
        // a dependent one.
        return None;
    }
    // slice_reserved_flag[], slice_type
    r.skip(info.num_extra_slice_header_bits)?;
    r.ue()?;
    if info.output_flag_present {
        r.skip(1)?;
    }
    if info.separate_colour_plane {
        r.skip(2)?;
    }
    if idr || !info.long_term_ref_pics_present {
        return Some(0);
    }

    // slice_pic_order_cnt_lsb, and the short-term reference picture set,
    // coded or an index of one of the SPS.
    r.skip(info.log2_max_pic_order_cnt_lsb)?;
    if !r.flag()? {
        r.skip(info.short_term_ref_pic_set_bits)?;
    } else {
        r.skip(index_bits(info.num_short_term_ref_pic_sets))?;
    }

    let start = r.pos;
    let num_long_term_sps = if info.num_long_term_ref_pics_sps > 0 {
        r.ue()?
    } else {
        0
    };
    let num_long_term_pics = r.ue()?;
    for i in 0..num_long_term_sps.checked_add(num_long_term_pics)? {
        if i < num_long_term_sps {
            // lt_idx_sps
            r.skip(index_bits(info.num_long_term_ref_pics_sps))?;
        } else {
            // poc_lsb_lt, used_by_curr_pic_lt_flag
            r.skip(info.log2_max_pic_order_cnt_lsb + 1)?;
        }
        // delta_poc_msb_present_flag, delta_poc_msb_cycle_lt
        if r.flag()? {
            r.ue()?;
        }
    }
    Some((r.pos - start).try_into().unwrap())
}

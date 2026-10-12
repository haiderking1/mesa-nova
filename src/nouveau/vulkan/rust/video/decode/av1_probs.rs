// SPDX-License-Identifier: MIT

//! The probability table of the AV1 decoder: the CDFs it decodes a frame with,
//! which it adapts and writes back once the frame is decoded.
//!
//! The decoder takes CDFs as inverse CDFs (32768 minus each value but the
//! last), 16 bits each, without libaom's counters. A table keeps each of its
//! CDFs at the same stride, mostly as many values as its CDFs have; the
//! decoder puts tables of small CDFs in the space larger CDFs leave in theirs.
//! NVIDIA's driver lays the default CDFs out so for the decoder (found by
//! comparing what its CUVID library builds with libaom's tables).

use super::av1_default_cdfs::*;

/// Size of a probability table in bytes.
pub const PROBS_B: usize = 0x2fc0;

/// A nesting of arrays of CDFs.
trait Cdfs {
    /// Calls `f` with each CDF, in the order of the arrays.
    fn for_each_cdf(&self, f: &mut dyn FnMut(&[u16]));
}

impl<const N: usize> Cdfs for [u16; N] {
    fn for_each_cdf(&self, f: &mut dyn FnMut(&[u16])) {
        f(self)
    }
}

impl<T: Cdfs, const N: usize> Cdfs for [T; N] {
    fn for_each_cdf(&self, f: &mut dyn FnMut(&[u16])) {
        for t in self {
            t.for_each_cdf(f);
        }
    }
}

struct Table<'a>(&'a mut [u8; PROBS_B]);

impl Table<'_> {
    /// Writes the CDFs of `cdfs` from byte `offset` on, each `stride` 16-bit
    /// values after the previous one.
    fn put(&mut self, offset: usize, stride: usize, cdfs: &impl Cdfs) {
        let mut at = offset;
        cdfs.for_each_cdf(&mut |cdf| {
            assert!(cdf.len() <= stride);
            for (i, v) in cdf.iter().enumerate() {
                let p = at + 2 * i;
                self.0[p..p + 2].copy_from_slice(&(32768 - v).to_le_bytes());
            }
            at += 2 * stride;
        });
    }
}

/// The index of the coefficient CDFs of a frame of base_q_idx `q`
/// (get_q_ctx() in libaom).
pub(super) fn q_ctx(q: u8) -> usize {
    match q {
        0..=20 => 0,
        21..=60 => 1,
        61..=120 => 2,
        _ => 3,
    }
}

/// Writes the default probability table of the frames of coefficient CDF
/// context `q` (see [`q_ctx`]), as the decoder lays it out, to `table`.
pub fn default_probs(q: usize, table: &mut [u8; PROBS_B]) {
    table.fill(0);
    let mut t = Table(table);

    t.put(0x0000, 3, &DEFAULT_PARTITION_8X8_CDF);
    t.put(0x0018, 1, &DEFAULT_WIENER_RESTORE_CDF);
    t.put(0x001a, 1, &DEFAULT_SGRPROJ_RESTORE_CDF);
    t.put(0x001c, 2, &DEFAULT_SWITCHABLE_RESTORE_CDF);
    t.put(0x0020, 16, &DEFAULT_PARTITION_CDF);
    // After the CDFs of 64x64 blocks.
    t.put(0x0132, 16, &DEFAULT_PARTITION_128X128_CDF);
    t.put(0x01a0, 12, &DEFAULT_KF_Y_MODE_CDF);
    t.put(0x03f8, 1, &DEFAULT_SEGMENT_PRED_CDF);
    t.put(0x03fe, 7, &DEFAULT_SPATIAL_PRED_SEG_TREE_CDF);
    t.put(0x0428, 1, &DEFAULT_SKIP_TXFM_CDFS);
    t.put(0x042e, 3, &DEFAULT_DELTA_Q_CDF);
    t.put(0x0434, 3, &DEFAULT_DELTA_LF_MULTI_CDF);
    t.put(0x044c, 3, &DEFAULT_DELTA_LF_CDF);
    t.put(0x0452, 1, &DEFAULT_SKIP_MODE_CDFS);
    t.put(0x0458, 1, &DEFAULT_TXFM_PARTITION_CDF);
    t.put(0x0484, 2, &DEFAULT_TX_SIZE_8X8_CDF);
    t.put(0x0490, 2, &DEFAULT_TX_SIZE_CDF);
    t.put(0x04b4, 12, &DEFAULT_IF_Y_MODE_CDF);
    t.put(0x0514, 13, &DEFAULT_UV_MODE_CFL_NOT_ALLOWED_CDF);
    t.put(0x0666, 13, &DEFAULT_UV_MODE_CFL_ALLOWED_CDF);
    t.put(0x07b8, 1, &DEFAULT_INTRA_INTER_CDF);
    t.put(0x07c0, 1, &DEFAULT_COMP_INTER_CDF);
    t.put(0x07ca, 1, &DEFAULT_SINGLE_REF_CDF);
    t.put(0x07ee, 1, &DEFAULT_COMP_REF_TYPE_CDF);
    t.put(0x07f8, 1, &DEFAULT_UNI_COMP_REF_CDF);
    t.put(0x080a, 1, &DEFAULT_COMP_REF_CDF);
    t.put(0x081c, 1, &DEFAULT_COMP_BWDREF_CDF);
    t.put(0x0828, 1, &DEFAULT_NEWMV_CDF);
    t.put(0x0834, 1, &DEFAULT_ZEROMV_CDF);
    t.put(0x0838, 1, &DEFAULT_REFMV_CDF);
    t.put(0x084a, 1, &DEFAULT_DRL_CDF);
    t.put(0x0850, 2, &DEFAULT_SWITCHABLE_INTERP_CDF);

    t.put(0x0890, 3, &DEFAULT_MV_JOINTS_CDF);
    t.put(0x0896, 1, &DEFAULT_MV_SIGN_CDF);
    t.put(0x089a, 10, &DEFAULT_MV_CLASSES_CDF);
    t.put(0x08c2, 3, &DEFAULT_MV_CLASS0_FP_CDF);
    t.put(0x08da, 3, &DEFAULT_MV_FP_CDF);
    t.put(0x08e6, 1, &DEFAULT_MV_CLASS0_HP_CDF);
    t.put(0x08ea, 1, &DEFAULT_MV_HP_CDF);
    t.put(0x08ee, 1, &DEFAULT_MV_CLASS0_CDF);
    t.put(0x08f2, 1, &DEFAULT_MV_BITS_CDF);

    t.put(0x091a, 1, &DEFAULT_OBMC_CDF);
    t.put(0x0946, 2, &DEFAULT_MOTION_MODE_CDF);
    t.put(0x099e, 7, &DEFAULT_INTER_COMPOUND_MODE_CDF);
    t.put(0x0a70, 1, &DEFAULT_COMPOUND_TYPE_CDF);
    t.put(0x0a9c, 1, &DEFAULT_INTERINTRA_CDF);
    t.put(0x0aa4, 3, &DEFAULT_INTERINTRA_MODE_CDF);
    t.put(0x0abc, 1, &DEFAULT_WEDGE_INTERINTRA_CDF);
    t.put(0x0ae8, 15, &DEFAULT_WEDGE_IDX_CDF);
    t.put(0x0d7c, 1, &DEFAULT_PALETTE_Y_MODE_CDF);
    t.put(0x0da6, 1, &DEFAULT_PALETTE_UV_MODE_CDF);
    t.put(0x0daa, 6, &DEFAULT_PALETTE_Y_SIZE_CDF);
    t.put(0x0dfe, 6, &DEFAULT_PALETTE_UV_SIZE_CDF);
    t.put(0x0e52, 7, &DEFAULT_CFL_SIGN_CDF);
    t.put(0x0e60, 15, &DEFAULT_CFL_ALPHA_CDF);
    t.put(0x0f14, 1, &DEFAULT_INTRABC_CDF);
    t.put(0x0f16, 6, &DEFAULT_ANGLE_DELTA_CDF);
    t.put(0x0f76, 4, &DEFAULT_FILTER_INTRA_MODE_CDF);
    t.put(0x0f7e, 1, &DEFAULT_FILTER_INTRA_CDFS);
    t.put(0x0faa, 1, &DEFAULT_COMP_GROUP_IDX_CDFS);
    t.put(0x0fb8, 1, &DEFAULT_COMPOUND_IDX_CDFS);

    // The color index CDFs of palettes of n and 10 - n colors share 8
    // values, those of 5 colors take 4.
    for (base, cdfs) in [
        (
            0x0fe0,
            (
                &DEFAULT_PALETTE_Y_COLOR_INDEX_2_CDF,
                &DEFAULT_PALETTE_Y_COLOR_INDEX_3_CDF,
                &DEFAULT_PALETTE_Y_COLOR_INDEX_4_CDF,
                &DEFAULT_PALETTE_Y_COLOR_INDEX_5_CDF,
                &DEFAULT_PALETTE_Y_COLOR_INDEX_6_CDF,
                &DEFAULT_PALETTE_Y_COLOR_INDEX_7_CDF,
                &DEFAULT_PALETTE_Y_COLOR_INDEX_8_CDF,
            ),
        ),
        (
            0x1100,
            (
                &DEFAULT_PALETTE_UV_COLOR_INDEX_2_CDF,
                &DEFAULT_PALETTE_UV_COLOR_INDEX_3_CDF,
                &DEFAULT_PALETTE_UV_COLOR_INDEX_4_CDF,
                &DEFAULT_PALETTE_UV_COLOR_INDEX_5_CDF,
                &DEFAULT_PALETTE_UV_COLOR_INDEX_6_CDF,
                &DEFAULT_PALETTE_UV_COLOR_INDEX_7_CDF,
                &DEFAULT_PALETTE_UV_COLOR_INDEX_8_CDF,
            ),
        ),
    ] {
        let (c2, c3, c4, c5, c6, c7, c8) = cdfs;
        t.put(base, 8, c2);
        t.put(base + 0x02, 8, c8);
        t.put(base + 0x50, 8, c3);
        t.put(base + 0x54, 8, c7);
        t.put(base + 0xa0, 8, c4);
        t.put(base + 0xa6, 8, c6);
        t.put(base + 0xf0, 4, c5);
    }

    t.put(0x1220, 8, &DEFAULT_INTRA_EXT_TX_SET1_CDF);
    t.put(0x1560, 4, &DEFAULT_INTRA_EXT_TX_SET2_CDF);
    t.put(0x1700, 16, &DEFAULT_INTER_EXT_TX_SET1_CDF);
    t.put(0x1780, 16, &DEFAULT_INTER_EXT_TX_SET2_CDF);
    // After the CDFs of set 2.
    t.put(0x1796, 16, &DEFAULT_INTER_EXT_TX_SET3_CDF);

    t.put(0x1800, 1, &AV1_DEFAULT_TXB_SKIP_CDFS[q]);
    t.put(0x1882, 1, &AV1_DEFAULT_EOB_EXTRA_CDFS[q]);
    t.put(0x1940, 4, &AV1_DEFAULT_EOB_MULTI16_CDFS[q]);
    t.put(0x1960, 8, &AV1_DEFAULT_EOB_MULTI32_CDFS[q]);
    t.put(0x19a0, 8, &AV1_DEFAULT_EOB_MULTI64_CDFS[q]);
    t.put(0x19e0, 8, &AV1_DEFAULT_EOB_MULTI128_CDFS[q]);
    t.put(0x1a20, 8, &AV1_DEFAULT_EOB_MULTI256_CDFS[q]);
    t.put(0x1a60, 16, &AV1_DEFAULT_EOB_MULTI512_CDFS[q]);
    t.put(0x1ae0, 16, &AV1_DEFAULT_EOB_MULTI1024_CDFS[q]);
    t.put(0x1b60, 2, &AV1_DEFAULT_COEFF_BASE_EOB_MULTI_CDFS[q]);
    t.put(0x1c00, 4, &AV1_DEFAULT_COEFF_BASE_MULTI_CDFS[q]);
    t.put(0x2920, 1, &AV1_DEFAULT_DC_SIGN_CDFS[q]);
    t.put(0x2930, 4, &AV1_DEFAULT_COEFF_LPS_MULTI_CDFS[q]);
}

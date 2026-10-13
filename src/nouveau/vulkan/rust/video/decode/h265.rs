// Copyright © 2024 Collabora, Ltd
// SPDX-License-Identifier: MIT

//! H.265 decode, on the decoders of Ampere on (`NVC7B0_VIDEO_DECODER`).
//!
//! For each picture, the driver gives the decoder:
//!
//! - a picture setup (`nvdec_hevc_pic_s`): the SPS and the PPS as Vulkan gives
//!   them; the references, by DPB slot, with their distance in picture order
//!   and whether they are long-term; the initial reference picture lists; and
//!   how many bits of the slice segment headers to skip, those of the picture
//!   order count and the short-term reference picture set, which Vulkan gives
//!   parsed;
//! - the scaling lists, and the size of each tile in CTBs;
//! - the session's memory: the co-located motion vectors of the picture of
//!   each DPB slot, which the decoder writes for the picture it decodes and
//!   reads of the collocated picture, and its buffers of the tile column and
//!   slice boundaries.
//!
//! The decoder finds the slice segments in the bitstream by their start
//! codes, and parses their headers, but for the bits it is told to skip.
//!
//! How NVIDIA's driver programs the decoder was found by recording what it
//! has it do for the same streams on an RTX 3070 (GA104).
//!
//! Based on Daniel Almeida's work in progress on H.265 decode for NVK.

use std::sync::{Arc, Mutex};

use nv_push_rs::*;
use nvidia_headers::classes::clc7b0::mthd as clc7b0;

use crate::ffi;
use crate::ffi::{
    _nvdec_hevc_pic_s, nvk_cmd_buffer, StdVideoDecodeH265PictureInfo,
    StdVideoH265PictureParameterSet, StdVideoH265ScalingLists,
    StdVideoH265SequenceParameterSet, VkVideoDecodeH265PictureInfoKHR,
};
use crate::nvk_cmd_buffer_push;
use crate::util::vk_find_struct_const;
use crate::video::decode::VideoDecoder;

use super::h265_slice::{index_bits, long_term_ref_pics_bits, SliceHeaderInfo};
use super::{
    add_submit_fixup, addr_hi, upload_bytes_mapped, Picture, SubmitFixup,
};

/// The pictures of the decoder, whose index is the DPB slot of the picture
/// in it: up to 16 references of the picture decoded, which its picture
/// setup has the picture order count of, and the picture decoded. A picture
/// decoded to no DPB slot is the last one.
const PICTURES: usize = 17;
const REFERENCES: usize = 16;
const NO_SLOT_INDEX: usize = PICTURES - 1;

/// The decoder takes the addresses of its buffers, and the offsets in the
/// filter memory, in units of 256 bytes.
const BUF_ALIGN: u64 = 256;

/// The picture setup, then the scaling lists, then the tile sizes, which the
/// decoder takes from one buffer, then 512 bytes of status.
const PIC_SETUP_B: usize = 0x200;
const SCALING_LIST_OFFSET: usize = 0x200;
const SCALING_LIST_B: usize = 0x400;
const TILE_SIZES_OFFSET: usize = 0x600;
/// Two bytes of width and two of height of each of up to 20x22 tiles, as
/// `GIP_ASIC_TILE_SIZE`.
const TILE_SIZES_B: usize = 0x700;
const STATUS_OFFSET: usize = 0xd00;
const SETUP_B: usize = 0xf00;

/// The most tile columns and rows of a picture.
const MAX_TILE_COLUMNS: u32 = 20;
const MAX_TILE_ROWS: u32 = 22;

/// The bytes of the intra prediction top line, after the boundary buffers.
const INTRA_TOP_B: u64 = 0x10000;

/// The decoder's buffers of the tile column and slice boundaries, in the
/// filter memory, in units of 256 bytes: per 32 rows of pixels of the largest
/// picture, those of deblocking, SAO and parser control, of the 19
/// boundaries between 20 tile columns; then per row of 64x64 CTBs of the
/// largest picture, the two buffers of the lines above, whose size depends on
/// the width and the CTBs of the picture decoded, and the slice edges; then
/// from 705 units per 32 rows, and 12 more per 64 bytes of the largest
/// picture's pitch, on, the intra prediction top line. These are the sizes
/// NVIDIA's driver lays the memory out with.
const DEBLOCK_ROW32: u32 = 76;
const SAO_ROW32: u32 = 608;
const BSD_CTRL_ROW32: u32 = 19;
const INTRA_TOP_ROW32: u32 = 705;
const INTRA_TOP_PITCH64_ROW32: u32 = 12;

/// The pitch of the luma plane of a picture of `width` of `bit_depth` bits, in
/// bytes, as the decoder writes pictures: a multiple of 64 bytes.
fn pitch_B(width: u32, bit_depth: u32) -> u32 {
    let sample_B = if bit_depth > 8 { 2 } else { 1 };
    (width * sample_B).next_multiple_of(64)
}

/// Where the buffers of the session memory of an H.265 session are, and how
/// big they are, for pictures of up to `width` x `height` of `bit_depth` bits.
#[repr(C)]
#[allow(non_camel_case_types)]
pub struct nvk_video_h265_mem_layout {
    /// The co-located motion vectors of a picture, 256 bytes per 64x64 CTB,
    /// and of all the decoder's pictures.
    pub col_mv_B: u64,
    pub coloc_mem_B: u64,
    /// The rows of 64x64 CTBs of the largest picture.
    pub ctb_rows: u32,
    /// The boundary buffers after the deblocking one, in units of 256 bytes
    /// from the start of the filter memory: those of SAO and parser control,
    /// and the first of the lines above.
    pub sao: u32,
    pub bsd_ctrl: u32,
    pub filter_above: u32,
    pub intra_top_offset_B: u64,
    pub filter_mem_B: u64,
}

#[unsafe(no_mangle)]
pub extern "C" fn nvk_video_h265_mem_layout(
    width: u32,
    height: u32,
    bit_depth: u32,
) -> nvk_video_h265_mem_layout {
    let ctb_cols = width.div_ceil(64);
    let ctb_rows = height.div_ceil(64);
    let rows32 = height.div_ceil(32);

    let col_mv_B = u64::from(ctb_cols * ctb_rows) * BUF_ALIGN;
    let coloc_mem_B = (col_mv_B * PICTURES as u64).next_multiple_of(4096);

    let sao = rows32 * DEBLOCK_ROW32;
    let bsd_ctrl = sao + rows32 * SAO_ROW32;
    let filter_above = bsd_ctrl + rows32 * BSD_CTRL_ROW32;
    let intra_top = rows32
        * (INTRA_TOP_ROW32
            + pitch_B(width, bit_depth) / 64 * INTRA_TOP_PITCH64_ROW32);
    let intra_top_offset_B = u64::from(intra_top) * BUF_ALIGN;

    nvk_video_h265_mem_layout {
        col_mv_B,
        coloc_mem_B,
        ctb_rows,
        sao,
        bsd_ctrl,
        filter_above,
        intra_top_offset_B,
        filter_mem_B: (intra_top_offset_B + INTRA_TOP_B).next_multiple_of(4096),
    }
}

impl nvk_video_h265_mem_layout {
    /// The offsets of the second buffer of the lines above and of the slice
    /// edges, in the filter memory, for a picture of `width` of `bit_depth`
    /// bits, of CTBs of `1 << ctb_log2`: the buffers of the lines above take
    /// so many units per row of 64x64 CTBs and per CTB column of the
    /// picture's pitch, the first one a column more.
    fn above_offsets(
        &self,
        width: u32,
        ctb_log2: u32,
        bit_depth: u32,
    ) -> (u32, u32) {
        let sample_B = if bit_depth > 8 { 2 } else { 1 };
        let cols =
            (pitch_B(width, bit_depth) / sample_B).div_ceil(1 << ctb_log2);
        let (filter_above, sao_above) = match (ctb_log2, bit_depth > 8) {
            (4, _) => (4, 2),
            (5, false) => (3, 1),
            (5, true) => (4, 2),
            (_, false) => (3, 1),
            (_, true) => (4, 1),
        };
        let sao_above_offset =
            self.filter_above + self.ctb_rows * (cols + 1) * filter_above;
        (
            sao_above_offset,
            sao_above_offset + self.ctb_rows * cols * sao_above,
        )
    }
}

/// The decoder's timeout for a picture, in cycles, after which it gives up on
/// it, which bounds how long a malformed picture keeps the engine: NVIDIA's
/// driver gives its VP9 decodes 39.0625 cycles per pixel, and leaves the
/// default of its H.265 ones, 1 << 27 cycles, which this never is less than.
fn timeout(width: u32, height: u32) -> u32 {
    let per_pixel = u64::from(width) * u64::from(height) * 625 / 16;
    per_pixel.max(1 << 27).try_into().unwrap_or(u32::MAX)
}

/// The up-right diagonal scan of a `size` x `size` block (6.5.3 of the H.265
/// specification): the position (x, y) of each coefficient, in the order of
/// the scaling lists.
fn diagonal_scan(size: usize) -> impl Iterator<Item = (usize, usize)> {
    (0..2 * size - 1).flat_map(move |d| {
        (0..=d)
            .map(move |x| (x, d - x))
            .filter(move |&(x, y)| x < size && y < size)
    })
}

/// The default scaling lists of the 8x8 and larger blocks, intra and inter,
/// in up-right diagonal order (table 7-6 of the H.265 specification).
const DEFAULT_SCALING_INTRA: [u8; 64] = [
    16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 17, 16, 17, 16, 17, 18, 17, 18, 18,
    17, 18, 21, 19, 20, 21, 20, 19, 21, 24, 22, 22, 24, 24, 22, 22, 24, 25, 25,
    27, 30, 27, 25, 25, 29, 31, 35, 35, 31, 29, 36, 41, 44, 41, 36, 47, 54, 54,
    47, 65, 70, 65, 88, 88, 115,
];
const DEFAULT_SCALING_INTER: [u8; 64] = [
    16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 17, 17, 17, 17, 17, 18, 18, 18, 18,
    18, 18, 20, 20, 20, 20, 20, 20, 20, 24, 24, 24, 24, 24, 24, 24, 24, 25, 25,
    25, 25, 25, 25, 25, 28, 28, 28, 28, 28, 28, 33, 33, 33, 33, 33, 41, 41, 41,
    41, 54, 54, 54, 71, 71, 91,
];

/// The scaling lists a picture is decoded with when they are enabled and
/// neither its SPS nor its PPS has its own (7.4.5 of the H.265
/// specification).
fn default_scaling_lists() -> StdVideoH265ScalingLists {
    let mut lists: StdVideoH265ScalingLists = unsafe { std::mem::zeroed() };
    lists.ScalingList4x4 = [[16; 16]; 6];
    for m in 0..6 {
        let default = if m < 3 {
            DEFAULT_SCALING_INTRA
        } else {
            DEFAULT_SCALING_INTER
        };
        lists.ScalingList8x8[m] = default;
        lists.ScalingList16x16[m] = default;
    }
    // Of 32x32 blocks, only luma ones have lists: matrixId 0 and 3.
    lists.ScalingList32x32 = [DEFAULT_SCALING_INTRA, DEFAULT_SCALING_INTER];
    lists.ScalingListDCCoef16x16 = [16; 6];
    lists.ScalingListDCCoef32x32 = [16; 2];
    lists
}

/// Writes scaling lists as the decoder takes them: the DC coefficients of the
/// 16x16 and 32x32 lists, then from 16 bytes on the lists of each block size,
/// each a column of the block after the other.
fn write_scaling_lists(lists: &StdVideoH265ScalingLists, dst: &mut [u8]) {
    let dc = lists
        .ScalingListDCCoef16x16
        .iter()
        .chain(&lists.ScalingListDCCoef32x32);
    for (d, &v) in dst.iter_mut().zip(dc) {
        *d = v;
    }

    let mut offset = 16;
    let mut write = |size: usize, list: &[u8]| {
        for ((x, y), &v) in diagonal_scan(size).zip(list) {
            dst[offset + x * size + y] = v;
        }
        offset += size * size;
    };
    for list in &lists.ScalingList4x4 {
        write(4, list);
    }
    for list in &lists.ScalingList8x8 {
        write(8, list);
    }
    // The lists of the 16x16 and 32x32 blocks are 8x8 ones, upsampled.
    for list in &lists.ScalingList16x16 {
        write(8, list);
    }
    for list in &lists.ScalingList32x32 {
        write(8, list);
    }
}

/// The widths of the tile columns, or the heights of the tile rows, of a
/// picture of `ctbs` CTBs in that dimension (6.5.1 of the H.265
/// specification).
fn tile_sizes(
    ctbs: u32,
    count: u32,
    uniform: bool,
    minus1: &[u16],
) -> impl Iterator<Item = u32> + '_ {
    let mut left = ctbs;
    (0..count).map(move |i| {
        let size = if uniform {
            ((i + 1) * ctbs) / count - (i * ctbs) / count
        } else if i + 1 < count {
            u32::from(minus1[i as usize]) + 1
        } else {
            left
        };
        left -= size;
        size
    })
}

/// Session data stored in the opaque `nvk_video_session::rust` pointer.
#[derive(Default, Debug)]
pub(crate) struct Decoder {
    state: Arc<Mutex<DecodeState>>,
}

/// The session's state of the pictures decoded, in the order the decoder
/// decodes them, which [`PictureFixup`]s keep up to date.
#[derive(Default, Debug)]
struct DecodeState {
    /// Pictures decoded, which the decoder takes as the picture's index.
    picture_index: u32,
}

/// Forgets the pictures decoded before, when the command buffer is submitted.
struct ResetFixup {
    state: Arc<Mutex<DecodeState>>,
}

impl SubmitFixup for ResetFixup {
    fn submit(&mut self) {
        *self.state.lock().unwrap() = DecodeState::default();
    }
}

/// The parts of a picture's decode filled in when the command buffer is
/// submitted: its picture index, and how many bits of the slice segment
/// headers the decoder skips when they have a long-term reference picture
/// set.
struct PictureFixup {
    state: Arc<Mutex<DecodeState>>,
    picture_index: *mut u32,
    /// The picture setup, where it is uploaded.
    pic_map: *mut _nvdec_hevc_pic_s,
    long_term: Option<LongTermBits>,
}

impl SubmitFixup for PictureFixup {
    fn submit(&mut self) {
        let mut state = self.state.lock().unwrap();
        // SAFETY: the pushbuffer and the uploads are the command buffer's,
        // which the GPU doesn't use until it is submitted.
        unsafe { self.picture_index.write(state.picture_index) };
        state.picture_index = state.picture_index.wrapping_add(1);

        if let Some(lt) = &self.long_term {
            // If the header can't be read, the decoder skips none of the set:
            // the picture is decoded wrong, as it is malformed.
            let skip = lt.skip + lt.read().unwrap_or(0);
            unsafe {
                std::ptr::addr_of_mut!((*self.pic_map).sw_hdr_skip_length)
                    .write(skip.try_into().unwrap_or(u16::MAX));
            }
        }
    }
}

/// The bytes of the first slice segment of a picture NVK reads at most to
/// find how many bits its long-term reference picture set takes.
const MAX_SLICE_HEADER_B: u64 = 1024;

/// What finding how many bits the long-term reference picture set of a
/// picture takes needs: the header of its first slice segment, which Vulkan
/// doesn't give parsed, and which only the bitstream has. The bitstream is
/// read when the command buffer is submitted, by which the application has
/// written it, from the CPU: the buffers of bitstreams are host visible (see
/// nvk_GetDeviceBufferMemoryRequirements()).
struct LongTermBits {
    dev: *mut ffi::nvk_device,
    /// The GPU address of the first slice segment, and how many bytes of the
    /// bitstream are from there on.
    addr: u64,
    len: u64,
    info: SliceHeaderInfo,
    /// The bits the decoder skips before the long-term reference picture
    /// set.
    skip: u32,
}

impl LongTermBits {
    /// The bits of the long-term reference picture set, if the slice
    /// segment header can be read up to its end.
    fn read(&self) -> Option<u32> {
        // SAFETY: the bitstream buffer is alive until the decode is done, and
        // its memory is too.
        unsafe {
            let dev = &mut *self.dev;
            let mut offset = 0;
            let mem = ffi::nvkmd_dev_lookup_mem_by_va(
                dev.nvkmd,
                self.addr,
                &mut offset,
            );
            if mem.is_null() {
                return None;
            }
            let bits = if (*mem).flags & ffi::NVKMD_MEM_CAN_MAP != 0
                && offset < (*mem).size_B
            {
                let len = self
                    .len
                    .min(MAX_SLICE_HEADER_B)
                    .min((*mem).size_B - offset);
                let mut map = std::ptr::null_mut();
                if ffi::nvkmd_mem_map(
                    mem,
                    &mut dev.vk.base,
                    ffi::NVKMD_MEM_MAP_RD,
                    std::ptr::null_mut(),
                    &mut map,
                ) == ffi::VK_SUCCESS
                {
                    ffi::nvkmd_mem_sync_from_gpu(mem, false, offset, len);
                    let data = std::slice::from_raw_parts(
                        map.cast::<u8>().add(offset.try_into().unwrap()),
                        len.try_into().unwrap(),
                    );
                    let bits = long_term_ref_pics_bits(data, &self.info);
                    ffi::nvkmd_mem_unmap(mem, 0);
                    bits
                } else {
                    None
                }
            } else {
                None
            };
            ffi::nvkmd_mem_unref(mem);
            bits
        }
    }
}

/// A reference of the picture decoded.
struct Reference {
    slot: usize,
    picture: Picture,
    poc: i32,
    long_term: bool,
}

/// The references of the decode `frame_info`, by DPB slot.
fn references(
    frame_info: &ffi::VkVideoDecodeInfoKHR,
) -> Result<[Option<Reference>; PICTURES], ffi::VkResult> {
    let mut refs: [Option<Reference>; PICTURES] = Default::default();
    for i in 0..frame_info.referenceSlotCount as usize {
        let slot = unsafe { &*frame_info.pReferenceSlots.add(i) };
        let index = usize::try_from(slot.slotIndex)
            .ok()
            .filter(|&s| s < PICTURES)
            .ok_or(ffi::VK_ERROR_INVALID_VIDEO_STD_PARAMETERS_KHR)?;
        let dpb = vk_find_struct_const!(
            slot.pNext,
            VIDEO_DECODE_H265_DPB_SLOT_INFO,
            KHR
        );
        let info = unsafe { &*dpb.pStdReferenceInfo };
        refs[index] = Some(Reference {
            slot: index,
            picture: Picture::new(unsafe { &*slot.pPictureResource }),
            poc: info.PicOrderCntVal,
            long_term: info.flags.used_for_long_term_reference() != 0,
        });
    }
    Ok(refs)
}

/// Whether a picture of `bit_depth` bits can be decoded to the image of
/// `picture`: P010 for 10 bits, NV12 for 8, which the decoder writes past
/// otherwise.
fn picture_fits(
    picture: &Picture,
    bit_depth: u32,
    width: u32,
    height: u32,
) -> bool {
    let image = unsafe { &(*picture.img).vk };
    let format = match bit_depth {
        8 => ffi::VK_FORMAT_G8_B8R8_2PLANE_420_UNORM,
        10 => ffi::VK_FORMAT_G10X6_B10X6R10X6_2PLANE_420_UNORM_3PACK16,
        _ => return false,
    };
    image.format == format
        && width <= image.extent.width
        && height <= image.extent.height
}

/// Checks what of the SPS and the PPS the decoder relies on to stay in its
/// buffers: they are the application's, which the decoder must not be given
/// anything it could read or write out of bounds with.
fn check_parameters(
    sps: &StdVideoH265SequenceParameterSet,
    pps: &StdVideoH265PictureParameterSet,
    max_width: u32,
    max_height: u32,
) -> Result<(), ffi::VkResult> {
    let invalid = Err(ffi::VK_ERROR_INVALID_VIDEO_STD_PARAMETERS_KHR);

    let min_cb_log2 = u32::from(sps.log2_min_luma_coding_block_size_minus3) + 3;
    let ctb_log2 =
        min_cb_log2 + u32::from(sps.log2_diff_max_min_luma_coding_block_size);
    let min_tb_log2 =
        u32::from(sps.log2_min_luma_transform_block_size_minus2) + 2;
    let max_tb_log2 = min_tb_log2
        + u32::from(sps.log2_diff_max_min_luma_transform_block_size);
    let width = sps.pic_width_in_luma_samples;
    let height = sps.pic_height_in_luma_samples;

    if sps.chroma_format_idc != ffi::STD_VIDEO_H265_CHROMA_FORMAT_IDC_420
        || sps.flags.separate_colour_plane_flag() != 0
        || sps.bit_depth_luma_minus8 != sps.bit_depth_chroma_minus8
        || sps.bit_depth_luma_minus8 > 2
        || !(4..=6).contains(&ctb_log2)
        || min_tb_log2 >= min_cb_log2
        || max_tb_log2 > ctb_log2.min(5)
        || width == 0
        || height == 0
        || width > max_width
        || height > max_height
        || width % (1 << min_cb_log2) != 0
        || height % (1 << min_cb_log2) != 0
    {
        return invalid;
    }

    if sps.flags.pcm_enabled_flag() != 0 {
        let pcm_min_log2 =
            u32::from(sps.log2_min_pcm_luma_coding_block_size_minus3) + 3;
        let pcm_max_log2 = pcm_min_log2
            + u32::from(sps.log2_diff_max_min_pcm_luma_coding_block_size);
        if sps.pcm_sample_bit_depth_luma_minus1 > sps.bit_depth_luma_minus8 + 7
            || sps.pcm_sample_bit_depth_chroma_minus1
                > sps.bit_depth_chroma_minus8 + 7
            || pcm_min_log2 < min_cb_log2
            || pcm_max_log2 > ctb_log2.min(5)
        {
            return invalid;
        }
    }

    if pps.flags.tiles_enabled_flag() != 0 {
        let ctb = 1 << ctb_log2;
        let cols = u32::from(pps.num_tile_columns_minus1) + 1;
        let rows = u32::from(pps.num_tile_rows_minus1) + 1;
        let ctb_cols = width.div_ceil(ctb);
        let ctb_rows = height.div_ceil(ctb);
        if cols > MAX_TILE_COLUMNS
            || rows > MAX_TILE_ROWS
            || cols > ctb_cols
            || rows > ctb_rows
        {
            return invalid;
        }
        if pps.flags.uniform_spacing_flag() == 0 {
            let sum = |minus1: &[u16]| -> u32 {
                minus1.iter().map(|&m| u32::from(m) + 1).sum()
            };
            if sum(&pps.column_width_minus1[..cols as usize - 1]) >= ctb_cols
                || sum(&pps.row_height_minus1[..rows as usize - 1]) >= ctb_rows
            {
                return invalid;
            }
        }
    }

    // SliceQpY + QpBdOffsetY, which the decoder starts from, is of 0 to 51 +
    // QpBdOffsetY.
    let qp_bd_offset = 6 * i32::from(sps.bit_depth_luma_minus8);
    let init_qp = i32::from(pps.init_qp_minus26) + 26 + qp_bd_offset;

    if !(-6..=6).contains(&pps.pps_beta_offset_div2)
        || !(-6..=6).contains(&pps.pps_tc_offset_div2)
        || !(-12..=12).contains(&pps.pps_cb_qp_offset)
        || !(-12..=12).contains(&pps.pps_cr_qp_offset)
    {
        return invalid;
    }

    if pps.diff_cu_qp_delta_depth > sps.log2_diff_max_min_luma_coding_block_size
        || pps.log2_parallel_merge_level_minus2 + 2 > ctb_log2 as u8
        || !(0..=51 + qp_bd_offset).contains(&init_qp)
    {
        return invalid;
    }

    Ok(())
}

/// The bit depth of the pictures of a session.
fn session_bit_depth(session: &ffi::vk_video_session) -> u32 {
    match session.luma_bit_depth {
        ffi::VK_VIDEO_COMPONENT_BIT_DEPTH_10_BIT_KHR => 10,
        ffi::VK_VIDEO_COMPONENT_BIT_DEPTH_12_BIT_KHR => 12,
        _ => 8,
    }
}

/// The initial reference picture lists of a picture (8.3.4 of the H.265
/// specification), as the decoder takes them: the DPB slots of its
/// references, the short-term ones before it in output order, then those
/// after it, then the long-term ones, for list 0, and those after it first
/// for list 1, each repeated over the list.
fn initial_lists(std: &StdVideoDecodeH265PictureInfo) -> ([u8; 16], [u8; 16]) {
    let set = |s: &[u8; 8]| -> Vec<u8> {
        s.iter().copied().take_while(|&i| i != 0xff).collect()
    };
    let before = set(&std.RefPicSetStCurrBefore);
    let after = set(&std.RefPicSetStCurrAfter);
    let long_term = set(&std.RefPicSetLtCurr);

    let list = |first: &[u8], second: &[u8]| -> [u8; 16] {
        let all: Vec<u8> = first
            .iter()
            .chain(second)
            .chain(&long_term)
            .copied()
            .collect();
        let mut list = [0; 16];
        if !all.is_empty() {
            for (i, l) in list.iter_mut().enumerate() {
                *l = all[i % all.len()];
            }
        }
        list
    };
    (list(&before, &after), list(&after, &before))
}

impl VideoDecoder for Decoder {
    fn reset(&mut self, cmd: &mut nvk_cmd_buffer) {
        add_submit_fixup(
            cmd,
            Box::new(ResetFixup {
                state: self.state.clone(),
            }),
        );
    }

    fn decode(
        &mut self,
        cmd: &mut nvk_cmd_buffer,
        frame_info: &ffi::VkVideoDecodeInfoKHR,
    ) -> Result<(), ffi::VkResult> {
        let h265 = vk_find_struct_const!(
            frame_info.pNext,
            VIDEO_DECODE_H265_PICTURE_INFO,
            KHR,
            VkVideoDecodeH265PictureInfoKHR
        );
        let std: &StdVideoDecodeH265PictureInfo =
            unsafe { &*h265.pStdPictureInfo };

        // The parameters given inline (VK_KHR_video_maintenance2), or else
        // those of the session parameters.
        let mut sps_ptr = std::ptr::null();
        let mut pps_ptr = std::ptr::null();
        unsafe {
            ffi::vk_video_get_h265_parameters(
                &(*cmd.state.video.vid).vk,
                cmd.state.video.params as *const _,
                frame_info as *const _,
                &h265 as *const _,
                &mut sps_ptr,
                &mut pps_ptr,
            );
        }
        let (Some(sps), Some(pps)) =
            (unsafe { sps_ptr.as_ref() }, unsafe { pps_ptr.as_ref() })
        else {
            return Err(ffi::VK_ERROR_INVALID_VIDEO_STD_PARAMETERS_KHR);
        };

        let session = unsafe { cmd.state.video.vid.as_ref().unwrap() };
        let max = session.vk.max_coded;
        check_parameters(sps, pps, max.width, max.height)?;
        // The session's memory is laid out for pictures of its bit depth.
        let session_bit_depth = session_bit_depth(&session.vk);
        if u32::from(sps.bit_depth_luma_minus8) + 8 != session_bit_depth {
            return Err(ffi::VK_ERROR_INVALID_VIDEO_STD_PARAMETERS_KHR);
        }

        let width = sps.pic_width_in_luma_samples;
        let height = sps.pic_height_in_luma_samples;
        let bit_depth = u32::from(sps.bit_depth_luma_minus8) + 8;
        let min_cb_log2 =
            u32::from(sps.log2_min_luma_coding_block_size_minus3) + 3;
        let ctb_log2 = min_cb_log2
            + u32::from(sps.log2_diff_max_min_luma_coding_block_size);
        let min_tb_log2 =
            u32::from(sps.log2_min_luma_transform_block_size_minus2) + 2;

        // The picture is decoded to its DPB slot's picture, which is the
        // destination picture (DPB_AND_OUTPUT_COINCIDE), and is the decoder's
        // picture of the index of the slot, or of the index of no slot.
        let dst = Picture::new(&frame_info.dstPictureResource);
        let setup = unsafe { frame_info.pSetupReferenceSlot.as_ref() };
        let cur_idx = match setup {
            Some(s) => usize::try_from(s.slotIndex)
                .ok()
                .filter(|&s| s < PICTURES)
                .ok_or(ffi::VK_ERROR_INVALID_VIDEO_STD_PARAMETERS_KHR)?,
            None => NO_SLOT_INDEX,
        };
        let refs = references(frame_info)?;
        if setup.is_none() && refs[NO_SLOT_INDEX].is_some() {
            return Err(ffi::VK_ERROR_INVALID_VIDEO_STD_PARAMETERS_KHR);
        }
        if !picture_fits(&dst, bit_depth, width, height)
            || refs
                .iter()
                .flatten()
                .any(|r| !picture_fits(&r.picture, bit_depth, width, height))
        {
            return Err(ffi::VK_ERROR_INVALID_VIDEO_STD_PARAMETERS_KHR);
        }
        // The reference picture sets may only name the references given, of
        // the slots the picture setup has the picture order count of.
        let in_dpb = |s: &[u8; 8]| {
            s.iter().all(|&i| {
                i == 0xff
                    || usize::from(i) < REFERENCES
                        && refs[usize::from(i)].is_some()
            })
        };
        if !in_dpb(&std.RefPicSetStCurrBefore)
            || !in_dpb(&std.RefPicSetStCurrAfter)
            || !in_dpb(&std.RefPicSetLtCurr)
        {
            return Err(ffi::VK_ERROR_INVALID_VIDEO_STD_PARAMETERS_KHR);
        }

        let layout =
            nvk_video_h265_mem_layout(max.width, max.height, bit_depth);
        let (sao_above, slice_edge) =
            layout.above_offsets(width, ctb_log2, bit_depth);

        let mut pic: _nvdec_hevc_pic_s = unsafe { std::mem::zeroed() };
        const { assert!(std::mem::size_of::<_nvdec_hevc_pic_s>() <= PIC_SETUP_B) };

        // The bitstream, from its address in units of 256 bytes, from the
        // block of the first slice segment on.
        let src_buffer =
            unsafe { ffi::nvk_buffer_from_handle(frame_info.srcBuffer) };
        let range: u32 = frame_info
            .srcBufferRange
            .try_into()
            .map_err(|_| ffi::VK_ERROR_INVALID_VIDEO_STD_PARAMETERS_KHR)?;
        if h265.sliceSegmentCount == 0 {
            return Err(ffi::VK_ERROR_INVALID_VIDEO_STD_PARAMETERS_KHR);
        }
        let first_slice = unsafe { *h265.pSliceSegmentOffsets };
        if first_slice >= range {
            return Err(ffi::VK_ERROR_INVALID_VIDEO_STD_PARAMETERS_KHR);
        }
        let src = unsafe {
            ffi::vk_buffer_address(
                &(*src_buffer).vk,
                frame_info.srcBufferOffset,
            )
        };
        let src_base = (src + u64::from(first_slice)) & !(BUF_ALIGN - 1);
        pic.stream_len = range - u32::try_from(src_base - src).unwrap();
        pic.gptimer_timeout_value = timeout(width, height);
        pic.set_tileformat(1);
        let y_log2 =
            unsafe { (*dst.img).planes[0].nil.levels[0].tiling.y_log2 };
        pic.set_gob_height((y_log2 - 1).into());
        pic.sw_start_code_e = 1;
        // Rec. 709 output, of 10 bits above 8 bits.
        pic.disp_output_mode = (bit_depth > 8).into();
        // The decoder takes the pictures' pitch in samples, of 2 bytes above
        // 8 bits.
        let pitch = |plane: usize| -> u32 {
            u32::from(dst.stride(plane)) >> u32::from(bit_depth > 8)
        };
        pic.framestride = [pitch(0), pitch(1)];

        // The session memory.
        pic.colMvBuffersize = (layout.col_mv_B / BUF_ALIGN).try_into().unwrap();
        pic.HevcSaoBufferOffset = layout.sao;
        pic.HevcBsdCtrlOffset = layout.bsd_ctrl;
        let ext = &mut pic.v1.hevc_main10_444_ext;
        ext.HevcFltAboveOffset = layout.filter_above;
        ext.HevcSaoAboveOffset = sao_above;
        // Transform skip, of 4x4 blocks only before the range extensions.
        ext.set_log2MaxTransformSkipSize(2);
        pic.v3.HevcSliceEdgeOffset = slice_edge;

        // The SPS.
        pic.pic_width_in_luma_samples = width.try_into().unwrap();
        pic.pic_height_in_luma_samples = height.try_into().unwrap();
        pic.set_chroma_format_idc(sps.chroma_format_idc);
        pic.set_bit_depth_luma(bit_depth);
        pic.set_bit_depth_chroma(u32::from(sps.bit_depth_chroma_minus8) + 8);
        pic.set_log2_min_luma_coding_block_size(min_cb_log2);
        pic.set_log2_max_luma_coding_block_size(ctb_log2);
        pic.set_log2_min_transform_block_size(min_tb_log2);
        pic.set_log2_max_transform_block_size(
            min_tb_log2
                + u32::from(sps.log2_diff_max_min_luma_transform_block_size),
        );
        pic.set_max_transform_hierarchy_depth_inter(
            sps.max_transform_hierarchy_depth_inter.into(),
        );
        pic.set_max_transform_hierarchy_depth_intra(
            sps.max_transform_hierarchy_depth_intra.into(),
        );
        let scaling = sps.flags.scaling_list_enabled_flag() != 0;
        pic.set_scalingListEnable(scaling.into());
        pic.set_amp_enable_flag(sps.flags.amp_enabled_flag());
        pic.set_sample_adaptive_offset_enabled_flag(
            sps.flags.sample_adaptive_offset_enabled_flag(),
        );
        if sps.flags.pcm_enabled_flag() != 0 {
            let pcm_min_log2 =
                u32::from(sps.log2_min_pcm_luma_coding_block_size_minus3) + 3;
            pic.set_pcm_enabled_flag(1);
            pic.set_pcm_sample_bit_depth_luma(
                u32::from(sps.pcm_sample_bit_depth_luma_minus1) + 1,
            );
            pic.set_pcm_sample_bit_depth_chroma(
                u32::from(sps.pcm_sample_bit_depth_chroma_minus1) + 1,
            );
            pic.set_log2_min_pcm_luma_coding_block_size(pcm_min_log2);
            pic.set_log2_max_pcm_luma_coding_block_size(
                pcm_min_log2
                    + u32::from(
                        sps.log2_diff_max_min_pcm_luma_coding_block_size,
                    ),
            );
            pic.set_pcm_loop_filter_disabled_flag(
                sps.flags.pcm_loop_filter_disabled_flag(),
            );
        }
        pic.set_sps_temporal_mvp_enabled_flag(
            sps.flags.sps_temporal_mvp_enabled_flag(),
        );
        pic.set_strong_intra_smoothing_enabled_flag(
            sps.flags.strong_intra_smoothing_enabled_flag(),
        );

        // The PPS.
        pic.set_dependent_slice_segments_enabled_flag(
            pps.flags.dependent_slice_segments_enabled_flag(),
        );
        pic.set_output_flag_present_flag(pps.flags.output_flag_present_flag());
        pic.set_num_extra_slice_header_bits(
            pps.num_extra_slice_header_bits.into(),
        );
        pic.set_sign_data_hiding_enabled_flag(
            pps.flags.sign_data_hiding_enabled_flag(),
        );
        pic.set_cabac_init_present_flag(pps.flags.cabac_init_present_flag());
        pic.set_num_ref_idx_l0_default_active(
            u32::from(pps.num_ref_idx_l0_default_active_minus1) + 1,
        );
        pic.set_num_ref_idx_l1_default_active(
            u32::from(pps.num_ref_idx_l1_default_active_minus1) + 1,
        );
        // SliceQpY + QpBdOffsetY, which is never negative.
        let qp_bd_offset = 6 * i32::from(sps.bit_depth_luma_minus8);
        pic.set_init_qp(
            (i32::from(pps.init_qp_minus26) + 26 + qp_bd_offset)
                .try_into()
                .unwrap(),
        );
        pic.set_constrained_intra_pred_flag(
            pps.flags.constrained_intra_pred_flag(),
        );
        pic.set_transform_skip_enabled_flag(
            pps.flags.transform_skip_enabled_flag(),
        );
        pic.set_cu_qp_delta_enabled_flag(pps.flags.cu_qp_delta_enabled_flag());
        pic.set_diff_cu_qp_delta_depth(pps.diff_cu_qp_delta_depth.into());
        pic.pps_cb_qp_offset = pps.pps_cb_qp_offset;
        pic.pps_cr_qp_offset = pps.pps_cr_qp_offset;
        pic.pps_beta_offset = pps.pps_beta_offset_div2 * 2;
        pic.pps_tc_offset = pps.pps_tc_offset_div2 * 2;
        pic.set_pps_slice_chroma_qp_offsets_present_flag(
            pps.flags.pps_slice_chroma_qp_offsets_present_flag(),
        );
        pic.set_weighted_pred_flag(pps.flags.weighted_pred_flag());
        pic.set_weighted_bipred_flag(pps.flags.weighted_bipred_flag());
        pic.set_transquant_bypass_enabled_flag(
            pps.flags.transquant_bypass_enabled_flag(),
        );
        let tiles = pps.flags.tiles_enabled_flag() != 0;
        pic.set_tiles_enabled_flag(tiles.into());
        pic.set_entropy_coding_sync_enabled_flag(
            pps.flags.entropy_coding_sync_enabled_flag(),
        );
        let (tile_cols, tile_rows) = if tiles {
            (
                u32::from(pps.num_tile_columns_minus1) + 1,
                u32::from(pps.num_tile_rows_minus1) + 1,
            )
        } else {
            (1, 1)
        };
        if tiles {
            pic.set_num_tile_columns(tile_cols);
            pic.set_num_tile_rows(tile_rows);
        }
        pic.set_loop_filter_across_tiles_enabled_flag(
            pps.flags.loop_filter_across_tiles_enabled_flag(),
        );
        pic.set_loop_filter_across_slices_enabled_flag(
            pps.flags.pps_loop_filter_across_slices_enabled_flag(),
        );
        pic.set_deblocking_filter_control_present_flag(
            pps.flags.deblocking_filter_control_present_flag(),
        );
        pic.set_deblocking_filter_override_enabled_flag(
            pps.flags.deblocking_filter_override_enabled_flag(),
        );
        pic.set_pps_deblocking_filter_disabled_flag(
            pps.flags.pps_deblocking_filter_disabled_flag(),
        );
        pic.set_lists_modification_present_flag(
            pps.flags.lists_modification_present_flag(),
        );
        pic.set_log2_parallel_merge_level(
            u32::from(pps.log2_parallel_merge_level_minus2) + 2,
        );
        pic.set_slice_segment_header_extension_present_flag(
            pps.flags.slice_segment_header_extension_present_flag(),
        );

        // The references, by DPB slot, which is the decoder's picture index.
        let idr = std.flags.IdrPicFlag() != 0;
        // The references the picture itself uses, NumPicTotalCurr, not those
        // kept for the pictures after it.
        pic.num_ref_frames = [
            &std.RefPicSetStCurrBefore,
            &std.RefPicSetStCurrAfter,
            &std.RefPicSetLtCurr,
        ]
        .iter()
        .map(|s| s.iter().take_while(|&&i| i != 0xff).count())
        .sum::<usize>()
        .try_into()
        .unwrap();
        for r in refs[..REFERENCES].iter().flatten() {
            // The references of RefPicSetLtCurr are long-term ones, whether
            // their reference information says so or not, which ffmpeg's (as
            // of 9.0) never does.
            let lt_curr = std.RefPicSetLtCurr.contains(&(r.slot as u8));
            if r.long_term || lt_curr {
                pic.longtermflag |= 0x8000 >> r.slot;
            }
            pic.RefDiffPicOrderCnts[r.slot] = (i64::from(std.PicOrderCntVal)
                - i64::from(r.poc))
            .clamp(i16::MIN.into(), i16::MAX.into())
                as i16;
        }
        (pic.initreflistidxl0, pic.initreflistidxl1) = initial_lists(std);
        pic.IDR_picture_flag = idr.into();
        pic.RAP_picture_flag = std.flags.IrapPicFlag().try_into().unwrap();
        pic.curr_pic_idx = cur_idx.try_into().unwrap();
        // As NVIDIA's driver: the table of dithering, which it gives 8-bit
        // pictures.
        pic.pattern_id = if bit_depth > 8 { 0 } else { 2 };

        // The bits of the slice segment headers the decoder skips, which
        // follow slice_type: pic_output_flag, and of the pictures which
        // aren't IDR ones, the picture order count and the reference
        // picture sets, which Vulkan gives parsed, but how many bits the
        // long-term one takes. That is found in the header of the first slice
        // segment when the command buffer is submitted.
        let poc_bits = u32::from(sps.log2_max_pic_order_cnt_lsb_minus4) + 4;
        let st_rps_bits = if std.flags.short_term_ref_pic_set_sps_flag() == 0 {
            u32::from(std.NumBitsForSTRefPicSetInSlice)
        } else {
            index_bits(sps.num_short_term_ref_pic_sets.into())
        };
        let output_flag_bits = pps.flags.output_flag_present_flag();
        let skip = if idr {
            output_flag_bits
        } else {
            output_flag_bits + poc_bits + 1 + st_rps_bits
        };
        pic.sw_hdr_skip_length = skip.try_into().unwrap();
        let long_term = (!idr
            && sps.flags.long_term_ref_pics_present_flag() != 0)
            .then(|| LongTermBits {
                dev: cmd.vk.base.device.cast(),
                addr: src + u64::from(first_slice),
                len: u64::from(range - first_slice),
                info: SliceHeaderInfo {
                    num_extra_slice_header_bits: pps
                        .num_extra_slice_header_bits
                        .into(),
                    output_flag_present: output_flag_bits != 0,
                    separate_colour_plane: false,
                    log2_max_pic_order_cnt_lsb: poc_bits,
                    num_short_term_ref_pic_sets: sps
                        .num_short_term_ref_pic_sets
                        .into(),
                    short_term_ref_pic_set_bits: std
                        .NumBitsForSTRefPicSetInSlice
                        .into(),
                    long_term_ref_pics_present: true,
                    num_long_term_ref_pics_sps: sps
                        .num_long_term_ref_pics_sps
                        .into(),
                },
                skip,
            });
        pic.set_log2_max_pic_order_cnt_lsb_minus4(
            sps.log2_max_pic_order_cnt_lsb_minus4.into(),
        );
        pic.set_num_short_term_ref_pic_sets(
            sps.num_short_term_ref_pic_sets.into(),
        );
        pic.set_num_long_term_ref_pics_sps(
            sps.num_long_term_ref_pics_sps.into(),
        );
        pic.set_long_term_ref_pics_present_flag(
            sps.flags.long_term_ref_pics_present_flag(),
        );
        pic.set_num_delta_pocs_of_rps_idx(std.NumDeltaPocsOfRefRpsIdx.into());
        pic.num_bits_short_term_ref_pics_in_slice =
            std.NumBitsForSTRefPicSetInSlice.into();

        // The scaling lists of the PPS, or else of the SPS, or else the
        // default ones.
        let mut setup_data = [0u8; SETUP_B];
        if scaling {
            let lists = if pps.flags.pps_scaling_list_data_present_flag() != 0 {
                unsafe { pps.pScalingLists.as_ref() }
            } else if sps.flags.sps_scaling_list_data_present_flag() != 0 {
                unsafe { sps.pScalingLists.as_ref() }
            } else {
                None
            };
            let default;
            let lists = match lists {
                Some(l) => l,
                None => {
                    default = default_scaling_lists();
                    &default
                }
            };
            write_scaling_lists(
                lists,
                &mut setup_data
                    [SCALING_LIST_OFFSET..SCALING_LIST_OFFSET + SCALING_LIST_B],
            );
        }

        // The size of each tile, in CTBs: its width, then its height.
        let ctb = 1 << ctb_log2;
        let (ctb_cols, ctb_rows) = (width.div_ceil(ctb), height.div_ceil(ctb));
        let uniform = !tiles || pps.flags.uniform_spacing_flag() != 0;
        let col_widths: Vec<u32> =
            tile_sizes(ctb_cols, tile_cols, uniform, &pps.column_width_minus1)
                .collect();
        let row_heights: Vec<u32> =
            tile_sizes(ctb_rows, tile_rows, uniform, &pps.row_height_minus1)
                .collect();
        let tile_data = &mut setup_data[TILE_SIZES_OFFSET..][..TILE_SIZES_B];
        for (i, (h, w)) in row_heights
            .iter()
            .flat_map(|h| col_widths.iter().map(move |w| (h, w)))
            .enumerate()
        {
            let entry = &mut tile_data[4 * i..4 * i + 4];
            entry[..2].copy_from_slice(&(*w as u16).to_le_bytes());
            entry[2..].copy_from_slice(&(*h as u16).to_le_bytes());
        }

        // SAFETY: the picture setup is plain data, of at most PIC_SETUP_B
        // bytes.
        let pic_bytes = unsafe {
            std::slice::from_raw_parts(
                (&pic as *const _nvdec_hevc_pic_s).cast::<u8>(),
                std::mem::size_of::<_nvdec_hevc_pic_s>(),
            )
        };
        setup_data[..pic_bytes.len()].copy_from_slice(pic_bytes);
        let (setup_addr, setup_map) = unsafe {
            upload_bytes_mapped(
                cmd,
                Some((setup_data.as_ptr(), SETUP_B)),
                SETUP_B.try_into().unwrap(),
            )
        }?;

        let coloc = session.mems[ffi::NVK_H265_MEM_COLOC as usize].addr;
        let filter = session.mems[ffi::NVK_H265_MEM_FILTER as usize].addr;

        let mut push = Push::new();
        push.push_mthd(clc7b0::SetApplicationId {
            id: clc7b0::SetApplicationIdId::Hevc,
        });
        push.push_mthd(clc7b0::SetControlParams {
            codec_type: clc7b0::SetControlParamsCodecType::Hevc,
            gptimer_on: 1,
            ret_error: 0,
            err_conceal_on: 1,
            error_frm_idx: 0,
            mbtimer_on: 1,
            ec_intra_frame_using_pslc: 0,
            ignore_some_fields_crc_check: 0,
            event_trace_logging_on: 0,
            all_intra_frame: 0,
            testrun_env: clc7b0::SetControlParamsTestrunEnv::Trace3DRun,
            reserved: 0,
            enable_encrypt: 0,
            encryptmode: 0,
        });
        push.push_mthd(clc7b0::SetDrvPicSetupOffset {
            offset: addr_hi(setup_addr),
        });
        push.push_mthd(clc7b0::SetInBufBaseOffset {
            offset: addr_hi(src_base),
        });
        // The pictures of the DPB slots, those the picture doesn't reference
        // being the picture itself, as NVIDIA's driver does: the last one
        // after the buffers.
        let picture = |i: usize| refs[i].as_ref().map_or(&dst, |r| &r.picture);
        for i in 0..NO_SLOT_INDEX {
            let p = picture(i);
            push.push_array_mthd(
                i,
                clc7b0::SetPictureLumaOffset { offset: p.luma() },
            );
            push.push_array_mthd(
                i,
                clc7b0::SetPictureChromaOffset { offset: p.chroma() },
            );
        }
        push.push_mthd(clc7b0::SetColocDataOffset {
            offset: addr_hi(coloc),
        });
        push.push_mthd(clc7b0::SetNvdecStatusOffset {
            offset: addr_hi(setup_addr + STATUS_OFFSET as u64),
        });
        let picture_index =
            push.push_mthd_patchable(clc7b0::SetPictureIndex { index: 0 });
        push.push_mthd(clc7b0::HevcSetScalingListOffset {
            offset: addr_hi(setup_addr + SCALING_LIST_OFFSET as u64),
        });
        push.push_mthd(clc7b0::HevcSetTileSizesOffset {
            offset: addr_hi(setup_addr + TILE_SIZES_OFFSET as u64),
        });
        push.push_mthd(clc7b0::HevcSetFilterBufferOffset {
            offset: addr_hi(filter),
        });
        push.push_mthd(clc7b0::SetIntraTopBufOffset {
            offset: addr_hi(filter + layout.intra_top_offset_B),
        });
        let last = picture(NO_SLOT_INDEX);
        push.push_array_mthd(
            NO_SLOT_INDEX,
            clc7b0::SetPictureLumaOffset {
                offset: last.luma(),
            },
        );
        push.push_array_mthd(
            NO_SLOT_INDEX,
            clc7b0::SetPictureChromaOffset {
                offset: last.chroma(),
            },
        );
        push.push_mthd(clc7b0::Execute {
            notify: clc7b0::ExecuteNotify::Disable,
            notify_on: clc7b0::ExecuteNotifyOn::End,
            predication: clc7b0::ExecutePredication::Disable,
            predication_op: clc7b0::ExecutePredicationOp::EqualZero,
            awaken: clc7b0::ExecuteAwaken::Disable,
        });
        let start = nvk_cmd_buffer_push(cmd, push);

        add_submit_fixup(
            cmd,
            Box::new(PictureFixup {
                state: self.state.clone(),
                picture_index: unsafe { start.add(picture_index) },
                pic_map: setup_map.cast(),
                long_term,
            }),
        );

        Ok(())
    }
}

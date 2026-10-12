// SPDX-License-Identifier: MIT

//! VP9 decode, on the decoders that parse VP9's compressed header and adapt
//! its probabilities themselves (`VP9_WITH_PARSER`, from Ampere on).
//!
//! For each frame, the driver gives the decoder:
//!
//! - a picture setup (`nvdec_vp9_pic_s`): the frame header as Vulkan gives it,
//!   the sizes of the references, and what it needs of the frame decoded
//!   before;
//! - a parser setup ([`ParserSetup`]): where the compressed header is, the
//!   segmentation probabilities, and how the frame uses the frame contexts;
//! - the size of each tile in superblocks;
//! - the session's memory: the probability tables (the 4 frame contexts the
//!   decoder loads, adapts and saves, and the defaults it resets them from),
//!   its symbol counters, its state of the previous picture, two segmentation
//!   maps and two sets of co-located motion vectors (one of the previous
//!   picture, the other written for the next), and the loop filter buffer.
//!
//! The decoder itself fills in the transform mode and the compound prediction
//! mode from the compressed header, and where the tiles start.
//!
//! How NVIDIA's driver programs the decoder was found by recording what it
//! has it do for the same streams on an RTX 3070 (GA104).

use nv_push_rs::*;
use nvidia_headers::classes::clc7b0::mthd as clc7b0;

use crate::ffi;
use crate::ffi::{
    _nvdec_vp9_pic_s, nvdec_vp9EntropyProbs_t, nvk_cmd_buffer, nvk_image,
    StdVideoDecodeVP9PictureInfo, VkVideoDecodeVP9PictureInfoKHR,
};
use crate::nvk_cmd_buffer_push;
use crate::util::vk_find_struct_const;
use crate::video::decode::VideoDecoder;

use super::vp9_default_probs::*;
use super::{addr_hi, plane_layer_base, slot_key, upload_bytes};

/// VP9 superblocks are 64x64 pixels.
const SB_SIZE: u32 = 64;

/// The references of a VP9 frame: LAST, GOLDEN and ALTREF.
const REFS: usize = 3;

/// The decoder's timeout for a frame, in cycles, which it needs to be told: a
/// wedged decode otherwise hangs the engine. NVIDIA's driver gives 36000000
/// cycles up to 1280x720, and 39.0625 cycles per pixel above.
fn timeout(width: u32, height: u32) -> u32 {
    let per_pixel = u64::from(width) * u64::from(height) * 625 / 16;
    per_pixel.max(36_000_000).try_into().unwrap_or(u32::MAX)
}

/// The parser setup, which `VP9_PARSER_SET_PIC_SETUP_OFFSET` points at.
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct ParserSetup {
    /// Size of the compressed header in bytes.
    compressed_header_size: u32,
    /// Offset of the compressed header from the start of the frame, i.e. the
    /// size of the uncompressed header.
    compressed_header_offset: u16,
    /// The segmentation probabilities, when segmentation is enabled.
    segmentation_tree_probs: [u8; 7],
    segmentation_pred_probs: [u8; 3],
    /// `PARSER_*` flags, and the frame context index and reset.
    flags: u32,
}

/// `reset_frame_context`.
const PARSER_RESET_CONTEXT_SHIFT: u32 = 0;
/// `frame_context_idx`.
const PARSER_CONTEXT_IDX_SHIFT: u32 = 2;
/// The references' sign biases differ: compound prediction is allowed.
const PARSER_COMPOUND_ALLOWED: u32 = 1 << 4;
/// `frame_parallel_decoding_mode`.
const PARSER_FRAME_PARALLEL: u32 = 1 << 5;
/// `refresh_frame_context`.
const PARSER_REFRESH_CONTEXT: u32 = 1 << 6;
/// `error_resilient_mode`, and the frame is the size of the previous one.
const PARSER_ERROR_RESILIENT_SAME_SIZE: u32 = 1 << 7;
/// `frame_parallel_decoding_mode`, and the frame is the size of the previous
/// one.
const PARSER_FRAME_PARALLEL_SAME_SIZE: u32 = 1 << 8;

/// What decoding a frame needs of the frame decoded before it.
#[derive(Clone, Copy, Debug)]
struct PrevFrame {
    width: u32,
    height: u32,
    key_frame: bool,
    show_frame: bool,
    /// Whether it wrote a segmentation map, which the next frame then reads.
    wrote_segmentation_map: bool,
}

/// Session data stored in the opaque `nvk_video_session::rust` pointer.
#[derive(Default, Debug)]
pub(crate) struct Decoder {
    /// The frame decoded last, in decode order.
    prev: Option<PrevFrame>,
    /// Which of the two segmentation maps the next frame reads, writing the
    /// other.
    seg_map_read: usize,
    /// Which of the two co-located motion vector buffers the next frame reads,
    /// writing the other.
    col_mv_read: usize,
    /// Frames decoded, which the decoder takes as the picture's index.
    picture_index: u32,
    /// The frame context the parser saves the previous frame's to, when it
    /// starts on the next frame.
    pending_save: Option<u8>,
}

/// A picture the decoder reads or writes.
struct Picture {
    img: *mut nvk_image,
    layer: u32,
    width: u32,
    height: u32,
}

impl Picture {
    fn new(res: &ffi::VkVideoPictureResourceInfoKHR) -> Self {
        let key = slot_key(res);
        Picture {
            img: unsafe { (*key.image_view).vk.image as *mut nvk_image },
            layer: key.layer,
            width: res.codedExtent.width,
            height: res.codedExtent.height,
        }
    }

    /// The row pitch of `plane` in bytes.
    fn stride(&self, plane: usize) -> u16 {
        let pitch =
            unsafe { (*self.img).planes[plane].nil.levels[0].row_stride_B };
        pitch.try_into().unwrap()
    }

    fn luma(&self) -> u32 {
        plane_layer_base(self.img, 0, self.layer)
    }

    fn chroma(&self) -> u32 {
        plane_layer_base(self.img, 1, self.layer)
    }
}

/// The offsets, in superblocks, of the tiles along a dimension of `sbs`
/// superblocks split in `1 << log2` tiles, as VP9 places them.
fn tile_offsets(sbs: u32, log2: u8) -> impl Iterator<Item = (u32, u32)> {
    let count = 1u32 << log2;
    (0..count).map(move |i| {
        let start = (i * sbs) >> log2;
        let end = ((i + 1) * sbs) >> log2;
        (start, end - start)
    })
}

/// The interpolation filter, as the decoder numbers them: VP9's
/// `literal_to_type`, which Vulkan follows, swaps the first two.
fn interpolation_filter(filter: ffi::StdVideoVP9InterpolationFilter) -> i8 {
    match filter {
        // EIGHTTAP_SMOOTH and EIGHTTAP
        0 => 1,
        1 => 0,
        // EIGHTTAP_SHARP, BILINEAR and SWITCHABLE
        f => f.try_into().unwrap(),
    }
}

/// The references of compound prediction, as VP9's
/// `setup_compound_reference_mode()` picks them: the one of a sign bias of its
/// own is fixed, the two others vary. Indices are reference names.
fn compound_references(sign_bias: [u8; 4]) -> (i8, [i8; 2]) {
    let (last, golden, altref) = (1, 2, 3);
    if sign_bias[last] == sign_bias[golden] {
        (altref as i8, [last as i8, golden as i8])
    } else if sign_bias[last] == sign_bias[altref] {
        (golden as i8, [last as i8, altref as i8])
    } else {
        (last as i8, [golden as i8, altref as i8])
    }
}

/// Number of 8-byte writes in a push of [`push_write`], which stays under
/// `NVK_CMD_BUFFER_MAX_PUSH`.
const WRITES_PER_PUSH: usize = 64;

/// Writes `data` to `addr`, after what the decoder was given before and before
/// what it is given next. The decoder has no method copying memory, so it
/// writes each 8 bytes as the 64-bit payload of a semaphore release.
fn push_write(cmd: &mut nvk_cmd_buffer, addr: u64, data: &[u8]) {
    assert!(addr % 8 == 0 && data.len() % 8 == 0);
    let writes = data.len() / 8;
    for (c, chunk) in data.chunks(8 * WRITES_PER_PUSH).enumerate() {
        let mut push = Push::new();
        for (i, bytes) in chunk.chunks_exact(8).enumerate() {
            let n = c * WRITES_PER_PUSH + i;
            let addr = addr + 8 * n as u64;
            let payload = u64::from_le_bytes(bytes.try_into().unwrap());
            push.push_mthd(clc7b0::SemaphoreA {
                upper: (addr >> 32) as u32,
            });
            push.push_mthd(clc7b0::SemaphoreB { lower: addr as u32 });
            push.push_mthd(clc7b0::SetSemaphorePayloadLower {
                payload_lower: payload as u32,
            });
            push.push_mthd(clc7b0::SetSemaphorePayloadUpper {
                payload_upper: (payload >> 32) as u32,
            });
            // The writes only need to land before the next decode: the last
            // one flushes them all.
            push.push_mthd(clc7b0::SemaphoreD {
                structure_size: clc7b0::SemaphoreDStructureSize::Two,
                awaken_enable: false,
                operation: clc7b0::SemaphoreDOperation::Release,
                flush_disable: n + 1 != writes,
                trap_type: clc7b0::SemaphoreDTrapType::Unconditional,
                payload_size: clc7b0::SemaphoreDPayloadSize::_64bit,
            });
        }
        nvk_cmd_buffer_push(cmd, push);
    }
}

/// VP9's default probabilities, as a probability table.
fn default_probs_bytes() -> Vec<u8> {
    let mut bytes = vec![0; ffi::NVK_VP9_PROBS_B as usize];
    nvk_video_vp9_default_probs(bytes.as_mut_ptr(), bytes.len());
    bytes
}

/// Writes VP9's default probabilities to `p`.
fn default_probs(p: &mut nvdec_vp9EntropyProbs_t) {
    for a in 0..10 {
        for l in 0..10 {
            p.kf_bmode_prob[a][l]
                .copy_from_slice(&VP9_KF_Y_MODE_PROB[a][l][..8]);
            p.kf_bmode_probB[a][l][0] = VP9_KF_Y_MODE_PROB[a][l][8];
        }
    }
    for m in 0..10 {
        p.kf_uv_mode_prob[m].copy_from_slice(&VP9_KF_UV_MODE_PROB[m][..8]);
        p.kf_uv_mode_probB[m][0] = VP9_KF_UV_MODE_PROB[m][8];
    }

    let a = &mut p.a;
    for (dst, src) in
        a.inter_mode_prob.iter_mut().zip(&DEFAULT_INTER_MODE_PROBS)
    {
        dst[..3].copy_from_slice(src);
    }
    a.intra_inter_prob = DEFAULT_INTRA_INTER_P;
    for m in 0..10 {
        a.uv_mode_prob[m].copy_from_slice(&DEFAULT_IF_UV_PROBS[m][..8]);
        a.uv_mode_probB[m][0] = DEFAULT_IF_UV_PROBS[m][8];
    }
    a.tx8x8_prob = DEFAULT_TX_PROBS_8X8;
    a.tx16x16_prob = DEFAULT_TX_PROBS_16X16;
    a.tx32x32_prob = DEFAULT_TX_PROBS_32X32;
    for i in 0..4 {
        a.sb_ymode_prob[i].copy_from_slice(&DEFAULT_IF_Y_PROBS[i][..8]);
        a.sb_ymode_probB[i][0] = DEFAULT_IF_Y_PROBS[i][8];
    }
    // Key frames and intra-only frames, then the other frames.
    for i in 0..16 {
        a.partition_prob[0][i][..3].copy_from_slice(&VP9_KF_PARTITION_PROBS[i]);
        a.partition_prob[1][i][..3]
            .copy_from_slice(&DEFAULT_PARTITION_PROBS[i]);
    }
    a.switchable_interp_prob = DEFAULT_SWITCHABLE_INTERP_PROB;
    a.comp_inter_prob = DEFAULT_COMP_INTER_P;
    a.mbskip_probs = DEFAULT_SKIP_PROBS;

    a.nmvc.joints = DEFAULT_MV_JOINTS;
    for (c, comp) in DEFAULT_MV_COMPONENTS.iter().enumerate() {
        a.nmvc.sign[c] = comp.sign;
        a.nmvc.class0[c] = comp.class0;
        a.nmvc.fp[c] = comp.fp;
        a.nmvc.class0_hp[c] = comp.class0_hp;
        a.nmvc.hp[c] = comp.hp;
        a.nmvc.classes[c] = comp.classes;
        a.nmvc.class0_fp[c] = comp.class0_fp;
        a.nmvc.bits[c] = comp.bits;
    }
    a.single_ref_prob = DEFAULT_SINGLE_REF_P;
    a.comp_ref_prob = DEFAULT_COMP_REF_P;

    let coefs = [
        (&mut a.probCoeffs, &DEFAULT_COEF_PROBS_4X4),
        (&mut a.probCoeffs8x8, &DEFAULT_COEF_PROBS_8X8),
        (&mut a.probCoeffs16x16, &DEFAULT_COEF_PROBS_16X16),
        (&mut a.probCoeffs32x32, &DEFAULT_COEF_PROBS_32X32),
    ];
    for (dst, src) in coefs {
        for plane in 0..2 {
            for r in 0..2 {
                for band in 0..6 {
                    for ctx in 0..6 {
                        dst[plane][r][band][ctx][..3]
                            .copy_from_slice(&src[plane][r][band][ctx]);
                    }
                }
            }
        }
    }
}

/// Writes VP9's default probabilities, as the decoder lays out a probability
/// table, to the `size` bytes at `dst`.
#[unsafe(no_mangle)]
pub extern "C" fn nvk_video_vp9_default_probs(dst: *mut u8, size: usize) {
    assert!(size >= std::mem::size_of::<nvdec_vp9EntropyProbs_t>());
    let mut probs: nvdec_vp9EntropyProbs_t = unsafe { std::mem::zeroed() };
    default_probs(&mut probs);
    unsafe {
        std::ptr::copy_nonoverlapping(
            (&probs as *const nvdec_vp9EntropyProbs_t).cast::<u8>(),
            dst,
            std::mem::size_of::<nvdec_vp9EntropyProbs_t>(),
        );
    }
}

/// Uploads `value` to the command buffer's upload heap, and returns its GPU
/// address.
fn upload<T>(
    cmd: &mut nvk_cmd_buffer,
    value: &T,
) -> Result<u64, ffi::VkResult> {
    let size = std::mem::size_of_val(value);
    unsafe {
        upload_bytes(
            cmd,
            Some(((value as *const T).cast(), size)),
            size.try_into().unwrap(),
        )
    }
}

impl Decoder {
    /// Finds the picture of the reference slot `slot` of `frame_info`.
    fn reference(
        frame_info: &ffi::VkVideoDecodeInfoKHR,
        slot: i32,
    ) -> Option<Picture> {
        (0..frame_info.referenceSlotCount as usize)
            .map(|i| unsafe { &*frame_info.pReferenceSlots.add(i) })
            .find(|r| r.slotIndex == slot && slot >= 0)
            .map(|r| Picture::new(unsafe { &*r.pPictureResource }))
    }
}

impl VideoDecoder for Decoder {
    fn reset(&mut self, cmd: &mut nvk_cmd_buffer) {
        *self = Decoder::default();
        // The parser's state of the previous frame goes too, with the frame
        // context it would save.
        let session = unsafe { cmd.state.video.vid.as_ref().unwrap() };
        let probs = session.mems[ffi::NVK_VP9_MEM_PROBS as usize].addr;
        push_write(
            cmd,
            probs + u64::from(ffi::NVK_VP9_PREV_PIC_OFFSET_B),
            &[0; ffi::NVK_VP9_PREV_PIC_B as usize],
        );
    }

    fn decode(
        &mut self,
        cmd: &mut nvk_cmd_buffer,
        frame_info: &ffi::VkVideoDecodeInfoKHR,
    ) -> Result<(), ffi::VkResult> {
        let vp9 = vk_find_struct_const!(
            frame_info.pNext,
            VIDEO_DECODE_VP9_PICTURE_INFO,
            KHR,
            VkVideoDecodeVP9PictureInfoKHR
        );
        let std: &StdVideoDecodeVP9PictureInfo =
            unsafe { &*vp9.pStdPictureInfo };
        let flags = &std.flags;
        let lf = unsafe { &*std.pLoopFilter };
        let seg = unsafe { std.pSegmentation.as_ref() };
        let segmentation = flags.segmentation_enabled() != 0;

        let dst = Picture::new(&frame_info.dstPictureResource);
        let (width, height) = (dst.width, dst.height);
        let key_frame = std.frame_type == ffi::STD_VIDEO_VP9_FRAME_TYPE_KEY;
        let intra_only = flags.intra_only() != 0;
        let same_size = self
            .prev
            .is_some_and(|p| p.width == width && p.height == height);

        // The segmentation maps and the co-located motion vectors of the
        // previous frame are of no use to a frame of another size.
        if !same_size {
            self.seg_map_read = 0;
            self.col_mv_read = 0;
        } else if self.prev.is_some_and(|p| p.wrote_segmentation_map) {
            self.seg_map_read ^= 1;
        }

        let refs: [Option<Picture>; REFS] = std::array::from_fn(|i| {
            Self::reference(frame_info, vp9.referenceNameSlotIndices[i])
        });

        let sb_cols = width.div_ceil(SB_SIZE);
        let sb_rows = height.div_ceil(SB_SIZE);
        let tile_cols = 1u32 << std.tile_cols_log2;
        // Units of 256 bytes of the loop filter buffer per superblock row and
        // boundary between tile columns.
        let filter_rows = sb_rows * (tile_cols - 1).max(1);

        let mut pic: _nvdec_vp9_pic_s = unsafe { std::mem::zeroed() };
        pic.stream_len = frame_info.srcBufferRange.try_into().unwrap();
        pic.gptimer_timeout_value = timeout(width, height);
        pic.set_tileformat(1);
        let y_log2 =
            unsafe { (*dst.img).planes[0].nil.levels[0].tiling.y_log2 };
        pic.set_gob_height((y_log2 - 1).into());
        pic.Vp9BsdCtrlOffset = 12 * filter_rows;
        pic.v1.Vp9FltAboveOffset = 13 * filter_rows;
        pic.v1.set_bit_depth(8);

        let ref_sizes = [
            (
                &mut pic.ref0_width,
                &mut pic.ref0_height,
                &mut pic.ref0_stride,
            ),
            (
                &mut pic.ref1_width,
                &mut pic.ref1_height,
                &mut pic.ref1_stride,
            ),
            (
                &mut pic.ref2_width,
                &mut pic.ref2_height,
                &mut pic.ref2_stride,
            ),
        ];
        for ((w, h, stride), r) in ref_sizes.into_iter().zip(&refs) {
            if let Some(r) = r {
                *w = r.width.try_into().unwrap();
                *h = r.height.try_into().unwrap();
                *stride = [r.stride(0), r.stride(1)];
            }
        }
        pic.width = width.try_into().unwrap();
        pic.height = height.try_into().unwrap();
        pic.framestride = [dst.stride(0), dst.stride(1)];

        pic.set_keyFrame(key_frame.into());
        pic.set_intraOnly(intra_only.into());
        pic.set_errorResilient(
            flags.error_resilient_mode().try_into().unwrap(),
        );
        pic.set_resolutionChange((!same_size).into());
        pic.set_prevIsKeyFrame(
            (same_size && self.prev.is_some_and(|p| p.key_frame)).into(),
        );
        pic.set_prevShowFrame(
            (same_size && self.prev.is_some_and(|p| p.show_frame)).into(),
        );

        let sign_bias: [u8; 4] =
            std::array::from_fn(|i| (std.ref_frame_sign_bias_mask >> i) & 1);
        pic.refFrameSignBias = sign_bias;
        let compound_allowed =
            sign_bias[1..].iter().any(|&s| s != sign_bias[1]);
        if compound_allowed {
            let (fixed, var) = compound_references(sign_bias);
            pic.comp_fixed_ref = fixed;
            pic.comp_var_ref = var;
        }

        pic.loopFilterLevel = lf.loop_filter_level.try_into().unwrap();
        pic.loopFilterSharpness = lf.loop_filter_sharpness.try_into().unwrap();
        pic.modeRefLfEnabled =
            lf.flags.loop_filter_delta_enabled().try_into().unwrap();
        pic.mbRefLfDelta = lf.loop_filter_ref_deltas;
        pic.mbModeLfDelta = lf.loop_filter_mode_deltas;

        pic.qpYAc = std.base_q_idx;
        pic.qpYDc = std.delta_q_y_dc;
        pic.qpChAc = std.delta_q_uv_ac;
        pic.qpChDc = std.delta_q_uv_dc;
        pic.lossless = (std.base_q_idx == 0
            && std.delta_q_y_dc == 0
            && std.delta_q_uv_ac == 0
            && std.delta_q_uv_dc == 0)
            .into();
        pic.allow_high_precision_mv =
            flags.allow_high_precision_mv().try_into().unwrap();
        pic.mcomp_filter_type = interpolation_filter(std.interpolation_filter);
        pic.log2_tile_columns = std.tile_cols_log2.try_into().unwrap();
        pic.log2_tile_rows = std.tile_rows_log2.try_into().unwrap();

        let mut parser = ParserSetup {
            compressed_header_size: vp9.tilesOffset
                - vp9.compressedHeaderOffset,
            compressed_header_offset: (vp9.compressedHeaderOffset
                - vp9.uncompressedHeaderOffset)
                .try_into()
                .unwrap(),
            ..Default::default()
        };

        if let Some(seg) = seg.filter(|_| segmentation) {
            pic.segmentEnabled = 1;
            pic.segmentMapUpdate =
                seg.flags.segmentation_update_map().try_into().unwrap();
            pic.segmentMapTemporalUpdate =
                seg.flags.segmentation_temporal_update().try_into().unwrap();
            pic.segmentFeatureMode = seg
                .flags
                .segmentation_abs_or_delta_update()
                .try_into()
                .unwrap();
            for i in 0..8 {
                for k in 0..4 {
                    pic.segmentFeatureEnable[i][k] =
                        (seg.FeatureEnabled[i] >> k) & 1;
                }
            }
            pic.segmentFeatureData = seg.FeatureData;
            parser.segmentation_tree_probs = seg.segmentation_tree_probs;
            parser.segmentation_pred_probs = seg.segmentation_pred_prob;
        }

        let error_resilient = flags.error_resilient_mode() != 0;
        let frame_parallel = flags.frame_parallel_decoding_mode() != 0;
        let refresh_context = flags.refresh_frame_context() != 0;
        // Key frames, intra-only frames and error resilient frames decode with
        // frame context 0, and save it, whatever frame_context_idx says: it
        // only says which frame context an intra-only frame resets
        // (setup_past_independence() in libvpx).
        let context_idx = if key_frame || intra_only || error_resilient {
            0
        } else {
            std.frame_context_idx
        };
        // The parser resets the frame contexts of key frames and error
        // resilient frames, and of intra-only frames resetting them all, but
        // not right the one an intra-only frame resets alone: GA10x's decoder
        // (GSP-RM r570) then writes the first half of the defaults to frame
        // context 2 and the second half to frame context 0, whatever the
        // frame's. NVK resets that one itself, see reset_context().
        let parser_reset: u8 = if intra_only && std.reset_frame_context == 3 {
            3
        } else {
            0
        };
        let reset_context = (intra_only && std.reset_frame_context == 2)
            .then_some(std.frame_context_idx);
        parser.flags = u32::from(parser_reset) << PARSER_RESET_CONTEXT_SHIFT
            | u32::from(context_idx) << PARSER_CONTEXT_IDX_SHIFT;
        if compound_allowed {
            parser.flags |= PARSER_COMPOUND_ALLOWED;
        }
        if frame_parallel {
            parser.flags |= PARSER_FRAME_PARALLEL;
        }
        if refresh_context {
            parser.flags |= PARSER_REFRESH_CONTEXT;
        }
        if error_resilient && same_size {
            parser.flags |= PARSER_ERROR_RESILIENT_SAME_SIZE;
        }
        if frame_parallel && same_size {
            parser.flags |= PARSER_FRAME_PARALLEL_SAME_SIZE;
        }

        // The size of each tile, a row of tiles after the other.
        let mut tiles = Vec::new();
        for (_, rows) in tile_offsets(sb_rows, std.tile_rows_log2) {
            for (_, cols) in tile_offsets(sb_cols, std.tile_cols_log2) {
                tiles.push(rows << 16 | cols);
            }
        }

        let session = unsafe { cmd.state.video.vid.as_ref().unwrap() };
        let max_sbs = u64::from(session.vk.max_coded.width.div_ceil(SB_SIZE))
            * u64::from(session.vk.max_coded.height.div_ceil(SB_SIZE));
        let seg_mv = session.mems[ffi::NVK_VP9_MEM_SEG_MV as usize].addr;
        let buf_align = u64::from(ffi::NVK_VP9_BUF_ALIGN_B);
        let seg_map_size = (max_sbs * u64::from(ffi::NVK_VP9_SEG_MAP_SB_B))
            .next_multiple_of(buf_align);
        let col_mv_size = (max_sbs * u64::from(ffi::NVK_VP9_COL_MV_SB_B))
            .next_multiple_of(buf_align);
        let seg_map = |i: usize| seg_mv + i as u64 * seg_map_size;
        let col_mv =
            |i: usize| seg_mv + 2 * seg_map_size + i as u64 * col_mv_size;
        let intra_top = session.mems[ffi::NVK_VP9_MEM_FILTER as usize].addr;
        let filter = intra_top + u64::from(ffi::NVK_VP9_INTRA_TOP_B);
        let probs = session.mems[ffi::NVK_VP9_MEM_PROBS as usize].addr;
        // The parser leaves the frame's probabilities, the frame context's
        // with the compressed header's updates, in the frame context when it
        // is saved as is, and in the current frame's table otherwise: when it
        // isn't saved, or adapted once the frame is decoded.
        let table = |i: u32| probs + u64::from(i * ffi::NVK_VP9_PROBS_B);
        let frame_probs = if refresh_context && frame_parallel {
            table(context_idx.into())
        } else {
            table(ffi::NVK_VP9_CURRENT_PROBS)
        };
        let counters = probs + u64::from(ffi::NVK_VP9_COUNTERS_OFFSET_B);
        let prev_pic = probs + u64::from(ffi::NVK_VP9_PREV_PIC_OFFSET_B);

        let src_buffer =
            unsafe { ffi::nvk_buffer_from_handle(frame_info.srcBuffer) };
        let src = unsafe {
            ffi::vk_buffer_address(
                &(*src_buffer).vk,
                frame_info.srcBufferOffset
                    + u64::from(vp9.uncompressedHeaderOffset),
            )
        };

        let pic_addr = upload(cmd, &pic)?;
        let parser_addr = upload(cmd, &parser)?;
        let tiles_addr = unsafe {
            upload_bytes(
                cmd,
                Some((tiles.as_ptr().cast(), tiles.len() * 4)),
                (tiles.len() * 4).try_into().unwrap(),
            )
        }?;
        // Where the decoder writes the frame's status.
        let status_addr = unsafe { upload_bytes(cmd, None, 512) }?;

        // The methods decoding the frame, with the parser setup and the
        // probability table the decoder writes the frame's probabilities to.
        let decode = |parser_addr: u64, frame_probs: u64| {
            let mut push = Push::new();

            push.push_mthd(clc7b0::SetApplicationId {
                id: clc7b0::SetApplicationIdId::Vp9WithParser,
            });
            // NVIDIA's driver sets the codec type field to 1 with VP9_WITH_PARSER,
            // not to VP9's.
            push.push_mthd(clc7b0::SetControlParams {
                codec_type: clc7b0::SetControlParamsCodecType::Mpeg2,
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
                offset: addr_hi(pic_addr),
            });
            push.push_mthd(clc7b0::SetInBufBaseOffset {
                offset: addr_hi(src),
            });
            // Frames without references, key frames, are given the frame itself.
            for (i, r) in refs.iter().enumerate() {
                let r = r.as_ref().unwrap_or(&dst);
                push.push_array_mthd(
                    i,
                    clc7b0::SetPictureLumaOffset { offset: r.luma() },
                );
                push.push_array_mthd(
                    i,
                    clc7b0::SetPictureChromaOffset { offset: r.chroma() },
                );
            }
            push.push_mthd(clc7b0::SetNvdecStatusOffset {
                offset: addr_hi(status_addr),
            });
            push.push_mthd(clc7b0::SetPictureIndex {
                index: self.picture_index,
            });
            push.push_mthd(clc7b0::Vp9ParserSetPicSetupOffset {
                offset: addr_hi(parser_addr),
            });
            push.push_mthd(clc7b0::Vp9ParserSetProbTabBufOffset {
                offset: addr_hi(probs),
            });
            push.push_mthd(clc7b0::Vp9ParserSetPrevPicSetupOffset {
                offset: addr_hi(prev_pic),
            });
            push.push_mthd(clc7b0::Vp9SetProbTabBufOffset {
                offset: addr_hi(frame_probs),
            });
            push.push_mthd(clc7b0::Vp9SetCtxCounterBufOffset {
                offset: addr_hi(counters),
            });
            push.push_mthd(clc7b0::Vp9SetTileSizeBufOffset {
                offset: addr_hi(tiles_addr),
            });
            push.push_mthd(clc7b0::Vp9SetSegmentReadBufOffset {
                offset: addr_hi(seg_map(self.seg_map_read)),
            });
            push.push_mthd(clc7b0::Vp9SetSegmentWriteBufOffset {
                offset: addr_hi(seg_map(self.seg_map_read ^ 1)),
            });
            push.push_mthd(clc7b0::Vp9SetColMvwriteBufOffset {
                offset: addr_hi(col_mv(self.col_mv_read ^ 1)),
            });
            push.push_mthd(clc7b0::Vp9SetColMvreadBufOffset {
                offset: addr_hi(col_mv(self.col_mv_read)),
            });
            push.push_mthd(clc7b0::Vp9SetFilterBufferOffset {
                offset: addr_hi(filter),
            });
            push.push_mthd(clc7b0::SetIntraTopBufOffset {
                offset: addr_hi(intra_top),
            });
            // The picture decoded, after the references.
            push.push_array_mthd(
                REFS,
                clc7b0::SetPictureLumaOffset { offset: dst.luma() },
            );
            push.push_array_mthd(
                REFS,
                clc7b0::SetPictureChromaOffset {
                    offset: dst.chroma(),
                },
            );
            push.push_mthd(clc7b0::Execute {
                notify: clc7b0::ExecuteNotify::Disable,
                notify_on: clc7b0::ExecuteNotifyOn::End,
                predication: clc7b0::ExecutePredication::Disable,
                predication_op: clc7b0::ExecutePredicationOp::EqualZero,
                awaken: clc7b0::ExecuteAwaken::Disable,
            });
            push
        };

        // The frame context an intra-only frame resets alone: the frame
        // context the previous frame refreshed is saved when the parser
        // starts on the next frame (adapted to what the previous frame
        // decoded), so the defaults have to be written after that, and before
        // the frame loads frame context 0.
        let mut reset_after = None;
        if let Some(reset) = reset_context {
            let reset_addr = table(reset.into());
            if self.pending_save != Some(reset) {
                push_write(cmd, reset_addr, &default_probs_bytes());
            } else if reset != 0 {
                // The frame doesn't load the frame context it resets.
                reset_after = Some(reset_addr);
            } else {
                // The frame loads the frame context it resets, which the
                // parser saves the previous frame's to first: a first pass
                // over the frame, which saves nothing itself, has it do that.
                let first = ParserSetup {
                    flags: parser.flags & !PARSER_REFRESH_CONTEXT,
                    ..parser
                };
                let first_addr = upload(cmd, &first)?;
                nvk_cmd_buffer_push(
                    cmd,
                    decode(first_addr, table(ffi::NVK_VP9_CURRENT_PROBS)),
                );
                push_write(cmd, reset_addr, &default_probs_bytes());
            }
        }

        let push = decode(parser_addr, frame_probs);
        nvk_cmd_buffer_push(cmd, push);
        if let Some(reset_addr) = reset_after {
            push_write(cmd, reset_addr, &default_probs_bytes());
        }

        self.pending_save = refresh_context.then_some(context_idx);
        self.col_mv_read ^= 1;
        self.prev = Some(PrevFrame {
            width,
            height,
            key_frame,
            show_frame: flags.show_frame() != 0,
            wrote_segmentation_map: segmentation
                && seg.is_some_and(|s| s.flags.segmentation_update_map() != 0),
        });
        self.picture_index = self.picture_index.wrapping_add(1);

        Ok(())
    }
}

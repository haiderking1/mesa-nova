// SPDX-License-Identifier: MIT

//! AV1 decode, on the decoders of Ampere on (`NVC7B0_VIDEO_DECODER`).
//!
//! For each frame, the driver gives the decoder:
//!
//! - a picture setup (`nvdec_av1_pic_s`): the frame header as Vulkan gives it,
//!   with what the decoder needs of the references (their size, scale and
//!   order) and of the motion field projection, which it does itself;
//! - the tiles: their size in superblocks and where they are in the
//!   bitstream;
//! - the global motion of each reference, with its warp shear;
//! - the CDFs it starts from, the defaults or those a reference saved, and
//!   where it saves the frame's;
//! - the temporal motion vectors and the segment ids the references saved,
//!   and where it saves the frame's;
//! - the loop filter, CDEF, loop restoration and superres buffers.
//!
//! How NVIDIA's driver programs the decoder was found in its CUVID library
//! (libnvcuvid 615.71.09), which does it for AV1 as the Vulkan driver does.

use std::sync::{Arc, Mutex};

use nv_push_rs::*;
use nvidia_headers::classes::clc7b0::mthd as clc7b0;

use crate::ffi;
use crate::ffi::{
    _nvdec_av1_pic_s, nvk_cmd_buffer, StdVideoDecodeAV1PictureInfo,
    StdVideoDecodeAV1ReferenceInfo, VkVideoDecodeAV1PictureInfoKHR,
};
use crate::nvk_cmd_buffer_push;
use crate::util::vk_find_struct_const;
use crate::video::decode::VideoDecoder;

use super::{
    add_submit_fixup, addr_hi, slot_key, upload_bytes_mapped, Picture,
    SubmitFixup,
};
use super::{av1_film_grain, av1_probs};

/// The references of a frame, LAST_FRAME to ALTREF_FRAME.
const REFS: usize = 7;
/// The reference names, as AV1 numbers them.
const LAST_FRAME: usize = 1;
const LAST2_FRAME: usize = 2;
const GOLDEN_FRAME: usize = 4;
const BWDREF_FRAME: usize = 5;
const ALTREF2_FRAME: usize = 6;
const ALTREF_FRAME: usize = 7;

/// The DPB slots, and the slot of the motion vectors and the segment ids of
/// frames which aren't references.
const DPB_SLOTS: usize = 9;
const SCRATCH_SLOT: usize = DPB_SLOTS;

/// The decoder takes the addresses of its buffers in units of 256 bytes.
const BUF_ALIGN: u64 = 256;

/// Bytes of the picture setup, then of the tile sizes, the global motion and
/// the film grain, which the decoder takes from one buffer: then 512 bytes of
/// status.
const PIC_SETUP_B: usize = 0x300;
const TILES_OFFSET: usize = 0x300;
const TILE_B: usize = 16;
const MAX_TILES: usize = 256;
const GLOBAL_MOTION_OFFSET: usize = 0x1300;
const FILM_GRAIN_OFFSET: usize = 0x1400;
const STATUS_OFFSET: usize = 0x4700;
const SETUP_B: usize = 0x4900;

/// The bytes of the intra prediction top line, after the filter buffers.
const INTRA_TOP_B: u64 = 0x10000;

fn align(v: u64, a: u64) -> u64 {
    v.next_multiple_of(a)
}

/// Where the buffers of the session memory of an AV1 session are, and how
/// big they are, for pictures of up to `width` x `height`. Each DPB slot, and
/// the scratch slot, has the temporal motion vectors and then the segment
/// ids of its frame.
#[repr(C)]
#[allow(non_camel_case_types)]
pub struct nvk_video_av1_mem_layout {
    pub mv_B: u64,
    pub seg_B: u64,
    pub slot_B: u64,
    pub zero_seg_offset_B: u64,
    pub mv_seg_mem_B: u64,
    /// The filter buffers, in units of 256 bytes from the start of the
    /// memory: loop filter top and vertical, CDEF, superres, loop
    /// restoration and film grain.
    pub flt_top: u32,
    pub flt_vert: u32,
    pub cdef_vert: u32,
    pub hus_vert: u32,
    pub lr_vert: u32,
    pub fgs_vert: u32,
    pub intra_top_offset_B: u64,
    pub filter_mem_B: u64,
}

#[unsafe(no_mangle)]
pub extern "C" fn nvk_video_av1_mem_layout(
    width: u32,
    height: u32,
) -> nvk_video_av1_mem_layout {
    let (w, h) = (u64::from(width), u64::from(height));

    // 4 bytes of motion vectors per 8x8 block, 4 bits of segment id per 4x4
    // block, of the picture in 128x128 superblocks.
    let area = align(w, 128) * align(h, 128);
    let mv_B = align(area / 16, 4096);
    let seg_B = align(area / 32, 4096);
    let slot_B = align(mv_B + seg_B, BUF_ALIGN);
    let zero_seg_offset_B = slot_B * (DPB_SLOTS as u64 + 1);

    // Per 64x64 superblock column and row, and two more.
    let sbw = w.div_ceil(64) + 2;
    let sbh = h.div_ceil(64) + 2;
    let flt_vert_B = align(sbw * 1920, BUF_ALIGN);
    let cdef_vert_B = flt_vert_B + sbh * 1536;
    let hus_vert_B = cdef_vert_B + sbh * 1792;
    // Two superres buffers, written in turn.
    let lr_vert_B = hus_vert_B + sbh * 2 * 12544;
    let fgs_vert_B = lr_vert_B + align(sbh * 1920, BUF_ALIGN);
    let filter_B = fgs_vert_B + align(sbh * 384, BUF_ALIGN);
    let units = |b: u64| u32::try_from(b / BUF_ALIGN).unwrap();

    nvk_video_av1_mem_layout {
        mv_B,
        seg_B,
        slot_B,
        zero_seg_offset_B,
        mv_seg_mem_B: zero_seg_offset_B + seg_B,
        flt_top: 0,
        flt_vert: units(flt_vert_B),
        cdef_vert: units(cdef_vert_B),
        hus_vert: units(hus_vert_B),
        lr_vert: units(lr_vert_B),
        fgs_vert: units(fgs_vert_B),
        intra_top_offset_B: filter_B,
        filter_mem_B: filter_B + INTRA_TOP_B,
    }
}

/// Writes the default probability table of coefficient CDF context `q_ctx`
/// to the `size` bytes at `dst`.
#[unsafe(no_mangle)]
pub extern "C" fn nvk_video_av1_default_probs(
    q_ctx: u32,
    dst: *mut u8,
    size: usize,
) {
    assert!(size >= av1_probs::PROBS_B);
    let mut table = [0; av1_probs::PROBS_B];
    av1_probs::default_probs(q_ctx.try_into().unwrap(), &mut table);
    unsafe {
        std::ptr::copy_nonoverlapping(table.as_ptr(), dst, table.len());
    }
}

/// A probability table of the session memory.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Probs {
    /// The default one of a coefficient CDF context.
    Default(usize),
    /// One of the pool.
    Pool(usize),
}

/// The session's state of the frames decoded, in the order the decoder
/// decodes them, which [`FrameFixup`]s keep up to date.
#[derive(Default, Debug)]
struct DecodeState {
    /// The probability table each DPB slot's frame saved its CDFs to.
    probs: [Option<Probs>; DPB_SLOTS],
    /// The width each DPB slot's frame was coded at, before superres
    /// upscaling.
    coded_width: [Option<u32>; DPB_SLOTS],
    /// Frames decoded, which the decoder takes as the picture's index.
    picture_index: u32,
}

/// Session data stored in the opaque `nvk_video_session::rust` pointer.
#[derive(Default, Debug)]
pub(crate) struct Decoder {
    state: Arc<Mutex<DecodeState>>,
}

/// Forgets the frames decoded before, when the command buffer is submitted.
struct ResetFixup {
    state: Arc<Mutex<DecodeState>>,
}

impl SubmitFixup for ResetFixup {
    fn submit(&mut self) {
        *self.state.lock().unwrap() = DecodeState::default();
    }
}

/// What choosing the motion field projections and the segment ids a frame
/// predicts its own from needs of a reference.
#[derive(Clone, Copy)]
struct RefMotion {
    slot: usize,
    intra: bool,
    order_hint: u8,
    saved_order_hints: [u8; 8],
    /// The size of the picture, after superres upscaling.
    width: u32,
    height: u32,
    segmentation: bool,
}

/// The motion field projections and the segment ids a frame predicts its
/// own from, which depend on the size its references were coded at (before
/// superres upscaling), which only the frames decoded before tell.
struct MotionSetup {
    refs: [Option<RefMotion>; REFS],
    /// The reference the segment ids are predicted from.
    primary: Option<usize>,
    intra: bool,
    order_hint: u8,
    /// The size the frame is coded at.
    width: u32,
    height: u32,
    use_ref_frame_mvs: bool,
    segmentation: bool,
    enable_order_hint: bool,
    order_hint_bits: u32,
}

/// What [`MotionSetup::choose`] chose: the DPB slots of the projections,
/// and the slot of the segment ids predicted from, if any.
struct MotionChoice {
    projections: [Option<usize>; 3],
    segment_ids: Option<(usize, bool)>,
}

impl MotionSetup {
    /// get_relative_dist() of the AV1 specification.
    fn dist(&self, a: u8, b: u8) -> i32 {
        if !self.enable_order_hint {
            return 0;
        }
        let m = 1i32 << (self.order_hint_bits - 1);
        let diff = i32::from(a) - i32::from(b);
        (diff & (m - 1)) - (diff & m)
    }

    /// Chooses the projections and the segment ids predicted from, given
    /// the width each reference was coded at, and sets them in `pic`.
    fn choose(
        &self,
        coded_width: impl Fn(&RefMotion) -> u32,
        pic: &mut _nvdec_av1_pic_s,
    ) -> MotionChoice {
        let reference = |name: usize| self.refs[name - LAST_FRAME].as_ref();
        let same_size = |r: &RefMotion| {
            (coded_width(r) + 7) >> 3 == (self.width + 7) >> 3
                && (r.height + 7) >> 3 == (self.height + 7) >> 3
        };

        // As libaom's av1_setup_motion_field() does: a reference projects
        // if it isn't intra and is of the frame's size.
        let mut names = Vec::new();
        if !self.intra {
            let projects = |name: usize| {
                reference(name).is_some_and(|r| !r.intra && same_size(r))
            };
            let ahead = |name: usize| {
                reference(name).is_some_and(|r| {
                    self.dist(r.order_hint, self.order_hint) > 0
                })
            };
            let mut stamp = 2i32;
            if let Some(last) = reference(LAST_FRAME) {
                let alt_of_last = last.saved_order_hints[ALTREF_FRAME];
                let golden =
                    reference(GOLDEN_FRAME).map_or(0, |g| g.order_hint);
                if alt_of_last != golden && projects(LAST_FRAME) {
                    names.push(LAST_FRAME);
                }
                stamp -= 1;
            }
            for name in [BWDREF_FRAME, ALTREF2_FRAME] {
                if ahead(name) && projects(name) {
                    names.push(name);
                    stamp -= 1;
                }
            }
            if ahead(ALTREF_FRAME) && stamp >= 0 && projects(ALTREF_FRAME) {
                names.push(ALTREF_FRAME);
                stamp -= 1;
            }
            if stamp >= 0 && projects(LAST2_FRAME) {
                names.push(LAST2_FRAME);
            }
            names.truncate(3);
        }

        let mut projections = [None; 3];
        let mut mfmv = 0u32;
        pic.mfmv_offset = [[0; REFS]; 3];
        pic.mfmv_side = [[0; REFS]; 3];
        for (k, &name) in names.iter().enumerate() {
            let src = reference(name).unwrap();
            projections[k] = Some(src.slot);
            mfmv |= (name as u32) << (3 + 3 * k);
            let d = self.dist(self.order_hint, src.order_hint);
            if self.use_ref_frame_mvs && (-31..=31).contains(&d) {
                mfmv |= 1 << k;
                for j in 0..REFS {
                    let saved = src.saved_order_hints[LAST_FRAME + j];
                    pic.mfmv_offset[k][j] =
                        self.dist(src.order_hint, saved) as i16;
                    pic.mfmv_side[k][j] =
                        (self.dist(saved, src.order_hint) > 0) as i8;
                }
            }
        }
        pic.set_use_temporal0_mvs(mfmv & 1);
        pic.set_use_temporal1_mvs((mfmv >> 1) & 1);
        pic.set_use_temporal2_mvs((mfmv >> 2) & 1);
        pic.set_mf1_type((mfmv >> 3) & 7);
        pic.set_mf2_type((mfmv >> 6) & 7);
        pic.set_mf3_type((mfmv >> 9) & 7);

        // The segment ids of the primary reference frame, if of the
        // frame's size: zeros if that frame had segmentation disabled.
        let segment_ids = self
            .primary
            .and_then(|name| self.refs[name].as_ref())
            .filter(|p| self.segmentation && same_size(p))
            .map(|p| (p.slot, p.segmentation));
        pic.prevsegid_flag = segment_ids.is_some().into();

        MotionChoice {
            projections,
            segment_ids,
        }
    }
}

/// The temporal motion vectors and segment ids of the session memory.
#[derive(Clone, Copy)]
struct MvSegs {
    addr: u64,
    slot_B: u64,
    mv_B: u64,
    zero_seg_offset_B: u64,
}

impl MvSegs {
    fn mv(&self, slot: usize) -> u64 {
        self.addr + slot as u64 * self.slot_B
    }

    fn segs(&self, slot: usize) -> u64 {
        self.mv(slot) + self.mv_B
    }

    fn zero_segs(&self) -> u64 {
        self.addr + self.zero_seg_offset_B
    }
}

/// The parts of a frame's decode which depend on the frames decoded before
/// it, filled in when the command buffer is submitted: its probability
/// tables, its motion field projections and segment ids predicted from, and
/// its picture index.
struct FrameFixup {
    state: Arc<Mutex<DecodeState>>,
    /// The probability table the frame starts from: a default one, or the
    /// one the DPB slot of the frame its CDFs come from saved.
    read: ReadProbs,
    /// The DPB slot the frame is decoded to, if it is a reference, and the
    /// slot of its motion vectors and segment ids.
    slot: Option<usize>,
    dst_slot: usize,
    /// Whether the frame saves the CDFs it starts from, not those it ends
    /// with.
    keep_read: bool,
    motion: MotionSetup,
    /// The probability tables, and the motion vectors and segment ids, of
    /// the session memory.
    probs_addr: u64,
    mv_segs: MvSegs,
    /// The picture setup, and where it is uploaded.
    pic: Box<_nvdec_av1_pic_s>,
    pic_map: *mut _nvdec_av1_pic_s,
    /// Where the methods depending on the frames decoded before have their
    /// data, in the pushbuffer.
    picture_index: *mut u32,
    read_probs: *mut u32,
    write_probs: *mut u32,
    col_mv_read: [*mut u32; 3],
    segment_read: *mut u32,
}

#[derive(Clone, Copy)]
enum ReadProbs {
    Default(usize),
    Slot(usize),
}

impl FrameFixup {
    fn addr(&self, probs: Probs) -> u32 {
        let index = match probs {
            Probs::Default(q) => q,
            Probs::Pool(i) => ffi::NVK_AV1_DEFAULT_PROBS as usize + i,
        };
        addr_hi(
            self.probs_addr + (index * ffi::NVK_AV1_PROBS_B as usize) as u64,
        )
    }
}

impl SubmitFixup for FrameFixup {
    fn submit(&mut self) {
        let mut state = self.state.lock().unwrap();

        let read = match self.read {
            ReadProbs::Default(q) => Probs::Default(q),
            // A reference which was never decoded has the defaults.
            ReadProbs::Slot(s) => state.probs[s].unwrap_or(Probs::Default(0)),
        };
        // The frame writes its CDFs to a table no other DPB slot's frame
        // saved its CDFs to: the pool has one more table than the DPB has
        // slots for the others.
        let others = |i: usize| {
            state.probs.iter().enumerate().any(|(s, p)| {
                Some(s) != self.slot && *p == Some(Probs::Pool(i))
            })
        };
        let write = (0..ffi::NVK_AV1_PROBS_POOL as usize)
            .find(|&i| !others(i))
            .map(Probs::Pool)
            .unwrap();

        // The width each reference was coded at, which is that of its
        // picture unless it was upscaled.
        let choice = self.motion.choose(
            |r| state.coded_width[r.slot].unwrap_or(r.width),
            &mut self.pic,
        );
        let mv_segs = self.mv_segs;
        let segment_read = match choice.segment_ids {
            Some((slot, true)) => mv_segs.segs(slot),
            _ => mv_segs.zero_segs(),
        };

        // SAFETY: the uploads and the pushbuffer are the command buffer's,
        // which the GPU doesn't use until it is submitted.
        unsafe {
            self.pic_map.write(*self.pic);
            self.picture_index.write(state.picture_index);
            self.read_probs.write(self.addr(read));
            self.write_probs.write(self.addr(write));
            // Projections not made read nothing.
            for (word, slot) in self.col_mv_read.iter().zip(choice.projections)
            {
                word.write(addr_hi(mv_segs.mv(slot.unwrap_or(self.dst_slot))));
            }
            self.segment_read.write(addr_hi(segment_read));
        }

        if let Some(slot) = self.slot {
            state.probs[slot] = Some(if self.keep_read { read } else { write });
            state.coded_width[slot] = Some(self.motion.width);
        }
        state.picture_index = state.picture_index.wrapping_add(1);
    }
}

/// A reference of the frame.
struct Reference {
    slot: usize,
    pic: Picture,
    info: StdVideoDecodeAV1ReferenceInfo,
}

impl Reference {
    fn is_intra(&self) -> bool {
        self.info.frame_type == ffi::STD_VIDEO_AV1_FRAME_TYPE_KEY as u8
            || self.info.frame_type
                == ffi::STD_VIDEO_AV1_FRAME_TYPE_INTRA_ONLY as u8
    }
}

/// The DPB slot index and the picture of each reference slot of the frame.
fn reference_slots(
    frame_info: &ffi::VkVideoDecodeInfoKHR,
) -> impl Iterator<Item = &ffi::VkVideoReferenceSlotInfoKHR> {
    (0..frame_info.referenceSlotCount as usize)
        .map(|i| unsafe { &*frame_info.pReferenceSlots.add(i) })
}

/// The reference slot `slot` of the frame, with what the frame decoded to it
/// left for the frames referencing it.
fn reference_slot(
    frame_info: &ffi::VkVideoDecodeInfoKHR,
    slot: i32,
) -> Option<Reference> {
    let r = reference_slots(frame_info)
        .find(|r| r.slotIndex == slot && slot >= 0)?;
    let dpb = vk_find_struct_const!(
        r.pNext,
        VIDEO_DECODE_AV1_DPB_SLOT_INFO,
        KHR,
        ffi::VkVideoDecodeAV1DpbSlotInfoKHR
    );
    Some(Reference {
        slot: slot.try_into().unwrap(),
        pic: Picture::new(unsafe { &*r.pPictureResource }),
        info: unsafe { *dpb.pStdReferenceInfo },
    })
}

/// The decoder's timeout for a frame, in cycles: 10000 per macroblock, of
/// at least 8100 macroblocks, as NVIDIA's driver gives it.
fn timeout(width: u32, height: u32) -> u32 {
    let mbs = u64::from(width.div_ceil(16)) * u64::from(height.div_ceil(16));
    (mbs.max(8100) * 10000).min(0x4fff_ffff).try_into().unwrap()
}

/// Div_Lut of the AV1 specification (section 7.11.3.7): 1/d in 14 bits for
/// d from 1 to 2 in 256 steps.
#[rustfmt::skip]
const DIV_LUT: [i32; 257] = [
    16384, 16320, 16257, 16194, 16132, 16070, 16009, 15948, 15888, 15828,
    15768, 15709, 15650, 15592, 15534, 15477, 15420, 15364, 15308, 15252,
    15197, 15142, 15087, 15033, 14980, 14926, 14873, 14821, 14769, 14717,
    14665, 14614, 14564, 14513, 14463, 14413, 14364, 14315, 14266, 14218,
    14170, 14122, 14075, 14028, 13981, 13935, 13888, 13843, 13797, 13752,
    13707, 13662, 13618, 13574, 13530, 13487, 13443, 13400, 13358, 13315,
    13273, 13231, 13190, 13148, 13107, 13066, 13026, 12985, 12945, 12906,
    12866, 12827, 12788, 12749, 12710, 12672, 12633, 12596, 12558, 12520,
    12483, 12446, 12409, 12373, 12336, 12300, 12264, 12228, 12193, 12157,
    12122, 12087, 12053, 12018, 11984, 11950, 11916, 11882, 11848, 11815,
    11782, 11749, 11716, 11683, 11651, 11619, 11586, 11555, 11523, 11491,
    11460, 11429, 11398, 11367, 11336, 11305, 11275, 11245, 11215, 11185,
    11155, 11125, 11096, 11067, 11038, 11009, 10980, 10951, 10923, 10894,
    10866, 10838, 10810, 10782, 10755, 10727, 10700, 10673, 10645, 10618,
    10592, 10565, 10538, 10512, 10486, 10460, 10434, 10408, 10382, 10356,
    10331, 10305, 10280, 10255, 10230, 10205, 10180, 10156, 10131, 10107,
    10082, 10058, 10034, 10010, 9986, 9963, 9939, 9916, 9892, 9869,
    9846, 9823, 9800, 9777, 9754, 9732, 9709, 9687, 9664, 9642,
    9620, 9598, 9576, 9554, 9533, 9511, 9489, 9468, 9447, 9425,
    9404, 9383, 9362, 9341, 9321, 9300, 9279, 9259, 9239, 9218,
    9198, 9178, 9158, 9138, 9118, 9098, 9079, 9059, 9039, 9020,
    9001, 8981, 8962, 8943, 8924, 8905, 8886, 8867, 8849, 8830,
    8812, 8793, 8775, 8756, 8738, 8720, 8702, 8684, 8666, 8648,
    8630, 8613, 8595, 8577, 8560, 8542, 8525, 8508, 8490, 8473,
    8456, 8439, 8422, 8405, 8389, 8372, 8355, 8339, 8322, 8306,
    8289, 8273, 8257, 8240, 8224, 8208, 8192,
];

/// Round2Signed() of the AV1 specification.
fn round2signed(x: i64, n: u32) -> i64 {
    if n == 0 {
        x
    } else if x >= 0 {
        (x + (1 << (n - 1))) >> n
    } else {
        -((-x + (1 << (n - 1))) >> n)
    }
}

/// The shear parameters of the warp model `m`, alpha, beta, gamma and delta,
/// if the model can be warped with (setupShear() of the AV1 specification).
fn shear(m: &[i32; 6]) -> Option<[i16; 4]> {
    if m[2] <= 0 {
        return None;
    }
    let clamp16 = |v: i64| v.clamp(i16::MIN.into(), i16::MAX.into());
    let alpha0 = clamp16(i64::from(m[2]) - (1 << 16));
    let beta0 = clamp16(m[3].into());

    // resolve_divisor() of the specification.
    let d = m[2] as u32;
    let n = 31 - d.leading_zeros();
    let e = i64::from(d) - (1i64 << n);
    let f = if n > 8 {
        round2signed(e, n - 8)
    } else {
        e << (8 - n)
    };
    let div_shift = n + 14;
    let div_factor = i64::from(DIV_LUT[f as usize]);

    let v = (i64::from(m[4]) << 16) * div_factor;
    let gamma0 = clamp16(round2signed(v, div_shift));
    let w = i64::from(m[3]) * i64::from(m[4]) * div_factor;
    let delta0 =
        clamp16(i64::from(m[5]) - round2signed(w, div_shift) - (1 << 16));

    let reduce = |v: i64| round2signed(v, 6) << 6;
    let [alpha, beta, gamma, delta] =
        [alpha0, beta0, gamma0, delta0].map(reduce);
    if 4 * alpha.abs() + 7 * beta.abs() >= 1 << 16
        || 4 * gamma.abs() + 4 * delta.abs() >= 1 << 16
    {
        return None;
    }
    Some([alpha, beta, gamma, delta].map(|v| v as i16))
}

/// The superres step and initial subpixel position of a plane `w` pixels
/// wide before upscaling, `up_w` after (section 7.16 of the AV1
/// specification).
fn superres(w: u32, up_w: u32) -> (u16, u16) {
    let (w, up_w) = (i64::from(w), i64::from(up_w));
    let step = ((w << 14) + up_w / 2) / up_w;
    let err = up_w * step - (w << 14);
    let init =
        ((-((up_w - w) << 13) + up_w / 2) / up_w + 128 - err / 2) & 0x3fff;
    (step as u16, init as u16)
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
        let av1 = vk_find_struct_const!(
            frame_info.pNext,
            VIDEO_DECODE_AV1_PICTURE_INFO,
            KHR,
            VkVideoDecodeAV1PictureInfoKHR
        );
        let std: &StdVideoDecodeAV1PictureInfo =
            unsafe { &*av1.pStdPictureInfo };
        let flags = &std.flags;

        let mut seq_ptr = std::ptr::null();
        unsafe {
            ffi::vk_video_get_av1_parameters(
                &(*cmd.state.video.vid).vk,
                cmd.state.video.params as *const _,
                frame_info as *const _,
                &mut seq_ptr,
            );
        }
        let seq = unsafe { &*seq_ptr };
        let color = unsafe { &*seq.pColorConfig };

        let frame_type = std.frame_type;
        let key_frame = frame_type == ffi::STD_VIDEO_AV1_FRAME_TYPE_KEY;
        // KEY_FRAME and INTRA_ONLY_FRAME.
        let intra = frame_type & 1 == 0;
        let allow_intrabc = flags.allow_intrabc() != 0;

        // The frame is decoded to the picture of its DPB slot, or to the
        // destination picture when it isn't a reference.
        let setup = unsafe { frame_info.pSetupReferenceSlot.as_ref() };
        let slot = setup.map(|s| usize::try_from(s.slotIndex).unwrap());
        let dst = Picture::new(match setup {
            Some(s) => unsafe { &*s.pPictureResource },
            None => &frame_info.dstPictureResource,
        });
        // The picture output, when it isn't the reconstructed one: the decoder
        // outputs the frame with its film grain to it.
        let display = setup
            .filter(|s| {
                slot_key(unsafe { &*s.pPictureResource })
                    != slot_key(&frame_info.dstPictureResource)
            })
            .map(|_| Picture::new(&frame_info.dstPictureResource));

        // The frame's size: its width before superres upscaling, `w`, and
        // after, `up_w`, which is that of the picture.
        let up_w = dst.width;
        let h = dst.height;
        let use_superres = flags.use_superres() != 0;
        let w = if use_superres {
            let d = u32::from(std.coded_denom) + 9;
            (up_w * 8 + d / 2) / d
        } else {
            up_w
        };

        let refs: [Option<Reference>; REFS] = std::array::from_fn(|i| {
            reference_slot(frame_info, av1.referenceNameSlotIndices[i])
        });

        let enable_order_hint = seq.flags.enable_order_hint() != 0;
        let order_bits = u32::from(seq.order_hint_bits_minus_1) + 1;
        // get_relative_dist() of the specification.
        let dist = |a: u8, b: u8| -> i32 {
            if !enable_order_hint {
                return 0;
            }
            let m = 1i32 << (order_bits - 1);
            let diff = i32::from(a) - i32::from(b);
            (diff & (m - 1)) - (diff & m)
        };
        let order_hint = std.OrderHint;

        let mut pic: _nvdec_av1_pic_s = unsafe { std::mem::zeroed() };
        const { assert!(std::mem::size_of::<_nvdec_av1_pic_s>() <= PIC_SETUP_B) };

        // The bitstream, from its address in units of 256 bytes: the bytes
        // before the frame count in the offsets.
        let src_buffer =
            unsafe { ffi::nvk_buffer_from_handle(frame_info.srcBuffer) };
        let src = unsafe {
            ffi::vk_buffer_address(
                &(*src_buffer).vk,
                frame_info.srcBufferOffset,
            )
        };
        let src_base = src & !(BUF_ALIGN - 1);
        let lead: u32 = (src - src_base).try_into().unwrap();
        pic.stream_len =
            lead + u32::try_from(frame_info.srcBufferRange).unwrap();
        pic.gptimer_timeout_value = timeout(up_w, h);

        // The sequence.
        let sb128 = seq.flags.use_128x128_superblock() != 0;
        pic.set_use_128x128_superblock(sb128.into());
        let chroma_format = if color.flags.mono_chrome() != 0 {
            0
        } else {
            match (color.subsampling_x, color.subsampling_y) {
                (1, 1) => 1,
                (1, 0) => 2,
                _ => 3,
            }
        };
        pic.set_chroma_format(chroma_format);
        pic.set_bit_depth(color.BitDepth.into());
        pic.set_enable_filter_intra(seq.flags.enable_filter_intra());
        pic.set_enable_intra_edge_filter(seq.flags.enable_intra_edge_filter());
        pic.set_enable_interintra_compound(
            seq.flags.enable_interintra_compound(),
        );
        pic.set_enable_masked_compound(seq.flags.enable_masked_compound());
        pic.set_enable_dual_filter(seq.flags.enable_dual_filter());
        pic.set_enable_jnt_comp(seq.flags.enable_jnt_comp());
        // What enables the film grain is the picture output being another
        // one: the decoder only applies it when told to.
        pic.set_enable_fgs(display.is_some().into());

        // The quantizers, and whether the frame is lossless: every segment
        // is, of quantizer index 0 without deltas (CodedLossless).
        let quant = unsafe { &*std.pQuantization };
        let seg = unsafe { std.pSegmentation.as_ref() }
            .filter(|_| flags.segmentation_enabled() != 0);
        let no_deltas = quant.DeltaQYDc == 0
            && quant.DeltaQUDc == 0
            && quant.DeltaQUAc == 0
            && quant.DeltaQVDc == 0
            && quant.DeltaQVAc == 0;
        let qindex = |s: usize| -> i32 {
            let base = i32::from(quant.base_q_idx);
            match seg {
                Some(seg) if seg.FeatureEnabled[s] & 1 != 0 => {
                    (base + i32::from(seg.FeatureData[s][0])).clamp(0, 255)
                }
                _ => base,
            }
        };
        let segments = if seg.is_some() { 8 } else { 1 };
        let coded_lossless = no_deltas && (0..segments).all(|s| qindex(s) == 0);

        pic.base_qindex = quant.base_q_idx;
        pic.y_dc_delta_q = quant.DeltaQYDc;
        pic.u_dc_delta_q = quant.DeltaQUDc;
        pic.v_dc_delta_q = quant.DeltaQVDc;
        pic.u_ac_delta_q = quant.DeltaQUAc;
        pic.v_ac_delta_q = quant.DeltaQVAc;
        let qm = |v: u8| {
            if quant.flags.using_qmatrix() != 0 {
                v
            } else {
                15
            }
        };
        pic.qm_y = qm(quant.qm_y);
        pic.qm_u = qm(quant.qm_u);
        pic.qm_v = qm(quant.qm_v);

        // CDEF, which is off on lossless frames and frames of intra block
        // copy, and when it would do nothing.
        let cdef = unsafe { std.pCDEF.as_ref() }.filter(|cdef| {
            seq.flags.enable_cdef() != 0
                && !coded_lossless
                && !allow_intrabc
                && !(cdef.cdef_bits == 0
                    && cdef.cdef_y_pri_strength[0] == 0
                    && cdef.cdef_y_sec_strength[0] == 0
                    && cdef.cdef_uv_pri_strength[0] == 0
                    && cdef.cdef_uv_sec_strength[0] == 0)
        });
        pic.set_enable_cdef(cdef.is_some().into());
        if let Some(cdef) = cdef {
            pic.set_cdef_damping_minus_3(cdef.cdef_damping_minus_3.into());
            pic.set_cdef_bits(cdef.cdef_bits.into());
            // The secondary strengths as coded: 0 to 3, 3 meaning 4.
            let sec = |v: u8| u32::from(v.min(3));
            let (mut y_sec, mut uv_sec) = (0, 0);
            for i in 0..1usize << cdef.cdef_bits {
                pic.cdef_y_pri_strength |=
                    u32::from(cdef.cdef_y_pri_strength[i] & 0xf) << (4 * i);
                pic.cdef_uv_pri_strength |=
                    u32::from(cdef.cdef_uv_pri_strength[i] & 0xf) << (4 * i);
                y_sec |= sec(cdef.cdef_y_sec_strength[i]) << (2 * i);
                uv_sec |= sec(cdef.cdef_uv_sec_strength[i]) << (2 * i);
            }
            pic.set_cdef_y_sec_strength(y_sec);
            pic.set_cdef_uv_sec_strength(uv_sec);
        }

        // The frame header.
        pic.set_frame_type(frame_type as u32);
        pic.set_show_frame(1);
        pic.set_disable_cdf_update(flags.disable_cdf_update());
        pic.set_allow_screen_content_tools(flags.allow_screen_content_tools());
        // Intra frames use integer motion vectors (intra block copy's).
        pic.set_cur_frame_force_integer_mv(
            (flags.force_integer_mv() != 0 || intra).into(),
        );
        pic.set_scale_denom_minus9(if use_superres {
            std.coded_denom.into()
        } else {
            0
        });
        pic.set_allow_intrabc(flags.allow_intrabc());
        pic.set_allow_high_precision_mv(flags.allow_high_precision_mv());
        pic.set_interp_filter(std.interpolation_filter as u32);
        pic.set_switchable_motion_mode(flags.is_motion_mode_switchable());
        pic.set_use_ref_frame_mvs(flags.use_ref_frame_mvs());
        let keep_read = flags.disable_frame_end_update_cdf() != 0;
        pic.set_refresh_frame_context((!keep_read).into());
        pic.set_delta_q_present_flag(flags.delta_q_present());
        pic.set_delta_q_res(std.delta_q_res.into());
        pic.set_delta_lf_present_flag(flags.delta_lf_present());
        pic.set_delta_lf_res(std.delta_lf_res.into());
        pic.set_delta_lf_multi(flags.delta_lf_multi());
        pic.set_coded_lossless(coded_lossless.into());
        pic.set_superres_is_scaled(use_superres.into());

        // The tiles, and their size in superblocks.
        let tile_info = unsafe { &*std.pTileInfo };
        let (cols, rows) = (
            usize::from(tile_info.TileCols),
            usize::from(tile_info.TileRows),
        );
        pic.set_tile_enabled((cols > 1 || rows > 1).into());
        pic.set_tile_cols(cols.try_into().unwrap());
        pic.set_tile_rows(rows.try_into().unwrap());
        pic.set_context_update_tile_id(tile_info.context_update_tile_id.into());
        let sizes = |minus_1: *const u16, count: usize| {
            unsafe { std::slice::from_raw_parts(minus_1, count) }
                .iter()
                .map(|&s| u32::from(s) + 1)
                .collect::<Vec<u32>>()
        };
        let tile_widths = sizes(tile_info.pWidthInSbsMinus1, cols);
        let tile_heights = sizes(tile_info.pHeightInSbsMinus1, rows);

        pic.set_frame_tx_mode(match std.TxMode {
            ffi::STD_VIDEO_AV1_TX_MODE_SELECT => 4,
            ffi::STD_VIDEO_AV1_TX_MODE_LARGEST => 3,
            _ => 0,
        });
        pic.set_frame_reference_mode(if flags.reference_select() != 0 {
            2
        } else {
            0
        });
        if flags.skip_mode_present() != 0 {
            pic.set_skip_mode_flag(1);
            pic.set_skip_ref0(std.SkipModeFrame[0].into());
            pic.set_skip_ref1(std.SkipModeFrame[1].into());
        }
        pic.set_allow_warp(flags.allow_warped_motion());
        pic.set_reduced_tx_set_used(flags.reduced_tx_set());

        // Superres, of steps of 14 without it, as NVIDIA's driver gives.
        pic.superres_upscaled_width = up_w.try_into().unwrap();
        if use_superres {
            let (luma_step, luma_x) = superres(w, up_w);
            let (chroma_step, chroma_x) = if color.subsampling_x != 0 {
                superres((w + 1) >> 1, (up_w + 1) >> 1)
            } else {
                (luma_step, luma_x)
            };
            pic.superres_luma_step = luma_step;
            pic.superres_chroma_step = chroma_step;
            pic.superres_init_luma_subpel_x = luma_x;
            pic.superres_init_chroma_subpel_x = chroma_x;
        } else {
            pic.superres_luma_step = 14;
            pic.superres_chroma_step = 14;
        }

        // Segmentation: the decoder takes the features enabled, with the
        // reference of REF_FRAME plus 1 but on key frames, and SKIP and
        // GLOBALMV as 1.
        pic.segment_enabled = flags.segmentation_enabled() as u8;
        pic.segment_update_map = flags.segmentation_update_map() as u8;
        pic.segment_temporal_update =
            flags.segmentation_temporal_update() as u8;
        if let Some(seg) = seg {
            let mut quant_sign = 0u8;
            for s in 0..8 {
                for j in 0..8 {
                    if seg.FeatureEnabled[s] & (1 << j) == 0 {
                        continue;
                    }
                    let v = seg.FeatureData[s][j];
                    pic.segment_feature_data[s][j] = match j {
                        0 => {
                            quant_sign |= u8::from(v < 0) << s;
                            v.unsigned_abs().min(255) as i16
                        }
                        5 if key_frame => 0,
                        5 => v + 1,
                        6 | 7 => 1,
                        _ => v,
                    };
                    pic.last_active_segid = s as u8;
                    if j >= 5 {
                        pic.segid_preskip = 1;
                    }
                }
            }
            pic.set_segment_quant_sign(quant_sign);
        }

        // The loop filter.
        let lf = unsafe { &*std.pLoopFilter };
        pic.filter_level = [lf.loop_filter_level[0], lf.loop_filter_level[1]];
        if color.flags.mono_chrome() == 0
            && (lf.loop_filter_level[0] != 0 || lf.loop_filter_level[1] != 0)
        {
            pic.filter_level_u = lf.loop_filter_level[2];
            pic.filter_level_v = lf.loop_filter_level[3];
        }
        pic.lf_sharpness_level = lf.loop_filter_sharpness;
        if lf.flags.loop_filter_delta_enabled() != 0 {
            pic.lf_ref_deltas = lf.loop_filter_ref_deltas;
            pic.lf_mode_deltas = lf.loop_filter_mode_deltas;
        }

        // Loop restoration, of each plane's type and unit size, 2 bits each:
        // the size as Vulkan gives it, log2 of the size in pixels minus 5.
        let lr = unsafe { std.pLoopRestoration.as_ref() }
            .filter(|_| !allow_intrabc && seq.flags.enable_restoration() != 0);
        if let Some(lr) = lr {
            for p in 0..3 {
                pic.lr_type |=
                    ((lr.FrameRestorationType[p] as u8) & 3) << (2 * p);
                pic.lr_unit_size |=
                    ((lr.LoopRestorationSize[p] as u8) & 3) << (2 * p);
            }
        } else {
            pic.lr_unit_size = 0x3f;
        }

        // The current frame and the references: their size, scale and
        // distance in display order.
        pic.current_frame.width = w.try_into().unwrap();
        pic.current_frame.height = h.try_into().unwrap();
        pic.current_frame.stride = [dst.stride(0) / 16, dst.stride(1) / 16];
        let gm = unsafe { std.pGlobalMotion.as_ref() };
        let mut scaled = [false; REFS];
        if !intra || allow_intrabc {
            for i in 0..REFS {
                let r = &mut pic.ref_frame[i];
                let (ref_w, ref_h, pitch) = match &refs[i] {
                    Some(rf) => {
                        let off = dist(order_hint, rf.info.OrderHint);
                        let roff = dist(rf.info.OrderHint, order_hint);
                        r.frame_off = off as i16;
                        r.roffset = roff as i16;
                        r.set_sign_bias((roff > 0).into());
                        (rf.pic.width, rf.pic.height, &rf.pic)
                    }
                    None => (w, h, &dst),
                };
                r.info.width = ref_w.try_into().unwrap();
                r.info.height = ref_h.try_into().unwrap();
                r.info.stride = [pitch.stride(0) / 16, pitch.stride(1) / 16];
                let x_scale = ((ref_w << 14) + w / 2) / w;
                let y_scale = ((ref_h << 14) + h / 2) / h;
                r.sf.x_scale_fp = x_scale as i16;
                r.sf.y_scale_fp = y_scale as i16;
                scaled[i] = x_scale != 1 << 14 || y_scale != 1 << 14;
                if let Some(gm) = gm {
                    r.set_wmtype(gm.GmType[LAST_FRAME + i].into());
                }
            }
            pic.set_ref_scaling_enable(scaled.iter().any(|&s| s).into());
        }

        // The motion field projections and the segment ids the frame
        // predicts its own from, which are chosen at submission.
        let primary = (std.primary_ref_frame
            != ffi::STD_VIDEO_AV1_PRIMARY_REF_NONE as u8)
            .then(|| refs[usize::from(std.primary_ref_frame)].as_ref())
            .flatten();
        let motion = MotionSetup {
            refs: std::array::from_fn(|i| {
                refs[i].as_ref().map(|r| RefMotion {
                    slot: r.slot,
                    intra: r.is_intra(),
                    order_hint: r.info.OrderHint,
                    saved_order_hints: r.info.SavedOrderHints,
                    width: r.pic.width,
                    height: r.pic.height,
                    segmentation: r.info.flags.segmentation_enabled() != 0,
                })
            }),
            primary: (std.primary_ref_frame
                != ffi::STD_VIDEO_AV1_PRIMARY_REF_NONE as u8)
                .then_some(usize::from(std.primary_ref_frame)),
            intra,
            order_hint,
            width: w,
            height: h,
            use_ref_frame_mvs: flags.use_ref_frame_mvs() != 0,
            segmentation: seg.is_some(),
            enable_order_hint,
            order_hint_bits: order_bits,
        };

        // The buffers.
        pic.set_tileformat(1);
        let y_log2 =
            unsafe { (*dst.img).planes[0].nil.levels[0].tiling.y_log2 };
        pic.set_gob_height((y_log2 - 1).into());

        let session = unsafe { cmd.state.video.vid.as_ref().unwrap() };
        let layout = nvk_video_av1_mem_layout(
            session.vk.max_coded.width,
            session.vk.max_coded.height,
        );
        pic.Av1FltTopOffset = layout.flt_top;
        pic.Av1FltVertOffset = layout.flt_vert;
        pic.Av1CdefVertOffset = layout.cdef_vert;
        pic.Av1LrVertOffset = layout.lr_vert;
        pic.Av1HusVertOffset = layout.hus_vert;
        pic.Av1FgsVertOffset = layout.fgs_vert;

        let mv_segs = MvSegs {
            addr: session.mems[ffi::NVK_AV1_MEM_MV_SEG as usize].addr,
            slot_B: layout.slot_B,
            mv_B: layout.mv_B,
            zero_seg_offset_B: layout.zero_seg_offset_B,
        };
        let filter = session.mems[ffi::NVK_AV1_MEM_FILTER as usize].addr;
        let probs = session.mems[ffi::NVK_AV1_MEM_PROBS as usize].addr;
        let dst_slot = slot.unwrap_or(SCRATCH_SLOT);

        // The setup buffer: the picture setup, the tiles and the global
        // motion, then the status.
        let film_grain = unsafe { std.pFilmGrain.as_ref() }.filter(|_| {
            display.is_some()
                && seq.flags.film_grain_params_present() != 0
                && flags.apply_grain() != 0
        });
        let mut setup_data = vec![0u8; STATUS_OFFSET];
        if let Some(fg) = film_grain {
            av1_film_grain::film_grain(
                fg,
                color.BitDepth,
                &mut pic.fgs_cfg,
                &mut setup_data[FILM_GRAIN_OFFSET..]
                    [..av1_film_grain::FILM_GRAIN_B],
            );
        }
        unsafe {
            std::ptr::copy_nonoverlapping(
                (&pic as *const _nvdec_av1_pic_s).cast::<u8>(),
                setup_data.as_mut_ptr(),
                std::mem::size_of::<_nvdec_av1_pic_s>(),
            );
        }

        let tile_count = usize::try_from(av1.tileCount).unwrap();
        assert!(tile_count == cols * rows && tile_count <= MAX_TILES);
        let tile_offsets =
            unsafe { std::slice::from_raw_parts(av1.pTileOffsets, tile_count) };
        let tile_sizes =
            unsafe { std::slice::from_raw_parts(av1.pTileSizes, tile_count) };
        for t in 0..tile_count {
            let e = &mut setup_data[TILES_OFFSET + t * TILE_B..][..TILE_B];
            let start = lead + tile_offsets[t];
            e[0] = tile_widths[t % cols] as u8;
            e[1] = tile_heights[t / cols] as u8;
            e[2..6].copy_from_slice(&start.to_le_bytes());
            e[6..10].copy_from_slice(&(start + tile_sizes[t]).to_le_bytes());
        }

        // The global motion of each reference: its warp model, and its
        // shear, or 0x8000 when it can't be warped with.
        for i in 0..REFS {
            let identity = [0, 0, 1 << 16, 0, 0, 1 << 16];
            let (model, shear) = if intra {
                (identity, Some([0; 4]))
            } else {
                let model =
                    gm.map_or(identity, |gm| gm.gm_params[LAST_FRAME + i]);
                (model, shear(&model).filter(|_| !scaled[i]))
            };
            let e = &mut setup_data[GLOBAL_MOTION_OFFSET + i * 32..][..32];
            for (j, v) in model.iter().enumerate() {
                e[4 * j..4 * j + 4].copy_from_slice(&v.to_le_bytes());
            }
            for (j, v) in shear.unwrap_or([i16::MIN; 4]).iter().enumerate() {
                e[24 + 2 * j..26 + 2 * j].copy_from_slice(&v.to_le_bytes());
            }
        }

        let (setup_addr, setup_map) = unsafe {
            upload_bytes_mapped(
                cmd,
                Some((setup_data.as_ptr(), setup_data.len())),
                SETUP_B.try_into().unwrap(),
            )
        }?;

        let mut push = Push::new();
        push.push_mthd(clc7b0::SetApplicationId {
            id: clc7b0::SetApplicationIdId::Avd,
        });
        push.push_mthd(clc7b0::SetControlParams {
            codec_type: clc7b0::SetControlParamsCodecType::Av1,
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
        // The references, LAST_FRAME to ALTREF_FRAME: intra frames, and
        // references missing, are given the frame itself.
        for (i, r) in refs.iter().enumerate() {
            let r = r.as_ref().filter(|_| !intra).map_or(&dst, |r| &r.pic);
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
            offset: addr_hi(setup_addr + STATUS_OFFSET as u64),
        });
        let picture_index =
            push.push_mthd_patchable(clc7b0::SetPictureIndex { index: 0 });
        push.push_mthd(clc7b0::SetTileSizeBufOffset {
            offset: addr_hi(setup_addr + TILES_OFFSET as u64),
        });
        push.push_mthd(clc7b0::Av1SetGlobalModelBufOffset {
            offset: addr_hi(setup_addr + GLOBAL_MOTION_OFFSET as u64),
        });
        push.push_mthd(clc7b0::Av1SetFilmGrainBufOffset {
            offset: addr_hi(setup_addr + FILM_GRAIN_OFFSET as u64),
        });
        // The methods FrameFixup fills in.
        let segment_read =
            push.push_mthd_patchable(clc7b0::Av1SetSegmentReadBufOffset {
                offset: 0,
            });
        push.push_mthd(clc7b0::Av1SetSegmentWriteBufOffset {
            offset: addr_hi(mv_segs.segs(dst_slot)),
        });
        let col_mv_read = [
            push.push_mthd_patchable(clc7b0::Av1SetColMv0ReadBufOffset {
                offset: 0,
            }),
            push.push_mthd_patchable(clc7b0::Av1SetColMv1ReadBufOffset {
                offset: 0,
            }),
            push.push_mthd_patchable(clc7b0::Av1SetColMv2ReadBufOffset {
                offset: 0,
            }),
        ];
        push.push_mthd(clc7b0::Av1SetColMvwriteBufOffset {
            offset: addr_hi(mv_segs.mv(dst_slot)),
        });
        push.push_mthd(clc7b0::SetFilterBufferOffset {
            offset: addr_hi(filter),
        });
        push.push_mthd(clc7b0::SetIntraTopBufOffset {
            offset: addr_hi(filter + layout.intra_top_offset_B),
        });
        let read_probs =
            push.push_mthd_patchable(clc7b0::Av1SetProbTabReadBufOffset {
                offset: 0,
            });
        let write_probs =
            push.push_mthd_patchable(clc7b0::Av1SetProbTabWriteBufOffset {
                offset: 0,
            });
        if let Some(display) = &display {
            push.push_mthd(clc7b0::SetDisplayBufLumaOffset {
                offset: display.luma(),
            });
            push.push_mthd(clc7b0::SetDisplayBufChromaOffset {
                offset: display.chroma(),
            });
        }
        // The picture decoded, after the references.
        push.push_array_mthd(
            8,
            clc7b0::SetPictureLumaOffset { offset: dst.luma() },
        );
        push.push_array_mthd(
            8,
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
        let start = nvk_cmd_buffer_push(cmd, push);
        let word = |i: usize| unsafe { start.add(i) };

        // The CDFs the frame starts from: the defaults, or those its primary
        // reference frame saved.
        let read = match primary {
            Some(p) if !intra => ReadProbs::Slot(p.slot),
            _ => ReadProbs::Default(av1_probs::q_ctx(quant.base_q_idx)),
        };
        add_submit_fixup(
            cmd,
            Box::new(FrameFixup {
                state: self.state.clone(),
                read,
                slot,
                dst_slot,
                keep_read,
                motion,
                probs_addr: probs,
                mv_segs,
                pic: Box::new(pic),
                pic_map: setup_map.cast(),
                picture_index: word(picture_index),
                read_probs: word(read_probs),
                write_probs: word(write_probs),
                col_mv_read: col_mv_read.map(word),
                segment_read: word(segment_read),
            }),
        );

        Ok(())
    }
}

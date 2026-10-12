// Copyright © 2024 Collabora, Ltd
// SPDX-License-Identifier: MIT

use crate::ffi;
use crate::ffi::{nvk_cmd_buffer, nvk_image, nvk_image_view, VK_SUCCESS};
use nv_push_rs;
use nvidia_headers::classes::cl906f::mthd as cl906f;
use nvidia_headers::Mthd;

mod av1;
mod av1_default_cdfs;
mod av1_film_grain;
mod av1_gaussian;
mod av1_probs;
mod h264;
mod vp9;
mod vp9_default_probs;

/// Suballocate `size` upload-heap bytes (256-byte aligned) and, if `data` is
/// given, copy it in. Returns the GPU address of the allocation.
unsafe fn upload_bytes(
    cmd: &mut nvk_cmd_buffer,
    data: Option<(*const u8, usize)>,
    size: u32,
) -> Result<u64, ffi::VkResult> {
    unsafe { upload_bytes_mapped(cmd, data, size) }.map(|(addr, _)| addr)
}

/// [`upload_bytes`], which also returns where the allocation is mapped.
unsafe fn upload_bytes_mapped(
    cmd: &mut nvk_cmd_buffer,
    data: Option<(*const u8, usize)>,
    size: u32,
) -> Result<(u64, *mut u8), ffi::VkResult> {
    let mut ptr = std::ptr::null_mut();
    let mut addr = 0;
    unsafe {
        let res = ffi::nvk_cmd_buffer_upload_alloc(
            cmd,
            size,
            256,
            &mut addr,
            &mut ptr as *mut *mut _ as *mut *mut std::ffi::c_void,
        );
        if res != VK_SUCCESS {
            return Err(res);
        }
        if let Some((src, len)) = data {
            std::ptr::copy_nonoverlapping(src, ptr as *mut u8, len);
        }
    }
    Ok((addr, ptr as *mut u8))
}

/// What a decode needs of the frames decoded before it, in the order the
/// decoder executes them: command buffers may be recorded in another order
/// than they are submitted in, so the methods and setups depending on it are
/// filled in at submission, as submissions execute in the order they are made.
///
/// They are filled in in the command buffer's own memory, which a command
/// buffer submitted again while it is pending (with
/// `VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT`) has the GPU read meanwhile.
pub(crate) trait SubmitFixup {
    /// Fills in what the command buffer leaves to its submission, from the
    /// session's state, and updates that state.
    fn submit(&mut self);
}

/// The fixups of a command buffer, in the order it recorded them, which
/// `nvk_cmd_buffer::video_fixups` owns.
#[derive(Default)]
struct CmdFixups {
    fixups: Vec<Box<dyn SubmitFixup>>,
}

/// Adds `fixup` to the ones applied when `cmd` is submitted.
fn add_submit_fixup(cmd: &mut nvk_cmd_buffer, fixup: Box<dyn SubmitFixup>) {
    if cmd.video_fixups.is_null() {
        cmd.video_fixups = Box::into_raw(Box::new(CmdFixups::default())).cast();
    }
    let fixups = unsafe { &mut *(cmd.video_fixups as *mut CmdFixups) };
    fixups.fixups.push(fixup);
}

/// Applies the fixups of a command buffer being submitted.
#[unsafe(no_mangle)]
pub extern "C" fn nvk_video_cmd_buffer_submit(cmd: &mut nvk_cmd_buffer) {
    // SAFETY: a command buffer isn't recorded, reset or freed while it is
    // submitted.
    let Some(fixups) =
        (unsafe { (cmd.video_fixups as *mut CmdFixups).as_mut() })
    else {
        return;
    };
    for fixup in &mut fixups.fixups {
        fixup.submit();
    }
}

/// Frees the fixups of a command buffer being reset or destroyed.
#[unsafe(no_mangle)]
pub extern "C" fn nvk_video_cmd_buffer_reset(cmd: &mut nvk_cmd_buffer) {
    if !cmd.video_fixups.is_null() {
        drop(unsafe { Box::from_raw(cmd.video_fixups as *mut CmdFixups) });
        cmd.video_fixups = std::ptr::null_mut();
    }
}

/// Identifies a DPB picture.
///
/// With "separated" DPB each picture has its own image view (always array layer
/// 0); with "layered" DPB a single array image view backs every picture and the
/// array layer selects the picture. Keying on the image view alone therefore
/// collides in the layered case, so the array layer is part of the key. Two
/// video picture resources only match if they refer to the same image
/// subresource *and* specify the same coded offset and extent, so those are
/// part of the key as well.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
struct SlotKey {
    image_view: *const nvk_image_view,
    layer: u32,
    coded_offset: (i32, i32),
    coded_extent: (u32, u32),
}

/// Build the [`SlotKey`] for a Vulkan picture resource.
fn slot_key(res: &ffi::VkVideoPictureResourceInfoKHR) -> SlotKey {
    let image_view =
        unsafe { ffi::nvk_image_view_from_handle(res.imageViewBinding) };
    SlotKey {
        image_view,
        layer: unsafe { (*image_view).vk.base_array_layer }
            + res.baseArrayLayer,
        coded_offset: (res.codedOffset.x, res.codedOffset.y),
        coded_extent: (res.codedExtent.width, res.codedExtent.height),
    }
}

/// A GPU address in the form the engine's offset methods take: shifted right
/// by 8, so the low 8 bits have to be zero for the address to survive the
/// round trip.
fn addr_hi(addr: u64) -> u32 {
    assert!(addr & 0xff == 0, "address {addr:#x} is not 256B aligned");
    (addr >> 8).try_into().unwrap()
}

/// Address (already shifted right by 8, as the hardware expects) of the given
/// plane and DPB array layer of an image. For "separated" DPB the layer is 0
/// and this is just the plane base address; for "layered" DPB the layer selects
/// the picture within the array image.
fn plane_layer_base(img: *mut nvk_image, plane: u8, layer: u32) -> u32 {
    let base = unsafe { ffi::nvk_image_base_address(img, plane) };
    let array_stride =
        unsafe { (*img).planes[plane as usize].nil.array_stride_B };
    addr_hi(base + u64::from(layer) * array_stride)
}

/// A picture the decoder reads or writes.
pub(super) struct Picture {
    pub(super) img: *mut nvk_image,
    pub(super) layer: u32,
    pub(super) width: u32,
    pub(super) height: u32,
}

impl Picture {
    pub(super) fn new(res: &ffi::VkVideoPictureResourceInfoKHR) -> Self {
        let key = slot_key(res);
        Picture {
            img: unsafe { (*key.image_view).vk.image as *mut nvk_image },
            layer: key.layer,
            width: res.codedExtent.width,
            height: res.codedExtent.height,
        }
    }

    /// The row pitch of `plane` in bytes.
    pub(super) fn stride(&self, plane: usize) -> u16 {
        let pitch =
            unsafe { (*self.img).planes[plane].nil.levels[0].row_stride_B };
        pitch.try_into().unwrap()
    }

    pub(super) fn luma(&self) -> u32 {
        plane_layer_base(self.img, 0, self.layer)
    }

    pub(super) fn chroma(&self) -> u32 {
        plane_layer_base(self.img, 1, self.layer)
    }
}

struct SessionData {
    decoder: Box<dyn VideoDecoder>,
}

pub(crate) trait VideoDecoder {
    /// Forgets what the session decoded: the next frame is decoded as the
    /// first one (`VK_VIDEO_CODING_CONTROL_RESET_BIT_KHR`).
    fn reset(&mut self, _cmd: &mut ffi::nvk_cmd_buffer) {}

    fn begin(
        &mut self,
        _cmd: &mut ffi::nvk_cmd_buffer,
        _begin_info: &ffi::VkVideoBeginCodingInfoKHR,
    ) {
    }

    fn decode(
        &mut self,
        cmd: &mut ffi::nvk_cmd_buffer,
        frame_info: &ffi::VkVideoDecodeInfoKHR,
    ) -> Result<(), ffi::VkResult>;
}

#[unsafe(no_mangle)]
pub extern "C" fn nvk_video_create_video_session(
    vid: &mut ffi::nvk_video_session,
) {
    let decoder: Box<dyn VideoDecoder> = match vid.vk.op {
        ffi::VK_VIDEO_CODEC_OPERATION_DECODE_VP9_BIT_KHR => {
            Box::new(vp9::Decoder::default())
        }
        ffi::VK_VIDEO_CODEC_OPERATION_DECODE_AV1_BIT_KHR => {
            Box::new(av1::Decoder::default())
        }
        _ => Box::new(h264::Decoder::default()),
    };
    vid.rust = Box::into_raw(Box::new(SessionData { decoder })).cast();
}

#[unsafe(no_mangle)]
pub extern "C" fn nvk_video_destroy_video_session(
    vid: &mut ffi::nvk_video_session,
) {
    drop(unsafe { Box::from_raw(vid.rust as *mut SessionData) });
}

/// Cast the opaque `nvk_video_session::rust` pointer to a mutable reference to
/// a `SessionData`.
///
/// # Safety:
///
/// The caller must ensure that the pointer is valid when accessing the
/// reference and that Rust's aliasing rules are followed.
///
fn to_session<'a>(cmd_buf: *mut ffi::nvk_cmd_buffer) -> &'a mut SessionData {
    unsafe {
        let video = (*cmd_buf).state.video;
        let rust_ptr = (*video.vid).rust as *mut SessionData;

        // SAFETY: Vulkan requires external synchronization for the command
        // buffer being recorded and for the video session bound to it
        // (vkCmdBeginVideoCodingKHR's videoSession parameter), so no other
        // thread can access this session concurrently. Each entry point drops
        // the returned reference before returning, so no two mutable
        // references ever coexist.
        &mut *rust_ptr
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn nvk_push_video_decode_state_init(
    _queue: &mut ffi::nvk_queue,
    p: *mut ffi::nv_push,
    cls_vdec: u16,
) -> ffi::VkResult {
    let mut push = nv_push_rs::Push::new();

    // The decoder object is of the device's video decoder class, not
    // Turing's: the methods are handled by the rules of the class bound here.
    let set_object = cl906f::SetObject {
        nvclass: cls_vdec.into(),
        engine: cl906f::SetObjectEngine::Sw,
    };
    push.push_mthd_bits(
        nv_push_rs::class_to_subc(cls_vdec),
        cl906f::SetObject::ADDR,
        set_object.to_bits(),
    );
    unsafe { crate::nv_push_append_push(p, push) };

    ffi::VK_SUCCESS
}

#[unsafe(no_mangle)]
pub extern "C" fn nvk_cmd_buffer_begin_video_decode(
    _cmd: &mut ffi::nvk_cmd_buffer,
    _begin_info: &ffi::VkCommandBufferBeginInfo,
) {
}

#[unsafe(no_mangle)]
pub extern "C" fn nvk_video_cmd_begin_video_coding_khr(
    cmd: &mut ffi::nvk_cmd_buffer,
    begin_info: &ffi::VkVideoBeginCodingInfoKHR,
) {
    let session = to_session(cmd);

    session.decoder.begin(cmd, &begin_info);
}

#[unsafe(no_mangle)]
pub extern "C" fn nvk_video_cmd_decode_video_khr(
    cmd: &mut ffi::nvk_cmd_buffer,
    frame_info: &ffi::VkVideoDecodeInfoKHR,
) -> ffi::VkResult {
    let session = to_session(cmd);

    session
        .decoder
        .decode(cmd, &frame_info)
        .err()
        .unwrap_or(ffi::VK_SUCCESS)
}

#[unsafe(no_mangle)]
pub extern "C" fn nvk_video_cmd_control_video_coding_khr(
    cmd: &mut ffi::nvk_cmd_buffer,
    control_info: &ffi::VkVideoCodingControlInfoKHR,
) {
    if control_info.flags & ffi::VK_VIDEO_CODING_CONTROL_RESET_BIT_KHR != 0 {
        to_session(cmd).decoder.reset(cmd);
    }
}

/*
 * Copyright © 2022 Collabora Ltd. and Red Hat Inc.
 * SPDX-License-Identifier: MIT
 */
#ifndef NVK_VIDEO_SESSION_H
#define NVK_VIDEO_SESSION_H 1

#include "nvk_private.h"
#include "nvk_device_memory.h"

#include "vulkan/runtime/vk_video.h"

/* The memory of VP9 sessions, by binding.
 *
 * NVK_VP9_MEM_SEG_MV: the segmentation maps and the co-located motion
 * vectors of the current and the previous picture.
 *
 * NVK_VP9_MEM_FILTER: the intra prediction top line, then the loop filter
 * buffer.
 *
 * NVK_VP9_MEM_PROBS: the probability tables (the 4 frame contexts, the
 * defaults the decoder resets them from, and the current frame's), the symbol
 * counters the decoder adapts them with, and the decoder's state of the
 * previous picture.
 */
enum nvk_vp9_mem {
   NVK_VP9_MEM_SEG_MV,
   NVK_VP9_MEM_FILTER,
   NVK_VP9_MEM_PROBS,
};

/* The decoder takes the addresses of its buffers in units of 256 bytes. */
#define NVK_VP9_BUF_ALIGN_B 256

/* Bytes per 64x64 superblock of a segmentation map. */
#define NVK_VP9_SEG_MAP_SB_B 32
/* Bytes per 64x64 superblock of co-located motion vectors. */
#define NVK_VP9_COL_MV_SB_B 1024
/* Bytes of the intra prediction top line, before the loop filter buffer. */
#define NVK_VP9_INTRA_TOP_B 0x10000
/* Bytes of a probability table: nvdec_vp9EntropyProbs_t, 256-byte aligned. */
#define NVK_VP9_PROBS_B 0xf00
/* Probability tables: the 4 frame contexts, the defaults, and the current
 * frame's, which the parser leaves for the decoder unless it saves them to
 * the frame context as they are.
 */
#define NVK_VP9_PROBS_COUNT 6
#define NVK_VP9_DEFAULT_PROBS 4
#define NVK_VP9_CURRENT_PROBS 5
/* Bytes of the symbol counters: nvdec_vp9EntropyCounts_t, 256-byte aligned. */
#define NVK_VP9_COUNTERS_B 0x3400
/* Bytes of the decoder's state of the previous picture. */
#define NVK_VP9_PREV_PIC_B 0x200
#define NVK_VP9_COUNTERS_OFFSET_B (NVK_VP9_PROBS_COUNT * NVK_VP9_PROBS_B)
#define NVK_VP9_PREV_PIC_OFFSET_B (NVK_VP9_COUNTERS_OFFSET_B + NVK_VP9_COUNTERS_B)
#define NVK_VP9_PROBS_MEM_B (NVK_VP9_PREV_PIC_OFFSET_B + NVK_VP9_PREV_PIC_B)

/* The memory of AV1 sessions, by binding.
 *
 * NVK_AV1_MEM_MV_SEG: for each DPB slot, and a scratch slot for frames which
 * aren't references, the temporal motion vectors and the segment ids of the
 * frame decoded to it; then a segment id map of zeros, which frames read when
 * the frame they predict the segment ids from had segmentation disabled.
 *
 * NVK_AV1_MEM_FILTER: the loop filter, CDEF, loop restoration, superres and
 * film grain buffers, then the intra prediction top line.
 *
 * NVK_AV1_MEM_PROBS: the probability tables: the default ones of the 4
 * quantizer contexts, then the pool frames save theirs to.
 *
 * The sizes depend on the largest picture, see nvk_video_av1_mem_layout().
 */
enum nvk_av1_mem {
   NVK_AV1_MEM_MV_SEG,
   NVK_AV1_MEM_FILTER,
   NVK_AV1_MEM_PROBS,
};

/* Bytes of an AV1 probability table, 256-byte aligned. */
#define NVK_AV1_PROBS_B 0x3000
/* The default probability tables, one per quantizer context. */
#define NVK_AV1_DEFAULT_PROBS 4
/* The probability tables frames save theirs to: one for each frame the 9 DPB
 * slots may hold, and one for the frame decoded.
 */
#define NVK_AV1_PROBS_POOL 10
#define NVK_AV1_PROBS_MEM_B \
   ((NVK_AV1_DEFAULT_PROBS + NVK_AV1_PROBS_POOL) * NVK_AV1_PROBS_B)

struct nvk_vid_mem {
   uint64_t size_B;
   uint32_t align_B;

   uint64_t addr;
};

struct nvk_video_session {
   /** The parent object */
   struct vk_video_session vk;
   /** Opaque memory objects needed by the GPU.
    *
    * We must ensure they're allocated and that the size is correctly computed
    * from codec parameters.
    */
   struct nvk_vid_mem mems[3];
   /** Opaque pointer to data managed by the Rust side. */
   void *rust;
};

VK_DEFINE_NONDISP_HANDLE_CASTS(nvk_video_session, vk.base, VkVideoSessionKHR,
                               VK_OBJECT_TYPE_VIDEO_SESSION_KHR);

#endif /* NVK_VIDEO_SESSION_H */

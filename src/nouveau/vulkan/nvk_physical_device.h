/*
 * Copyright © 2022 Collabora Ltd. and Red Hat Inc.
 * SPDX-License-Identifier: MIT
 */
#ifndef NVK_PHYSICAL_DEVICE_H
#define NVK_PHYSICAL_DEVICE_H 1

#include "nvk_private.h"

#include "nvk_debug.h"
#include "nvk_instance.h"
#include "nv_device_info.h"
#include "clc7b0.h"

#include "vk_physical_device.h"
#include "vk_sync.h"

#include "wsi_common.h"

#include <sys/types.h>

struct nak_compiler;
struct nvk_instance;
struct nvk_physical_device;
struct nvkmd_pdev;

struct nvk_queue_family {
   VkQueueFlags queue_flags;
   uint32_t queue_count;
   VkQueueGlobalPriority max_priority;
};

struct nvk_memory_heap {
   uint64_t size;
   uint64_t used;
   VkMemoryHeapFlags flags;
   uint64_t (*available)(struct nvk_physical_device *pdev);
};

struct nvk_physical_device {
   struct vk_physical_device vk;
   struct nv_device_info info;
   enum nvk_debug debug_flags;
   bool ssbo_align_4b;

   struct nvkmd_pdev *nvkmd;

   struct nak_compiler *nak;
   struct wsi_device wsi_device;

   uint8_t device_uuid[VK_UUID_SIZE];

   // TODO: add mapable VRAM heap if possible
   struct nvk_memory_heap mem_heaps[3];
   VkMemoryType mem_types[3];
   uint8_t mem_heap_count;
   uint8_t mem_type_count;

   struct nvk_queue_family queue_families[3];
   uint8_t queue_family_count;
};

static inline uint32_t
nvk_min_cbuf_alignment(const struct nv_device_info *info)
{
   return info->cls_eng3d >= 0xC597 /* TURING_A */ ? 64 : 256;
}

VK_DEFINE_HANDLE_CASTS(nvk_physical_device,
   vk.base,
   VkPhysicalDevice,
   VK_OBJECT_TYPE_PHYSICAL_DEVICE)

static inline uint32_t
nvk_use_edb_buffer_views(const struct nvk_physical_device *pdev)
{
   return pdev->debug_flags & NVK_DEBUG_FORCE_EDB_BVIEW;
}

static inline const struct nvk_instance *
nvk_physical_device_instance(const struct nvk_physical_device *pdev)
{
   return (struct nvk_instance *)pdev->vk.instance;
}

/* Whether video decode may be enabled at all: the hardware has NVDEC and the
 * user opted in via NVK_EXPERIMENTAL=video.
 */
static inline bool
nvk_video_available(const struct nvk_instance *instance,
                    const struct nv_device_info *info)
{
   return (instance->experimental_flags & NVK_EXPERIMENTAL_VIDEO) &&
          info->has_video;
}

/* Whether H.264 video decode is enabled: video is available and the codec is
 * built in.
 */
static inline bool
nvk_video_h264_enabled(const struct nvk_instance *instance,
                       const struct nv_device_info *info)
{
   return nvk_video_available(instance, info) && VIDEO_CODEC_H264DEC;
}

/* Whether VP9 video decode is enabled: video is available, the codec is built
 * in, and the decoder parses VP9's compressed header and adapts its
 * probabilities itself, which it does from Ampere on (the VP9_PARSER methods
 * of NVC7B0_VIDEO_DECODER).
 */
static inline bool
nvk_video_vp9_enabled(const struct nvk_instance *instance,
                      const struct nv_device_info *info)
{
   return nvk_video_available(instance, info) && VIDEO_CODEC_VP9DEC &&
          info->cls_vdec >= NVC7B0_VIDEO_DECODER;
}

/* Whether any video decode codec is enabled. */
static inline bool
nvk_video_enabled(const struct nvk_instance *instance,
                  const struct nv_device_info *info)
{
   return nvk_video_h264_enabled(instance, info) ||
          nvk_video_vp9_enabled(instance, info);
}

VkResult nvk_create_drm_physical_device(struct vk_instance *vk_instance,
                                        struct _drmDevice *drm_device,
                                        struct vk_physical_device **pdev_out);

void nvk_physical_device_destroy(struct vk_physical_device *vk_device);

VkExtent2D nvk_max_shading_rate(const struct nvk_physical_device *pdev,
                                VkSampleCountFlagBits samples);

#if defined(VK_USE_PLATFORM_WAYLAND_KHR) || \
    defined(VK_USE_PLATFORM_XCB_KHR) || \
    defined(VK_USE_PLATFORM_XLIB_KHR) || \
    defined(VK_USE_PLATFORM_DISPLAY_KHR)
#define NVK_USE_WSI_PLATFORM
#endif

#endif

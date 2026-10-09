/*
 * Copyright © 2026 soka
 * SPDX-License-Identifier: MIT
 */

#ifndef NVKMD_NOVA_H
#define NVKMD_NOVA_H 1

#include "nvkmd/nvkmd.h"
#include "vk_drm_syncobj.h"

#include "util/simple_mtx.h"
#include "util/u_dynarray.h"
#include "util/vma.h"

#include "drm-uapi/nova_drm.h"

#include <sys/types.h>

/* nvkmd backend for nova-drm, the DRM driver of nova-core, the Rust driver
 * of GSP-based NVIDIA GPUs.
 */

struct nvkmd_nova_pdev {
   struct nvkmd_pdev base;

   int primary_fd;

   struct vk_sync_type syncobj_sync_type;
   const struct vk_sync_type *sync_types[2];
};

NVKMD_DECL_SUBCLASS(pdev, nova);

VkResult nvkmd_nova_try_create_pdev(struct _drmDevice *drm_device,
                                    struct vk_object_base *log_obj,
                                    enum nvk_debug debug_flags,
                                    struct nvkmd_pdev **pdev_out);

/* The GPU virtual address space of a device: NVK's heaps first, then the
 * range nova-core maps what it needs in, such as the memory of channels.
 */
#define NVKMD_NOVA_HEAP_START ((uint64_t)4096)
#define NVKMD_NOVA_HEAP_END ((uint64_t)(1ull << 38))
#define NVKMD_NOVA_REPLAY_HEAP_START NVKMD_NOVA_HEAP_END
#define NVKMD_NOVA_REPLAY_HEAP_END NVKMD_NOVA_KERNEL_START
#define NVKMD_NOVA_KERNEL_START ((uint64_t)(1ull << 39))

struct nvkmd_nova_dev {
   struct nvkmd_dev base;

   /* A file of its own, whose GPU virtual address space is the device's */
   int fd;

   simple_mtx_t heap_mutex;
   struct util_vma_heap heap;
   struct util_vma_heap replay_heap;

   /* How many memory objects use each GEM handle of the file. Importing a
    * dma-buf the file already has a handle for gives that handle again, so
    * a handle is only closed once no memory object uses it.
    */
   simple_mtx_t handles_mutex;
   struct hash_table *handles;
};

NVKMD_DECL_SUBCLASS(dev, nova);

void nvkmd_nova_dev_ref_handle_locked(struct nvkmd_nova_dev *dev,
                                      uint32_t handle);
void nvkmd_nova_dev_unref_handle(struct nvkmd_nova_dev *dev,
                                 uint32_t handle);

VkResult nvkmd_nova_create_dev(struct nvkmd_pdev *pdev,
                               struct vk_object_base *log_obj,
                               struct nvkmd_dev **dev_out);

struct nvkmd_nova_mem {
   struct nvkmd_mem base;

   uint32_t handle;
   uint64_t mmap_offset;
};

NVKMD_DECL_SUBCLASS(mem, nova);

VkResult nvkmd_nova_alloc_mem(struct nvkmd_dev *dev,
                              struct vk_object_base *log_obj,
                              uint64_t size_B, uint64_t align_B,
                              enum nvkmd_mem_flags flags,
                              struct nvkmd_mem **mem_out);

VkResult nvkmd_nova_alloc_tiled_mem(struct nvkmd_dev *dev,
                                    struct vk_object_base *log_obj,
                                    uint64_t size_B, uint64_t align_B,
                                    uint8_t pte_kind, uint16_t tile_mode,
                                    enum nvkmd_mem_flags flags,
                                    struct nvkmd_mem **mem_out);

VkResult nvkmd_nova_import_dma_buf(struct nvkmd_dev *dev,
                                   struct vk_object_base *log_obj,
                                   int fd, struct nvkmd_mem **mem_out);

struct nvkmd_nova_va {
   struct nvkmd_va base;
};

NVKMD_DECL_SUBCLASS(va, nova);

VkResult nvkmd_nova_alloc_va(struct nvkmd_dev *dev,
                             struct vk_object_base *log_obj,
                             enum nvkmd_va_flags flags, uint8_t pte_kind,
                             uint64_t size_B, uint64_t align_B,
                             uint64_t fixed_addr, struct nvkmd_va **va_out);

#define NVKMD_NOVA_MAX_SYNCS 256

struct nvkmd_nova_exec_ctx {
   struct nvkmd_ctx base;

   int fd;
   uint32_t channel;
   uint32_t syncobj;
   uint32_t max_push;

   struct drm_nova_exec req;
   struct drm_nova_sync req_wait[NVKMD_NOVA_MAX_SYNCS];
   struct drm_nova_sync req_sig[NVKMD_NOVA_MAX_SYNCS];
   struct util_dynarray req_push;
};

NVKMD_DECL_SUBCLASS(ctx, nova_exec);

VkResult nvkmd_nova_create_ctx(struct nvkmd_dev *dev,
                               struct vk_object_base *log_obj,
                               enum nvkmd_engines engines,
                               struct nvkmd_ctx **ctx_out);

#endif /* NVKMD_NOVA_H */

/*
 * Copyright © 2026 soka
 * SPDX-License-Identifier: MIT
 */

#include "nvkmd_nova.h"

#include "util/bitscan.h"
#include "util/u_math.h"
#include "util/u_memory.h"
#include "vk_log.h"

#include <errno.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <xf86drm.h>

/* Creates the memory object of the GEM object `handle`, one use of which is
 * counted for it, and maps it at a virtual address of its own. If that
 * fails, the use of `handle` is given back.
 */
static VkResult
create_mem_or_unref_handle(struct nvkmd_nova_dev *dev,
                           struct vk_object_base *log_obj,
                           enum nvkmd_mem_flags flags,
                           uint32_t handle,
                           const struct drm_nova_gem_info *info,
                           uint8_t pte_kind, uint64_t align_B,
                           struct nvkmd_mem **mem_out)
{
   VkResult result;

   struct nvkmd_nova_mem *mem = CALLOC_STRUCT(nvkmd_nova_mem);
   if (mem == NULL) {
      result = vk_error(log_obj, VK_ERROR_OUT_OF_HOST_MEMORY);
      goto fail_handle;
   }

   /* Imports have no alignment: use the page size, which nova maps memory
    * in.
    */
   align_B = MAX2(align_B, dev->base.pdev->bind_align_B);
   nvkmd_mem_init(&dev->base, &mem->base, &nvkmd_nova_mem_ops,
                  flags, info->size, align_B);
   mem->handle = handle;
   mem->mmap_offset = info->mmap_offset;

   enum nvkmd_va_flags va_flags = 0;
   if (!(info->flags & DRM_NOVA_GEM_VRAM))
      va_flags |= NVKMD_VA_GART;

   result = nvkmd_dev_alloc_va(&dev->base, log_obj, va_flags, pte_kind,
                               info->size, align_B, 0 /* fixed_addr */,
                               &mem->base.va);
   if (result != VK_SUCCESS)
      goto fail_mem;

   result = nvkmd_va_bind_mem(mem->base.va, log_obj, 0 /* va_offset_B */,
                              &mem->base, 0 /* mem_offset_B */, info->size);
   if (result != VK_SUCCESS)
      goto fail_va;

   *mem_out = &mem->base;

   return VK_SUCCESS;

fail_va:
   nvkmd_va_free(mem->base.va);
fail_mem:
   FREE(mem);
fail_handle:
   nvkmd_nova_dev_unref_handle(dev, handle);
   return result;
}

VkResult
nvkmd_nova_alloc_mem(struct nvkmd_dev *dev,
                     struct vk_object_base *log_obj,
                     uint64_t size_B, uint64_t align_B,
                     enum nvkmd_mem_flags flags,
                     struct nvkmd_mem **mem_out)
{
   return nvkmd_nova_alloc_tiled_mem(dev, log_obj, size_B, align_B,
                                     0 /* pte_kind */, 0 /* tile_mode */,
                                     flags, mem_out);
}

/* Creates a GEM object of `size_B` bytes with the DRM_NOVA_GEM_* `gem_flags`,
 * counts a use of its handle, and returns the handle and what nova-drm says
 * of the object.
 */
static int
nova_gem_create(struct nvkmd_nova_dev *dev, uint64_t size_B,
                uint32_t gem_flags, uint32_t *handle_out,
                struct drm_nova_gem_info *info_out)
{
   struct drm_nova_gem_create create = {
      .flags = gem_flags,
      .size = size_B,
   };
   if (drmIoctl(dev->fd, DRM_IOCTL_NOVA_GEM_CREATE, &create))
      return -errno;

   simple_mtx_lock(&dev->handles_mutex);
   nvkmd_nova_dev_ref_handle_locked(dev, create.handle);
   simple_mtx_unlock(&dev->handles_mutex);

   *info_out = (struct drm_nova_gem_info) { .handle = create.handle };
   if (drmIoctl(dev->fd, DRM_IOCTL_NOVA_GEM_INFO, info_out)) {
      int err = -errno;
      nvkmd_nova_dev_unref_handle(dev, create.handle);
      return err;
   }

   *handle_out = create.handle;

   return 0;
}

VkResult
nvkmd_nova_alloc_tiled_mem(struct nvkmd_dev *_dev,
                           struct vk_object_base *log_obj,
                           uint64_t size_B, uint64_t align_B,
                           uint8_t pte_kind, uint16_t tile_mode,
                           enum nvkmd_mem_flags flags,
                           struct nvkmd_mem **mem_out)
{
   struct nvkmd_nova_dev *dev = nvkmd_nova_dev(_dev);
   const struct nv_device_info *dev_info = &dev->base.pdev->dev_info;
   int err;

   /* Only one placement flag may be specified */
   assert(util_bitcount(flags & NVKMD_MEM_PLACEMENT_FLAGS) == 1);

   /* Not all callers care about alignment: enforce the page size, which nova
    * maps memory in.
    */
   assert(util_is_power_of_two_or_zero64(align_B));
   align_B = MAX2(align_B, dev->base.pdev->bind_align_B);
   size_B = align64(size_B, align_B);

   /* Local memory is VRAM, unless VRAM is full. The CPU only maps VRAM
    * objects that are made visible to it, through the VRAM BAR. When the BAR
    * covers all of VRAM, NVK maps any VRAM, to copy images from the host, so
    * all of it is made visible.
    */
   const bool local = flags & NVKMD_MEM_LOCAL;
   const bool vram = (flags & NVKMD_MEM_VRAM) ||
                     (local && dev_info->vram_size_B > 0);
   const bool vram_all_visible = dev_info->bar_size_B >= dev_info->vram_size_B;
   uint32_t gem_flags = 0;
   if (vram) {
      gem_flags |= DRM_NOVA_GEM_VRAM;
      if ((flags & NVKMD_MEM_CAN_MAP) || vram_all_visible)
         gem_flags |= DRM_NOVA_GEM_CPU_VISIBLE;
   }

   uint32_t handle;
   struct drm_nova_gem_info info;
   err = nova_gem_create(dev, size_B, gem_flags, &handle, &info);
   if ((err == -ENOMEM || err == -ENOSPC) && vram && local)
      err = nova_gem_create(dev, size_B, 0, &handle, &info);
   if (err)
      return vk_errorf(log_obj, VK_ERROR_OUT_OF_DEVICE_MEMORY,
                       "GEM_CREATE failed: %s", strerror(-err));

   /* System memory maps are cached and coherent, VRAM maps write-combined. */
   flags |= NVKMD_MEM_COHERENT;

   return create_mem_or_unref_handle(dev, log_obj, flags, handle, &info,
                                     pte_kind, align_B, mem_out);
}

VkResult
nvkmd_nova_import_dma_buf(struct nvkmd_dev *_dev,
                          struct vk_object_base *log_obj,
                          int fd, struct nvkmd_mem **mem_out)
{
   struct nvkmd_nova_dev *dev = nvkmd_nova_dev(_dev);
   uint32_t handle;

   /* The handle and the count of its uses change together, so that a
    * concurrent free doesn't close the handle the import is given.
    */
   simple_mtx_lock(&dev->handles_mutex);
   if (drmPrimeFDToHandle(dev->fd, fd, &handle)) {
      simple_mtx_unlock(&dev->handles_mutex);
      return vk_errorf(log_obj, VK_ERROR_INVALID_EXTERNAL_HANDLE,
                       "Failed to import dma-buf: %m");
   }
   nvkmd_nova_dev_ref_handle_locked(dev, handle);
   simple_mtx_unlock(&dev->handles_mutex);

   struct drm_nova_gem_info info = { .handle = handle };
   if (drmIoctl(dev->fd, DRM_IOCTL_NOVA_GEM_INFO, &info)) {
      VkResult result = vk_errorf(log_obj, VK_ERROR_INVALID_EXTERNAL_HANDLE,
                                  "GEM_INFO failed: %m");
      nvkmd_nova_dev_unref_handle(dev, handle);
      return result;
   }

   /* Shared memory is local: where it is is up to whoever allocated it. */
   enum nvkmd_mem_flags flags = NVKMD_MEM_SHARED | NVKMD_MEM_LOCAL |
                                NVKMD_MEM_COHERENT;
   if (info.flags & DRM_NOVA_GEM_CPU_VISIBLE)
      flags |= NVKMD_MEM_CAN_MAP;

   return create_mem_or_unref_handle(dev, log_obj, flags, handle, &info,
                                     0 /* pte_kind */, 0 /* align_B */,
                                     mem_out);
}

static void
nvkmd_nova_mem_free(struct nvkmd_mem *_mem)
{
   struct nvkmd_nova_mem *mem = nvkmd_nova_mem(_mem);
   struct nvkmd_nova_dev *dev = nvkmd_nova_dev(_mem->dev);

   nvkmd_va_free(mem->base.va);
   nvkmd_nova_dev_unref_handle(dev, mem->handle);
   FREE(mem);
}

static VkResult
nvkmd_nova_mem_map(struct nvkmd_mem *_mem,
                   struct vk_object_base *log_obj,
                   enum nvkmd_mem_map_flags map_flags,
                   void *fixed_addr,
                   void **map_out)
{
   struct nvkmd_nova_mem *mem = nvkmd_nova_mem(_mem);
   struct nvkmd_nova_dev *dev = nvkmd_nova_dev(_mem->dev);

   int prot = 0;
   if (map_flags & NVKMD_MEM_MAP_RD)
      prot |= PROT_READ;
   if (map_flags & NVKMD_MEM_MAP_WR)
      prot |= PROT_WRITE;

   int flags = MAP_SHARED;
   if (map_flags & NVKMD_MEM_MAP_FIXED)
      flags |= MAP_FIXED;

   void *map = mmap(fixed_addr, mem->base.size_B, prot, flags,
                    dev->fd, mem->mmap_offset);
   if (map == MAP_FAILED)
      return vk_errorf(log_obj, VK_ERROR_MEMORY_MAP_FAILED, "mmap failed: %m");

   *map_out = map;

   return VK_SUCCESS;
}

static void
nvkmd_nova_mem_unmap(struct nvkmd_mem *_mem,
                     enum nvkmd_mem_map_flags flags,
                     void *map)
{
   munmap(map, _mem->size_B);
}

static VkResult
nvkmd_nova_mem_overmap(struct nvkmd_mem *_mem,
                       struct vk_object_base *log_obj,
                       enum nvkmd_mem_map_flags flags,
                       void *map)
{
   void *new_map = mmap(map, _mem->size_B, PROT_NONE,
                        MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED, -1, 0);
   if (new_map == MAP_FAILED) {
      return vk_errorf(log_obj, VK_ERROR_MEMORY_MAP_FAILED,
                       "Failed to map over original mapping");
   }

   assert(new_map == map);

   return VK_SUCCESS;
}

static VkResult
nvkmd_nova_mem_export_dma_buf(struct nvkmd_mem *_mem,
                              struct vk_object_base *log_obj,
                              int *fd_out)
{
   struct nvkmd_nova_mem *mem = nvkmd_nova_mem(_mem);
   struct nvkmd_nova_dev *dev = nvkmd_nova_dev(_mem->dev);

   if (drmPrimeHandleToFD(dev->fd, mem->handle, DRM_CLOEXEC | DRM_RDWR,
                          fd_out))
      return vk_errorf(log_obj, VK_ERROR_TOO_MANY_OBJECTS,
                       "Failed to export dma-buf: %m");

   return VK_SUCCESS;
}

static uint32_t
nvkmd_nova_mem_log_handle(struct nvkmd_mem *mem)
{
   return nvkmd_nova_mem(mem)->handle;
}

const struct nvkmd_mem_ops nvkmd_nova_mem_ops = {
   .free = nvkmd_nova_mem_free,
   .map = nvkmd_nova_mem_map,
   .unmap = nvkmd_nova_mem_unmap,
   .overmap = nvkmd_nova_mem_overmap,
   .export_dma_buf = nvkmd_nova_mem_export_dma_buf,
   .log_handle = nvkmd_nova_mem_log_handle,
};

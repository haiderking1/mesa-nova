/*
 * Copyright © 2026 soka
 * SPDX-License-Identifier: MIT
 */

#include "nvkmd_nova.h"

#include "util/hash_table.h"
#include "util/u_memory.h"
#include "vk_log.h"

#include <errno.h>
#include <fcntl.h>
#include <unistd.h>
#include <xf86drm.h>

VkResult
nvkmd_nova_create_dev(struct nvkmd_pdev *_pdev,
                      struct vk_object_base *log_obj,
                      struct nvkmd_dev **dev_out)
{
   struct nvkmd_nova_pdev *pdev = nvkmd_nova_pdev(_pdev);
   struct drm_nova_info_vm vm_info = {};
   VkResult result;

   struct nvkmd_nova_dev *dev = CALLOC_STRUCT(nvkmd_nova_dev);
   if (dev == NULL)
      return vk_error(log_obj, VK_ERROR_OUT_OF_HOST_MEMORY);

   dev->base.ops = &nvkmd_nova_dev_ops;
   dev->base.pdev = &pdev->base;
   dev->base.va_start = 0;
   dev->base.va_end = NVKMD_NOVA_KERNEL_START;

   list_inithead(&dev->base.mems);
   simple_mtx_init(&dev->base.mems_mutex, mtx_plain);

   /* Each device has a GPU virtual address space of its own, which is that of
    * its file.
    */
   drmDevicePtr drm_device = NULL;
   if (drmGetDeviceFromDevId(pdev->base.drm.render_dev, 0, &drm_device)) {
      result = vk_errorf(log_obj, VK_ERROR_INITIALIZATION_FAILED,
                         "Failed to get DRM device: %m");
      goto fail_dev;
   }
   dev->fd = open(drm_device->nodes[DRM_NODE_RENDER], O_RDWR | O_CLOEXEC);
   drmFreeDevice(&drm_device);
   if (dev->fd < 0) {
      result = vk_errorf(log_obj, VK_ERROR_INITIALIZATION_FAILED,
                         "Failed to open the DRM device: %m");
      goto fail_dev;
   }

   struct drm_nova_info info = {
      .id = DRM_NOVA_INFO_VM,
      .size = sizeof(vm_info),
      .data = (uintptr_t)&vm_info,
   };
   if (drmIoctl(dev->fd, DRM_IOCTL_NOVA_INFO, &info) ||
       vm_info.va_size <= NVKMD_NOVA_KERNEL_START) {
      result = vk_errorf(log_obj, VK_ERROR_INITIALIZATION_FAILED,
                         "No room for the kernel's mappings above 0x%llx",
                         (unsigned long long)NVKMD_NOVA_KERNEL_START);
      goto fail_fd;
   }

   struct drm_nova_vm_init vm_init = {
      .kernel_managed_addr = NVKMD_NOVA_KERNEL_START,
      .kernel_managed_size = vm_info.va_size - NVKMD_NOVA_KERNEL_START,
   };
   if (drmIoctl(dev->fd, DRM_IOCTL_NOVA_VM_INIT, &vm_init)) {
      result = vk_errorf(log_obj, VK_ERROR_INITIALIZATION_FAILED,
                         "VM_INIT failed: %m");
      goto fail_fd;
   }

   simple_mtx_init(&dev->heap_mutex, mtx_plain);

   STATIC_ASSERT(NVKMD_NOVA_HEAP_START < NVKMD_NOVA_HEAP_END);
   util_vma_heap_init(&dev->heap, NVKMD_NOVA_HEAP_START,
                      NVKMD_NOVA_HEAP_END - NVKMD_NOVA_HEAP_START);

   STATIC_ASSERT(NVKMD_NOVA_REPLAY_HEAP_START < NVKMD_NOVA_REPLAY_HEAP_END);
   util_vma_heap_init(&dev->replay_heap, NVKMD_NOVA_REPLAY_HEAP_START,
                      NVKMD_NOVA_REPLAY_HEAP_END - NVKMD_NOVA_REPLAY_HEAP_START);

   simple_mtx_init(&dev->handles_mutex, mtx_plain);
   dev->handles = _mesa_hash_table_create_u32_keys(NULL);
   if (dev->handles == NULL) {
      result = vk_error(log_obj, VK_ERROR_OUT_OF_HOST_MEMORY);
      goto fail_heaps;
   }

   *dev_out = &dev->base;

   return VK_SUCCESS;

fail_heaps:
   simple_mtx_destroy(&dev->handles_mutex);
   util_vma_heap_finish(&dev->heap);
   util_vma_heap_finish(&dev->replay_heap);
   simple_mtx_destroy(&dev->heap_mutex);
fail_fd:
   close(dev->fd);
fail_dev:
   FREE(dev);
   return result;
}

static void
nvkmd_nova_dev_destroy(struct nvkmd_dev *_dev)
{
   struct nvkmd_nova_dev *dev = nvkmd_nova_dev(_dev);

   _mesa_hash_table_destroy(dev->handles, NULL);
   simple_mtx_destroy(&dev->handles_mutex);
   util_vma_heap_finish(&dev->heap);
   util_vma_heap_finish(&dev->replay_heap);
   simple_mtx_destroy(&dev->heap_mutex);
   close(dev->fd);
   FREE(dev);
}

static uint64_t
nvkmd_nova_dev_get_gpu_timestamp(struct nvkmd_dev *_dev)
{
   struct nvkmd_nova_dev *dev = nvkmd_nova_dev(_dev);
   struct drm_nova_info_timestamp timestamp = {};

   struct drm_nova_info info = {
      .id = DRM_NOVA_INFO_TIMESTAMP,
      .size = sizeof(timestamp),
      .data = (uintptr_t)&timestamp,
   };
   if (drmIoctl(dev->fd, DRM_IOCTL_NOVA_INFO, &info))
      return 0;

   return timestamp.timestamp;
}

static int
nvkmd_nova_dev_get_drm_fd(struct nvkmd_dev *_dev)
{
   return nvkmd_nova_dev(_dev)->fd;
}

const struct nvkmd_dev_ops nvkmd_nova_dev_ops = {
   .destroy = nvkmd_nova_dev_destroy,
   .get_gpu_timestamp = nvkmd_nova_dev_get_gpu_timestamp,
   .get_drm_fd = nvkmd_nova_dev_get_drm_fd,
   .alloc_mem = nvkmd_nova_alloc_mem,
   .alloc_tiled_mem = nvkmd_nova_alloc_tiled_mem,
   .import_dma_buf = nvkmd_nova_import_dma_buf,
   .alloc_va = nvkmd_nova_alloc_va,
   .create_ctx = nvkmd_nova_create_ctx,
};

/* Counts one more use of `handle`, by a new memory object. */
void
nvkmd_nova_dev_ref_handle_locked(struct nvkmd_nova_dev *dev, uint32_t handle)
{
   simple_mtx_assert_locked(&dev->handles_mutex);

   struct hash_entry *entry = _mesa_hash_table_search(dev->handles,
                                                      (void *)(uintptr_t)handle);
   if (entry != NULL) {
      entry->data = (void *)((uintptr_t)entry->data + 1);
   } else {
      _mesa_hash_table_insert(dev->handles, (void *)(uintptr_t)handle,
                              (void *)(uintptr_t)1);
   }
}

/* Counts one less use of `handle`, and closes it if that was the last. */
void
nvkmd_nova_dev_unref_handle(struct nvkmd_nova_dev *dev, uint32_t handle)
{
   simple_mtx_lock(&dev->handles_mutex);

   struct hash_entry *entry = _mesa_hash_table_search(dev->handles,
                                                      (void *)(uintptr_t)handle);
   assert(entry != NULL && entry->data != NULL);
   entry->data = (void *)((uintptr_t)entry->data - 1);
   if (entry->data == NULL) {
      _mesa_hash_table_remove(dev->handles, entry);

      struct drm_gem_close close = { .handle = handle };
      drmIoctl(dev->fd, DRM_IOCTL_GEM_CLOSE, &close);
   }

   simple_mtx_unlock(&dev->handles_mutex);
}

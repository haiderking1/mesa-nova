/*
 * Copyright © 2026 soka
 * SPDX-License-Identifier: MIT
 */

#include "nvkmd_nova.h"

#include "util/u_math.h"
#include "util/u_memory.h"
#include "vk_log.h"

#include <inttypes.h>
#include <xf86drm.h>

static VkResult MUST_CHECK
alloc_heap_addr_locked(struct nvkmd_nova_dev *dev,
                       struct vk_object_base *log_obj,
                       enum nvkmd_va_flags flags,
                       uint64_t size_B, uint64_t align_B,
                       uint64_t fixed_addr, uint64_t *addr_out)
{
   if (flags & NVKMD_VA_ALLOC_FIXED) {
      assert(flags & NVKMD_VA_REPLAY);

      if (fixed_addr < NVKMD_NOVA_REPLAY_HEAP_START ||
          fixed_addr >= NVKMD_NOVA_REPLAY_HEAP_END) {
         return vk_errorf(log_obj, VK_ERROR_INVALID_OPAQUE_CAPTURE_ADDRESS,
                          "Capture address 0x%" PRIx64 " not in the replay "
                          "heap address range [0x%" PRIx64 ", 0x%" PRIx64 ")",
                          fixed_addr, NVKMD_NOVA_REPLAY_HEAP_START,
                          NVKMD_NOVA_REPLAY_HEAP_END);
      }

      if (fixed_addr & (align_B - 1)) {
         return vk_errorf(log_obj, VK_ERROR_INVALID_OPAQUE_CAPTURE_ADDRESS,
                          "Unaligned capture address: 0x%" PRIx64, fixed_addr);
      }

      if (!util_vma_heap_alloc_addr(&dev->replay_heap, fixed_addr, size_B)) {
         return vk_errorf(log_obj, VK_ERROR_INVALID_OPAQUE_CAPTURE_ADDRESS,
                          "Replay address collision: 0x%" PRIx64, fixed_addr);
      }

      *addr_out = fixed_addr;
   } else {
      struct util_vma_heap *heap = (flags & NVKMD_VA_REPLAY) ?
                                   &dev->replay_heap : &dev->heap;

      *addr_out = util_vma_heap_alloc(heap, size_B, align_B);
      if (*addr_out == 0)
         return vk_errorf(log_obj, VK_ERROR_OUT_OF_DEVICE_MEMORY,
                          "Failed to allocate virtual address range");
   }

   return VK_SUCCESS;
}

static void
free_heap_addr(struct nvkmd_nova_dev *dev, enum nvkmd_va_flags flags,
               uint64_t addr, uint64_t size_B)
{
   simple_mtx_lock(&dev->heap_mutex);
   if (flags & NVKMD_VA_REPLAY)
      util_vma_heap_free(&dev->replay_heap, addr, size_B);
   else
      util_vma_heap_free(&dev->heap, addr, size_B);
   simple_mtx_unlock(&dev->heap_mutex);
}

static VkResult MUST_CHECK
vm_bind(struct nvkmd_nova_dev *dev,
        struct vk_object_base *log_obj,
        struct drm_nova_vm_bind_op *op)
{
   struct drm_nova_vm_bind bind = {
      .op_count = 1,
      .op_ptr = (uintptr_t)op,
   };

   if (drmIoctl(dev->fd, DRM_IOCTL_NOVA_VM_BIND, &bind))
      return vk_errorf(log_obj, VK_ERROR_UNKNOWN, "VM_BIND failed: %m");

   return VK_SUCCESS;
}

VkResult
nvkmd_nova_alloc_va(struct nvkmd_dev *_dev,
                    struct vk_object_base *log_obj,
                    enum nvkmd_va_flags flags, uint8_t pte_kind,
                    uint64_t size_B, uint64_t align_B,
                    uint64_t fixed_addr, struct nvkmd_va **va_out)
{
   struct nvkmd_nova_dev *dev = nvkmd_nova_dev(_dev);
   VkResult result;

   /* nova doesn't map sparse ranges yet: kmd_info.has_sparse is false. */
   if (flags & NVKMD_VA_SPARSE)
      return vk_errorf(log_obj, VK_ERROR_FEATURE_NOT_PRESENT,
                       "nova doesn't map sparse ranges yet");

   struct nvkmd_nova_va *va = CALLOC_STRUCT(nvkmd_nova_va);
   if (va == NULL)
      return vk_error(log_obj, VK_ERROR_OUT_OF_HOST_MEMORY);

   /* Not all callers care about alignment: enforce the page size, which nova
    * maps memory in, and align the range to the alignment.
    */
   assert(util_is_power_of_two_or_zero64(align_B));
   align_B = MAX2(align_B, _dev->pdev->bind_align_B);
   size_B = align64(size_B, align_B);

   assert((fixed_addr == 0) == !(flags & NVKMD_VA_ALLOC_FIXED));

   simple_mtx_lock(&dev->heap_mutex);
   result = alloc_heap_addr_locked(dev, log_obj, flags, size_B, align_B,
                                   fixed_addr, &va->base.addr);
   simple_mtx_unlock(&dev->heap_mutex);
   if (result != VK_SUCCESS) {
      FREE(va);
      return result;
   }

   va->base.ops = &nvkmd_nova_va_ops;
   va->base.dev = &dev->base;
   va->base.flags = flags;
   va->base.pte_kind = pte_kind;
   va->base.size_B = size_B;

   *va_out = &va->base;

   return VK_SUCCESS;
}

static void
nvkmd_nova_va_free(struct nvkmd_va *_va)
{
   struct nvkmd_nova_dev *dev = nvkmd_nova_dev(_va->dev);

   struct drm_nova_vm_bind_op op = {
      .op = DRM_NOVA_VM_BIND_OP_UNMAP,
      .addr = _va->addr,
      .range = _va->size_B,
   };

   /* If unbinding fails, we leak the range */
   if (vm_bind(dev, NULL, &op) == VK_SUCCESS)
      free_heap_addr(dev, _va->flags, _va->addr, _va->size_B);

   FREE(nvkmd_nova_va(_va));
}

static VkResult
nvkmd_nova_va_bind_mem(struct nvkmd_va *_va,
                       struct vk_object_base *log_obj,
                       uint64_t va_offset_B,
                       struct nvkmd_mem *_mem,
                       uint64_t mem_offset_B,
                       uint64_t range_B)
{
   struct nvkmd_nova_dev *dev = nvkmd_nova_dev(_va->dev);

   assert(_mem->dev == _va->dev);

   struct drm_nova_vm_bind_op op = {
      .op = DRM_NOVA_VM_BIND_OP_MAP,
      .handle = nvkmd_nova_mem(_mem)->handle,
      .kind = _va->pte_kind,
      .addr = _va->addr + va_offset_B,
      .bo_offset = mem_offset_B,
      .range = range_B,
   };

   return vm_bind(dev, log_obj, &op);
}

static VkResult
nvkmd_nova_va_unbind(struct nvkmd_va *_va,
                     struct vk_object_base *log_obj,
                     uint64_t va_offset_B,
                     uint64_t range_B)
{
   struct nvkmd_nova_dev *dev = nvkmd_nova_dev(_va->dev);

   struct drm_nova_vm_bind_op op = {
      .op = DRM_NOVA_VM_BIND_OP_UNMAP,
      .addr = _va->addr + va_offset_B,
      .range = range_B,
   };

   return vm_bind(dev, log_obj, &op);
}

const struct nvkmd_va_ops nvkmd_nova_va_ops = {
   .free = nvkmd_nova_va_free,
   .bind_mem = nvkmd_nova_va_bind_mem,
   .unbind = nvkmd_nova_va_unbind,
};

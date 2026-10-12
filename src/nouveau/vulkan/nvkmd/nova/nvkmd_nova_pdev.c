/*
 * Copyright © 2026 soka
 * SPDX-License-Identifier: MIT
 */

#include "nvkmd_nova.h"

#include "nv_device_info.h"
#include "util/cache_ops.h"
#include "util/log.h"
#include "util/os_misc.h"
#include "util/u_memory.h"
#include "vk_log.h"

#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>
#include <xf86drm.h>

/* Queries the device information `id` of `fd` into the `size` bytes at `data`. */
static int
nova_info(int fd, uint32_t id, void *data, uint32_t size)
{
   struct drm_nova_info info = {
      .id = id,
      .size = size,
      .data = (uintptr_t)data,
   };

   if (drmIoctl(fd, DRM_IOCTL_NOVA_INFO, &info))
      return -errno;

   /* nova-drm returns how much it wrote: all of it, for the structs above. */
   return info.size == size ? 0 : -EINVAL;
}

/* Returns whether the DRM driver behind `fd` is nova-drm. */
static bool
drm_fd_is_nova(int fd)
{
   drmVersionPtr version = drmGetVersion(fd);
   if (version == NULL)
      return false;

   const bool is_nova = strcmp(version->name, "nova-drm") == 0;
   drmFreeVersion(version);

   return is_nova;
}

/* Fills `info` from what nova-drm reports of the GPU behind `fd`. */
static int
nova_device_info(int fd, const struct _drmDevice *drm_device,
                 struct nv_device_info *info)
{
   struct drm_nova_info_gpu gpu = {};
   struct drm_nova_info_graphics gr = {};
   int err;

   err = nova_info(fd, DRM_NOVA_INFO_GPU, &gpu, sizeof(gpu));
   if (err)
      return err;
   err = nova_info(fd, DRM_NOVA_INFO_GRAPHICS, &gr, sizeof(gr));
   if (err)
      return err;

   *info = (struct nv_device_info) {
      .type = NV_DEVICE_TYPE_DIS,
      .device_id = drm_device->deviceinfo.pci->device_id,
      /* nova-drm's chip IDs are the chipsets NVK knows them as. */
      .chipset = gpu.chipid,
      .pci = {
         .domain = drm_device->businfo.pci->domain,
         .bus = drm_device->businfo.pci->bus,
         .dev = drm_device->businfo.pci->dev,
         .func = drm_device->businfo.pci->func,
         .revision_id = drm_device->deviceinfo.pci->revision_id,
      },
      .sm = gr.sm_major * 10 + gr.sm_minor,
      .gpc_count = gr.gpc_count,
      .tpc_count = gr.tpc_count,
      .mp_per_tpc = gr.sm_per_tpc,
      .max_warps_per_mp = gr.max_warps_per_sm,
      /* nova runs work on copy engines of their own. */
      .has_transfer_queue = true,
      .cls_copy = gr.class_copy,
      .cls_eng2d = gr.class_2d,
      .cls_eng3d = gr.class_3d,
      .cls_m2mf = gr.class_m2mf,
      .cls_compute = gr.class_compute,
      .cls_gpfifo = gr.class_gpfifo,
      .vram_size_B = gpu.vram_size,
      .bar_size_B = gpu.vram_bar_size,
   };
   info->max_blocks_per_mp = nv_max_blocks_per_mp_for_sm(info->sm);
   nv_device_info_init_shared_mem_sizes(info);

   /* Video decoding runs on channels of the video decoder engine, which not
    * every GPU has, nor every nova-drm knows.
    */
   struct drm_nova_info_video_decoder dec = {};
   err = nova_info(fd, DRM_NOVA_INFO_VIDEO_DECODER, &dec, sizeof(dec));
   if (err == 0) {
      info->cls_vdec = dec.class_video_decoder;
      info->has_video = true;
   } else if (err != -ENODEV && err != -EINVAL) {
      return err;
   }

   /* The marketing name of the board, if known, else nova-core's. */
   const char *name =
      nv_device_name_for_chip(drm_device->deviceinfo.pci->device_id,
                              drm_device->deviceinfo.pci->subdevice_id,
                              drm_device->deviceinfo.pci->subvendor_id);
   if (name == NULL)
      name = (const char *)gpu.gpu_name;
   snprintf(info->device_name, sizeof(info->device_name), "%s", name);
   snprintf(info->chipset_name, sizeof(info->chipset_name), "%s",
            (const char *)gpu.gpu_short_name);

   return 0;
}

VkResult
nvkmd_nova_try_create_pdev(struct _drmDevice *drm_device,
                           struct vk_object_base *log_obj,
                           enum nvk_debug debug_flags,
                           struct nvkmd_pdev **pdev_out)
{
   if (!(drm_device->available_nodes & (1 << DRM_NODE_RENDER)))
      return VK_ERROR_INCOMPATIBLE_DRIVER;

   if (drm_device->bustype != DRM_BUS_PCI ||
       drm_device->deviceinfo.pci->vendor_id != NVIDIA_VENDOR_ID)
      return VK_ERROR_INCOMPATIBLE_DRIVER;

   const char *path = drm_device->nodes[DRM_NODE_RENDER];
   int fd = open(path, O_RDWR | O_CLOEXEC);
   if (fd < 0)
      return VK_ERROR_INCOMPATIBLE_DRIVER;

   if (!drm_fd_is_nova(fd)) {
      mesa_logd("nvkmd: %s is not driven by nova-drm", path);
      close(fd);
      return VK_ERROR_INCOMPATIBLE_DRIVER;
   }

   struct stat st;
   if (fstat(fd, &st)) {
      close(fd);
      return vk_errorf(log_obj, VK_ERROR_INITIALIZATION_FAILED,
                       "fstat() failed on %s: %m", path);
   }

   struct nvkmd_nova_pdev *pdev = CALLOC_STRUCT(nvkmd_nova_pdev);
   if (pdev == NULL) {
      close(fd);
      return vk_error(log_obj, VK_ERROR_OUT_OF_HOST_MEMORY);
   }

   int err = nova_device_info(fd, drm_device, &pdev->base.dev_info);
   if (err) {
      close(fd);
      FREE(pdev);
      return vk_errorf(log_obj, VK_ERROR_INITIALIZATION_FAILED,
                       "Failed to query the GPU from nova: %s",
                       strerror(-err));
   }

   pdev->base.ops = &nvkmd_nova_pdev_ops;
   pdev->base.debug_flags = debug_flags;
   /* Tiled memory is mapped with the PTE kind it's allocated with. Shared
    * images give their layouts as DRM format modifiers.
    */
   pdev->base.kmd_info = (struct nvkmd_info) {
      .has_alloc_tiled = true,
      .has_dma_buf = true,
      .has_map_fixed = true,
      .has_overmap = true,
   };

   pdev->base.dev_info.nc_atom_size_B = util_cache_granularity();

   /* nova maps memory in pages of the OS's size. */
   uint64_t os_page_size;
   os_get_page_size(&os_page_size);
   assert(os_page_size <= UINT32_MAX);
   pdev->base.bind_align_B = os_page_size;

   pdev->base.drm.render_dev = st.st_rdev;

   /* DRM primary is optional */
   if ((drm_device->available_nodes & (1 << DRM_NODE_PRIMARY)) &&
       !stat(drm_device->nodes[DRM_NODE_PRIMARY], &st))
      pdev->base.drm.primary_dev = st.st_rdev;

   pdev->primary_fd = -1;

   pdev->syncobj_sync_type = vk_drm_syncobj_get_type(fd);
   pdev->sync_types[0] = &pdev->syncobj_sync_type;
   pdev->sync_types[1] = NULL;
   pdev->base.sync_types = pdev->sync_types;

   close(fd);

   *pdev_out = &pdev->base;

   return VK_SUCCESS;
}

static void
nvkmd_nova_pdev_destroy(struct nvkmd_pdev *_pdev)
{
   struct nvkmd_nova_pdev *pdev = nvkmd_nova_pdev(_pdev);

   if (pdev->primary_fd >= 0)
      close(pdev->primary_fd);

   FREE(pdev);
}

static uint64_t
nvkmd_nova_pdev_get_vram_used(struct nvkmd_pdev *_pdev)
{
   /* Not reported: kmd_info.has_get_vram_used is false. */
   return 0;
}

static int
nvkmd_nova_pdev_get_drm_primary_fd(struct nvkmd_pdev *_pdev)
{
   struct nvkmd_nova_pdev *pdev = nvkmd_nova_pdev(_pdev);

   if (pdev->primary_fd >= 0)
      return pdev->primary_fd;

   if (pdev->base.drm.primary_dev == 0)
      return -1;

   drmDevicePtr drm_device = NULL;
   int ret = drmGetDeviceFromDevId(pdev->base.drm.primary_dev, 0, &drm_device);
   if (ret != 0)
      return -1;

   pdev->primary_fd = open(drm_device->nodes[DRM_NODE_PRIMARY],
                           O_RDWR | O_CLOEXEC);
   drmFreeDevice(&drm_device);

   return pdev->primary_fd;
}

const struct nvkmd_pdev_ops nvkmd_nova_pdev_ops = {
   .destroy = nvkmd_nova_pdev_destroy,
   .get_vram_used = nvkmd_nova_pdev_get_vram_used,
   .get_drm_primary_fd = nvkmd_nova_pdev_get_drm_primary_fd,
   .create_dev = nvkmd_nova_create_dev,
};

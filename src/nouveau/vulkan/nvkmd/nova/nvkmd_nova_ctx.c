/*
 * Copyright © 2026 soka
 * SPDX-License-Identifier: MIT
 */

#include "nvkmd_nova.h"

#include "util/u_memory.h"
#include "vk_drm_syncobj.h"
#include "vk_log.h"

#include <errno.h>
#include <xf86drm.h>

static VkResult
nvkmd_nova_exec_ctx_flush(struct nvkmd_ctx *_ctx,
                          struct vk_object_base *log_obj);

static VkResult
add_sync(struct nvkmd_ctx *ctx, struct vk_object_base *log_obj,
         uint32_t *sync_count, struct drm_nova_sync *syncs,
         struct vk_sync *sync, uint64_t sync_value)
{
   if (unlikely(*sync_count >= NVKMD_NOVA_MAX_SYNCS)) {
      VkResult result = nvkmd_nova_exec_ctx_flush(ctx, log_obj);
      if (result != VK_SUCCESS)
         return result;
   }

   struct vk_drm_syncobj *syncobj = vk_sync_as_drm_syncobj(sync);
   assert(syncobj != NULL);

   syncs[(*sync_count)++] = (struct drm_nova_sync) {
      .flags = sync_value ? DRM_NOVA_SYNC_TIMELINE_SYNCOBJ :
                            DRM_NOVA_SYNC_SYNCOBJ,
      .handle = syncobj->syncobj,
      .timeline_value = sync_value,
   };

   return VK_SUCCESS;
}

/* Returns the channel to create for `engines`: on the graphics engine with
 * the classes they need, a copy object coming with it, or on a copy engine.
 */
static VkResult
channel_for_engines(enum nvkmd_engines engines,
                    struct drm_nova_channel_create *create)
{
   const enum nvkmd_engines graphics =
      NVKMD_ENGINE_3D | NVKMD_ENGINE_COMPUTE |
      NVKMD_ENGINE_2D | NVKMD_ENGINE_M2MF;

   if (engines & ~(graphics | NVKMD_ENGINE_COPY))
      return VK_ERROR_FEATURE_NOT_PRESENT;

   if (!(engines & graphics)) {
      *create = (struct drm_nova_channel_create) {
         .engine = DRM_NOVA_ENGINE_COPY,
      };
      return VK_SUCCESS;
   }

   uint32_t classes = 0;
   if (engines & NVKMD_ENGINE_3D)
      classes |= DRM_NOVA_CLASS_3D;
   if (engines & NVKMD_ENGINE_COMPUTE)
      classes |= DRM_NOVA_CLASS_COMPUTE;
   if (engines & NVKMD_ENGINE_2D)
      classes |= DRM_NOVA_CLASS_2D;
   if (engines & NVKMD_ENGINE_M2MF)
      classes |= DRM_NOVA_CLASS_M2MF;

   *create = (struct drm_nova_channel_create) {
      .engine = DRM_NOVA_ENGINE_GRAPHICS,
      .classes = classes,
   };
   return VK_SUCCESS;
}

VkResult
nvkmd_nova_create_ctx(struct nvkmd_dev *_dev,
                      struct vk_object_base *log_obj,
                      enum nvkmd_engines engines,
                      struct nvkmd_ctx **ctx_out)
{
   struct nvkmd_nova_dev *dev = nvkmd_nova_dev(_dev);
   struct drm_nova_channel_create create;
   VkResult result;

   /* Binding queues are only for sparse resources: kmd_info.has_sparse is
    * false.
    */
   result = channel_for_engines(engines, &create);
   if (result != VK_SUCCESS)
      return vk_errorf(log_obj, result, "nova has no channel for engines 0x%x",
                       engines);

   struct drm_nova_info_exec exec_info = {};
   struct drm_nova_info info = {
      .id = DRM_NOVA_INFO_EXEC,
      .size = sizeof(exec_info),
      .data = (uintptr_t)&exec_info,
   };
   if (drmIoctl(dev->fd, DRM_IOCTL_NOVA_INFO, &info))
      return vk_errorf(log_obj, VK_ERROR_INITIALIZATION_FAILED,
                       "INFO_EXEC failed: %m");

   struct nvkmd_nova_exec_ctx *ctx = CALLOC_STRUCT(nvkmd_nova_exec_ctx);
   if (ctx == NULL)
      return vk_error(log_obj, VK_ERROR_OUT_OF_HOST_MEMORY);

   if (drmIoctl(dev->fd, DRM_IOCTL_NOVA_CHANNEL_CREATE, &create)) {
      result = errno == ENOSPC ? VK_ERROR_TOO_MANY_OBJECTS :
                                 VK_ERROR_INITIALIZATION_FAILED;
      result = vk_errorf(log_obj, result, "CHANNEL_CREATE failed: %m");
      goto fail_ctx;
   }

   if (drmSyncobjCreate(dev->fd, 0, &ctx->syncobj)) {
      result = vk_error(log_obj, VK_ERROR_OUT_OF_HOST_MEMORY);
      goto fail_channel;
   }

   ctx->base.ops = &nvkmd_nova_exec_ctx_ops;
   ctx->base.dev = &dev->base;
   ctx->fd = dev->fd;
   ctx->channel = create.channel;
   ctx->max_push = exec_info.push_max;
   ctx->req_push = UTIL_DYNARRAY_INIT;
   ctx->req = (struct drm_nova_exec) {
      .channel = create.channel,
      .wait_ptr = (uintptr_t)&ctx->req_wait,
      .sig_ptr = (uintptr_t)&ctx->req_sig,
   };

   *ctx_out = &ctx->base;

   return VK_SUCCESS;

fail_channel:;
   struct drm_nova_channel_destroy destroy = { .channel = create.channel };
   drmIoctl(dev->fd, DRM_IOCTL_NOVA_CHANNEL_DESTROY, &destroy);
fail_ctx:
   FREE(ctx);
   return result;
}

static void
nvkmd_nova_exec_ctx_destroy(struct nvkmd_ctx *_ctx)
{
   struct nvkmd_nova_exec_ctx *ctx = nvkmd_nova_exec_ctx(_ctx);

   util_dynarray_fini(&ctx->req_push);

   ASSERTED int err = drmSyncobjDestroy(ctx->fd, ctx->syncobj);
   assert(err == 0);

   /* nova destroys the channel once the work it runs is done. */
   struct drm_nova_channel_destroy destroy = { .channel = ctx->channel };
   drmIoctl(ctx->fd, DRM_IOCTL_NOVA_CHANNEL_DESTROY, &destroy);

   FREE(ctx);
}

static VkResult
nvkmd_nova_exec_ctx_wait(struct nvkmd_ctx *_ctx,
                         struct vk_object_base *log_obj,
                         uint32_t wait_count,
                         const struct vk_sync_wait *waits)
{
   struct nvkmd_nova_exec_ctx *ctx = nvkmd_nova_exec_ctx(_ctx);

   for (uint32_t i = 0; i < wait_count; i++) {
      VkResult result = add_sync(&ctx->base, log_obj, &ctx->req.wait_count,
                                 ctx->req_wait, waits[i].sync,
                                 waits[i].wait_value);
      if (result != VK_SUCCESS)
         return result;
   }

   return VK_SUCCESS;
}

static VkResult
nvkmd_nova_exec_ctx_flush(struct nvkmd_ctx *_ctx,
                          struct vk_object_base *log_obj)
{
   struct nvkmd_nova_exec_ctx *ctx = nvkmd_nova_exec_ctx(_ctx);

   if (ctx->req.push_count == 0 &&
       ctx->req.wait_count == 0 &&
       ctx->req.sig_count == 0)
      return VK_SUCCESS;

   ctx->req.push_ptr = (uintptr_t)util_dynarray_begin(&ctx->req_push);

   if (drmIoctl(ctx->fd, DRM_IOCTL_NOVA_EXEC, &ctx->req)) {
      VkResult result = errno == ENODEV ? VK_ERROR_DEVICE_LOST :
                                          VK_ERROR_UNKNOWN;
      return vk_errorf(log_obj, result, "EXEC failed: %m");
   }

   util_dynarray_clear(&ctx->req_push);
   ctx->req.push_count = 0;
   ctx->req.wait_count = 0;
   ctx->req.sig_count = 0;

   return VK_SUCCESS;
}

static VkResult
nvkmd_nova_exec_ctx_exec(struct nvkmd_ctx *_ctx,
                         struct vk_object_base *log_obj,
                         uint32_t exec_count,
                         const struct nvkmd_ctx_exec *execs)
{
   struct nvkmd_nova_exec_ctx *ctx = nvkmd_nova_exec_ctx(_ctx);

   const uint32_t size_needed = MIN2(
      ctx->req_push.size + exec_count * sizeof(struct drm_nova_exec_push),
      ctx->max_push * sizeof(struct drm_nova_exec_push));
   if (util_dynarray_ensure_cap(&ctx->req_push, size_needed) == NULL)
      return vk_error(log_obj, VK_ERROR_OUT_OF_HOST_MEMORY);

   for (uint32_t i = 0; i < exec_count; i++) {
      /* A push that ends in an incomplete method must be submitted along
       * with the pushes that complete it.
       */
      uint32_t incomplete_count = 0;
      for (uint32_t j = i; j < exec_count; j++) {
         if (!execs[j].incomplete)
            break;

         /* The last exec cannot be incomplete */
         assert(j < exec_count - 1);
         incomplete_count++;
      }

      assert(incomplete_count < ctx->max_push);
      if (unlikely(ctx->req.push_count + incomplete_count >= ctx->max_push)) {
         VkResult result = nvkmd_nova_exec_ctx_flush(&ctx->base, log_obj);
         if (result != VK_SUCCESS)
            return result;
      }

      assert((execs[i].addr % 4) == 0 && (execs[i].size_B % 4) == 0);

      struct drm_nova_exec_push push = {
         .va = execs[i].addr,
         .range = execs[i].size_B,
         .flags = execs[i].no_prefetch ? DRM_NOVA_EXEC_PUSH_NO_PREFETCH : 0,
      };
      util_dynarray_append(&ctx->req_push, push);
      ctx->req.push_count++;
   }

   return VK_SUCCESS;
}

static VkResult
nvkmd_nova_exec_ctx_signal(struct nvkmd_ctx *_ctx,
                           struct vk_object_base *log_obj,
                           uint32_t signal_count,
                           const struct vk_sync_signal *signals)
{
   struct nvkmd_nova_exec_ctx *ctx = nvkmd_nova_exec_ctx(_ctx);

   for (uint32_t i = 0; i < signal_count; i++) {
      VkResult result = add_sync(&ctx->base, log_obj, &ctx->req.sig_count,
                                 ctx->req_sig, signals[i].sync,
                                 signals[i].signal_value);
      if (result != VK_SUCCESS)
         return result;
   }

   return nvkmd_nova_exec_ctx_flush(&ctx->base, log_obj);
}

static VkResult
nvkmd_nova_exec_ctx_sync(struct nvkmd_ctx *_ctx,
                         struct vk_object_base *log_obj)
{
   struct nvkmd_nova_exec_ctx *ctx = nvkmd_nova_exec_ctx(_ctx);
   VkResult result;

   if (unlikely(ctx->req.sig_count >= NVKMD_NOVA_MAX_SYNCS)) {
      result = nvkmd_nova_exec_ctx_flush(&ctx->base, log_obj);
      if (result != VK_SUCCESS)
         return result;
   }

   ctx->req_sig[ctx->req.sig_count++] = (struct drm_nova_sync) {
      .flags = DRM_NOVA_SYNC_SYNCOBJ,
      .handle = ctx->syncobj,
   };

   result = nvkmd_nova_exec_ctx_flush(&ctx->base, log_obj);
   if (result != VK_SUCCESS)
      return result;

   if (drmSyncobjWait(ctx->fd, &ctx->syncobj, 1, INT64_MAX, 0, NULL))
      return vk_errorf(log_obj, VK_ERROR_UNKNOWN,
                       "DRM_SYNCOBJ_WAIT failed: %m");

   /* An EXEC without push buffers fails once the channel runs no more work,
    * such as after its work failed.
    */
   struct drm_nova_exec empty = { .channel = ctx->channel };
   if (drmIoctl(ctx->fd, DRM_IOCTL_NOVA_EXEC, &empty))
      return vk_errorf(log_obj, VK_ERROR_DEVICE_LOST, "EXEC failed: %m");

   return VK_SUCCESS;
}

const struct nvkmd_ctx_ops nvkmd_nova_exec_ctx_ops = {
   .destroy = nvkmd_nova_exec_ctx_destroy,
   .wait = nvkmd_nova_exec_ctx_wait,
   .exec = nvkmd_nova_exec_ctx_exec,
   .signal = nvkmd_nova_exec_ctx_signal,
   .flush = nvkmd_nova_exec_ctx_flush,
   .sync = nvkmd_nova_exec_ctx_sync,
};

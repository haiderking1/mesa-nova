/*
 * Copyright © 2021 Red Hat
 * Copyright © 2021 Valve Corporation
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice (including the next
 * paragraph) shall be included in all copies or substantial portions of the
 * Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 *
 * Authors:
 *    Dave Airlie <airlied@redhat.com>
 *    Mike Blumenkrantz <michael.blumenkrantz@gmail.com>
 */

#include "zink_context.h"
#include "zink_resource.h"
#include "zink_screen.h"
#include "zink_surface.h"
#include "zink_video.h"

#include "util/u_video.h"
#include "pipe/p_video_codec.h"
#include "vl/vl_video_buffer.h"

#include "util/vl_zscan_data.h"

#ifndef _WIN32
#include "drm-uapi/drm_fourcc.h"
#else
/* these won't actually be used */
#define DRM_FORMAT_MOD_INVALID 0
#define DRM_FORMAT_MOD_LINEAR 0
#endif

/* The size of the buffers the bitstreams of the frames are uploaded to one
 * after the other, larger frames getting a buffer of their own.
 */
#define BITSTREAM_UPLOAD_SIZE (4 * 1024 * 1024)

static void
zink_destroy_video_codec(struct pipe_video_codec *codec) {
   struct zink_video_codec *zvc = (struct zink_video_codec *)codec;
   struct zink_screen *screen = (struct zink_screen *)zvc->screen;

   simple_mtx_lock(screen->queue_lock);
   VKSCR(QueueWaitIdle)(screen->queue_video_decode);
   simple_mtx_unlock(screen->queue_lock);
   util_dynarray_foreach(&zvc->retired_views, VkImageView, view)
      VKSCR(DestroyImageView)(screen->dev, *view, NULL);
   util_dynarray_fini(&zvc->retired_views);
   if (!zvc->coincide_dpb) {
      for (unsigned i = 0; i < zvc->max_dpb_slots; i++)
         VKSCR(DestroyImageView)(screen->dev, zvc->dpb_resources[i].imageViewBinding, NULL);

      if (zvc->dpb_array) {
         pipe_resource_reference(&zvc->dpb_res[0], NULL);
      } else {
         for (unsigned i = 0; i < zvc->max_dpb_slots; i++) {
            pipe_resource_reference(&zvc->dpb_res[i], NULL);
         }
      }
   }
   u_upload_destroy(zvc->bitstream_mgr);
   zink_resource_reference(&zvc->bitstream_res, NULL);
   free(zvc->bs_data);
   for (unsigned i = 0; i < zvc->num_priv_mems; i++)
      zink_bo_unref(screen, zvc->priv_mems[i]);

   /* The pictures keep their data, which other codecs may use, and which is
    * freed with them: see picture_data().
    */
   zvc->decode_ctx->destroy(zvc->decode_ctx);
   /* ensure session is no longer in use */
   VKSCR(QueueWaitIdle)(screen->queue_video_decode);
   VKSCR(DestroyVideoSessionKHR)(screen->dev,
                                 zvc->session, NULL);
   free(zvc);
}

static void
zink_destroy_associated_data(void *data)
{
   struct zink_video_surf_data *surf = data;
   struct zink_screen *screen = surf->screen;
   simple_mtx_lock(screen->queue_lock);
   VKSCR(QueueWaitIdle)(screen->queue_video_decode);
   simple_mtx_unlock(screen->queue_lock);
   VKSCR(DestroyImageView)(screen->dev, surf->resource.imageViewBinding, NULL);

   /* destroy iv */
   free(surf);
}

/* With separate DPB and output pictures, creates the DPB image of the slot,
 * or the array of all the slots' images (slot 0), of the given size, and the
 * views of the slots' pictures in it.
 */
static void
create_dpb_image(struct zink_video_codec *zvc, unsigned slot,
                 uint32_t width, uint32_t height)
{
   struct zink_screen *screen = zink_screen(zvc->screen);
   const unsigned first = zvc->dpb_array ? 0 : slot;
   const unsigned count = zvc->dpb_array ? zvc->max_dpb_slots : 1;
   struct pipe_resource dpb_templ = {};
   struct pipe_surface psurf = {};

   dpb_templ.width0 = width;
   dpb_templ.height0 = height * 2;
   dpb_templ.depth0 = 1;
   dpb_templ.format = zvc->dpb_format;
   dpb_templ.target = zvc->dpb_array ? PIPE_TEXTURE_2D_ARRAY : PIPE_TEXTURE_2D;
   dpb_templ.usage = PIPE_USAGE_DEFAULT;
   dpb_templ.flags = ZINK_RESOURCE_FLAG_VIDEO_DPB | PIPE_RESOURCE_FLAG_DONT_MAP_DIRECTLY | ZINK_RESOURCE_FLAG_INTERNAL_ONLY;
   dpb_templ.bind = ZINK_BIND_VIDEO;
   dpb_templ.array_size = count;

   psurf.format = zvc->dpb_format;

   zvc->dpb_res[slot] = screen->base.resource_create(&screen->base, &dpb_templ);
   for (unsigned i = first; i < first + count; i++) {
      psurf.first_layer = psurf.last_layer = i - first;
      VkImageViewCreateInfo ivci = create_ivci(screen, zink_resource(zvc->dpb_res[slot]),
                                               &psurf, PIPE_TEXTURE_2D);
      ivci.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
      zvc->dpb_resources[i].sType = VK_STRUCTURE_TYPE_VIDEO_PICTURE_RESOURCE_INFO_KHR;
      zvc->dpb_resources[i].pNext = NULL;
      zvc->dpb_resources[i].codedExtent.width = width;
      zvc->dpb_resources[i].codedExtent.height = height;
      VKSCR(CreateImageView)(screen->dev, &ivci, NULL, &zvc->dpb_resources[i].imageViewBinding);
   }
}

/* With separate DPB and output pictures, makes the DPB image of the slot of
 * the picture surf at least the size of the frame decoded to it, as a VP9
 * frame may be larger than the frames before it. The slot is the one set up
 * for the frame, which holds no reference. A DPB array holds all the slots'
 * pictures, which are lost with it: without separate reference images,
 * Vulkan wants all the references in one image.
 */
static void
fit_dpb_image(struct zink_video_codec *zvc,
              const struct zink_video_surf_data *surf)
{
   const unsigned slot = zvc->dpb_array ? 0 : surf->dpb_index;
   const VkExtent2D have = zvc->dpb_resources[surf->dpb_index].codedExtent;
   const VkExtent2D need = surf->resource.codedExtent;

   if (zvc->coincide_dpb ||
       (need.width <= have.width && need.height <= have.height))
      return;

   for (unsigned i = 0; i < zvc->max_dpb_slots; i++) {
      if (i == surf->dpb_index || zvc->dpb_array)
         util_dynarray_append(&zvc->retired_views, zvc->dpb_resources[i].imageViewBinding);
   }
   /* The batches using the image keep it until they are done */
   pipe_resource_reference(&zvc->dpb_res[slot], NULL);
   create_dpb_image(zvc, slot, MAX2(need.width, have.width),
                    MAX2(need.height, have.height));
}

static void
zink_video_create_session(struct zink_video_codec *zvc,
                          uint32_t width, uint32_t height,
                          enum pipe_format format,
                          enum pipe_video_profile profile,
                          enum pipe_video_entrypoint entrypoint)
{
   struct zink_screen *screen = (struct zink_screen *)zvc->base.context->screen;
   uint32_t bit_depth = zink_video_get_format_bit_depth(format);
   struct zink_video_profile vk_profile = { 0 };
   enum pipe_video_format vid_format = u_reduce_video_profile(profile);
   /* VA creates surfaces before the stream profile is known, but the profile
    * must be specified at image creation. HIGH profile is most compatible, so
    * use that for now
    */
   if (vid_format == PIPE_VIDEO_FORMAT_MPEG4_AVC)
      profile = PIPE_VIDEO_PROFILE_MPEG4_AVC_HIGH;
   zink_video_fill_single_profile(screen, profile, bit_depth, &vk_profile);

   struct zink_video_caps_info caps_info = { 0 };
   zink_video_fill_caps(screen, profile,
                        entrypoint, bit_depth != 8, &caps_info);

   zvc->max_dpb_slots = caps_info.caps.maxDpbSlots;
   zvc->coincide_dpb = caps_info.dec_caps.flags & VK_VIDEO_DECODE_CAPABILITY_DPB_AND_OUTPUT_COINCIDE_BIT_KHR;
   zvc->dpb_array = !(caps_info.caps.flags & VK_VIDEO_CAPABILITY_SEPARATE_REFERENCE_IMAGES_BIT_KHR);
   zvc->dpb_format = format;
   if (!zvc->coincide_dpb) {
      if (zvc->dpb_array) {
         create_dpb_image(zvc, 0, width, height);
      } else {
         for (unsigned i = 0; i < zvc->max_dpb_slots; i++)
            create_dpb_image(zvc, i, width, height);
      }
   }

   const VkExtensionProperties h264_props = {
      VK_STD_VULKAN_VIDEO_CODEC_H264_DECODE_EXTENSION_NAME,
      VK_STD_VULKAN_VIDEO_CODEC_H264_DECODE_SPEC_VERSION
   };
   const VkExtensionProperties vp9_props = {
      VK_STD_VULKAN_VIDEO_CODEC_VP9_DECODE_EXTENSION_NAME,
      VK_STD_VULKAN_VIDEO_CODEC_VP9_DECODE_SPEC_VERSION
   };
   const VkExtensionProperties av1_props = {
      VK_STD_VULKAN_VIDEO_CODEC_AV1_DECODE_EXTENSION_NAME,
      VK_STD_VULKAN_VIDEO_CODEC_AV1_DECODE_SPEC_VERSION
   };

   VkVideoSessionCreateInfoKHR sci = { 0 };
   sci.sType = VK_STRUCTURE_TYPE_VIDEO_SESSION_CREATE_INFO_KHR;
   sci.queueFamilyIndex = screen->video_decode_queue;
   sci.pVideoProfile = &vk_profile.profile;
   /* A VP9 frame may be of any size, and use the frames before it as
    * references, and the frame contexts they saved, which are the session's,
    * whatever their size: the session is for the largest frames of the
    * profile, not only for the size of the first frames.
    */
   if (vid_format == PIPE_VIDEO_FORMAT_VP9) {
      sci.maxCodedExtent = caps_info.caps.maxCodedExtent;
   } else {
      sci.maxCodedExtent.width = width;
      sci.maxCodedExtent.height = height;
   }
   sci.pictureFormat = zink_get_format(screen, format);
   sci.referencePictureFormat = zink_get_format(screen, format);
   sci.maxActiveReferencePictures = caps_info.caps.maxActiveReferencePictures;
   sci.maxDpbSlots = caps_info.caps.maxDpbSlots;
   switch (vid_format) {
   case PIPE_VIDEO_FORMAT_VP9:
      sci.pStdHeaderVersion = &vp9_props;
      break;
   case PIPE_VIDEO_FORMAT_AV1:
      sci.pStdHeaderVersion = &av1_props;
      break;
   default:
      sci.pStdHeaderVersion = &h264_props;
      break;
   }

   VKSCR(CreateVideoSessionKHR)(screen->dev, &sci, NULL, &zvc->session);

   uint32_t mem_req_count;
   VKSCR(GetVideoSessionMemoryRequirementsKHR)(screen->dev, zvc->session,
                                               &mem_req_count, NULL);
   const uint32_t max_reqs = 8; //hacky
   VkVideoSessionMemoryRequirementsKHR session_memory_reqs[max_reqs];

   memset(session_memory_reqs, 0, sizeof(session_memory_reqs));
   for (uint32_t i = 0; i < mem_req_count; i++) {
      session_memory_reqs[i].sType = VK_STRUCTURE_TYPE_VIDEO_SESSION_MEMORY_REQUIREMENTS_KHR;
   }
   VKSCR(GetVideoSessionMemoryRequirementsKHR)(screen->dev, zvc->session,
                                               &mem_req_count, session_memory_reqs);

   zvc->num_priv_mems = mem_req_count;
   zvc->priv_mems = calloc(sizeof(struct zink_bo *), mem_req_count);
   assert(zvc->priv_mems);

   VkBindVideoSessionMemoryInfoKHR bind_memory[max_reqs];
   enum zink_heap heap = ZINK_HEAP_DEVICE_LOCAL;
   for (unsigned i = 0; i < mem_req_count; i++) {
      for (unsigned j = 0; !zvc->priv_mems[i] && j < screen->heap_count[heap]; j++) {
         if (!(session_memory_reqs[i].memoryRequirements.memoryTypeBits & BITFIELD_BIT(screen->heap_map[heap][j])))
            continue;
         zvc->priv_mems[i] = zink_bo(zink_bo_create(screen, session_memory_reqs[i].memoryRequirements.size, session_memory_reqs[i].memoryRequirements.alignment, heap, ZINK_ALLOC_NO_SUBALLOC, screen->heap_map[heap][j], NULL));
      }
      VkDeviceMemory mem = zink_bo_get_mem(zvc->priv_mems[i]);
      bind_memory[i].pNext = NULL;
      bind_memory[i].sType = VK_STRUCTURE_TYPE_BIND_VIDEO_SESSION_MEMORY_INFO_KHR;
      bind_memory[i].memory = mem;
      bind_memory[i].memoryBindIndex = session_memory_reqs[i].memoryBindIndex;
      bind_memory[i].memoryOffset = 0;
      bind_memory[i].memorySize = session_memory_reqs[i].memoryRequirements.size;
   }
   VKSCR(BindVideoSessionMemoryKHR)(screen->dev, zvc->session,
                                    mem_req_count, bind_memory);

   zvc->srcbuf_align = caps_info.caps.minBitstreamBufferSizeAlignment;
   zvc->srcbuf_offset_align = caps_info.caps.minBitstreamBufferOffsetAlignment;
}

static void
zink_begin_frame(struct pipe_video_codec *codec,
                 struct pipe_video_buffer *target,
                 struct pipe_picture_desc *picture)
{
   struct zink_video_codec *zvc = (struct zink_video_codec *)codec;
   struct zink_context *ctx = zink_context(zvc->decode_ctx);
   struct zink_screen *screen = zink_screen(ctx->base.screen);
   zink_resource_reference(&zvc->bitstream_res, NULL);
   zvc->bs_size = 0;

   if (!zvc->session)
      zink_video_create_session(zvc, codec->width, codec->height,
                                target->buffer_format,
                                codec->profile, codec->entrypoint);

   if (zvc->dpb_array) {
      zink_batch_reference_resource_rw(ctx, zink_resource(zvc->dpb_res[0]), true);
   } else {
      for (unsigned i = 0; i < zvc->max_dpb_slots; i++) {
         if (zvc->dpb_res[i])
            zink_batch_reference_resource_rw(ctx, zink_resource(zvc->dpb_res[i]), true);
      }
   }
}

static void
zink_decode_macroblock(struct pipe_video_codec *codec,
                       struct pipe_video_buffer *target,
                       struct pipe_picture_desc *picture,
                       const struct pipe_macroblock *macroblocks,
                       unsigned num_macroblocks)
{
}

static void
zink_decode_bitstream(struct pipe_video_codec *codec,
                        struct pipe_video_buffer *target,
                        struct pipe_picture_desc *picture,
                        unsigned num_buffers,
                        const void * const *buffers,
                        const unsigned *sizes)
{
   struct zink_video_codec *zvc = (struct zink_video_codec *)codec;
   for (unsigned i = 0; i < num_buffers; ++i) {
      if (zvc->bs_size + sizes[i] > zvc->bs_capacity) {
         size_t capacity = MAX2(zvc->bs_capacity * 2, zvc->bs_size + sizes[i]);
         uint8_t *data = realloc(zvc->bs_data, capacity);
         if (!data)
            return;
         zvc->bs_data = data;
         zvc->bs_capacity = capacity;
      }
      memcpy(zvc->bs_data + zvc->bs_size, buffers[i], sizes[i]);
      zvc->bs_size += sizes[i];
   }
}


static void
convert_pps_sps(struct pipe_picture_desc *picture,
                StdVideoH264SequenceParameterSet *vsps,
                StdVideoH264PictureParameterSet *vpps,
                StdVideoH264ScalingLists *pps_scaling_list)
{
   struct pipe_h264_picture_desc *h264 = (struct pipe_h264_picture_desc *)picture;
   struct pipe_h264_pps *pps = h264->pps;
   struct pipe_h264_sps *sps = pps->sps;

   vpps->flags.transform_8x8_mode_flag = pps->transform_8x8_mode_flag;
   vpps->flags.redundant_pic_cnt_present_flag = pps->redundant_pic_cnt_present_flag;
   vpps->flags.constrained_intra_pred_flag = pps->constrained_intra_pred_flag;
   vpps->flags.deblocking_filter_control_present_flag = pps->deblocking_filter_control_present_flag;
   vpps->flags.weighted_pred_flag = pps->weighted_pred_flag;
   vpps->flags.entropy_coding_mode_flag = pps->entropy_coding_mode_flag;
   vpps->flags.pic_scaling_matrix_present_flag = true;
   vpps->flags.bottom_field_pic_order_in_frame_present_flag = pps->bottom_field_pic_order_in_frame_present_flag;

   vpps->num_ref_idx_l0_default_active_minus1 = pps->num_ref_idx_l0_default_active_minus1;
   vpps->num_ref_idx_l1_default_active_minus1 = pps->num_ref_idx_l1_default_active_minus1;
   vpps->weighted_bipred_idc = pps->weighted_bipred_idc;
   vpps->pic_init_qp_minus26 = pps->pic_init_qp_minus26;
//   vpps->pic_init_qs_minus26;
   vpps->chroma_qp_index_offset = pps->chroma_qp_index_offset;
   vpps->second_chroma_qp_index_offset = pps->second_chroma_qp_index_offset;

   pps_scaling_list->scaling_list_present_mask = 0xff;

   /* have to convert scaling matrix through zscan */
   for (unsigned m = 0; m < 6; m++)
      for (unsigned q = 0; q < 16; q++)
         pps_scaling_list->ScalingList4x4[m][q] = pps->ScalingList4x4[m][vl_zscan_normal_16[q]];

   for (unsigned m = 0; m < 6; m++)
      for (unsigned q = 0; q < 64; q++)
         pps_scaling_list->ScalingList8x8[m][q] = pps->ScalingList8x8[m][vl_zscan_normal[q]];


   switch (h264->base.profile) {
   case PIPE_VIDEO_PROFILE_MPEG4_AVC_BASELINE:
   case PIPE_VIDEO_PROFILE_MPEG4_AVC_CONSTRAINED_BASELINE:
     vsps->profile_idc = STD_VIDEO_H264_PROFILE_IDC_BASELINE;
     break;
   case PIPE_VIDEO_PROFILE_MPEG4_AVC_MAIN:
     vsps->profile_idc = STD_VIDEO_H264_PROFILE_IDC_MAIN;
     break;
   case PIPE_VIDEO_PROFILE_MPEG4_AVC_HIGH:
     vsps->profile_idc = STD_VIDEO_H264_PROFILE_IDC_HIGH;
     break;
   default:
     assert(0);
   }
   vsps->level_idc = STD_VIDEO_H264_LEVEL_IDC_1_0;

   vsps->flags.direct_8x8_inference_flag = sps->direct_8x8_inference_flag;
   vsps->flags.mb_adaptive_frame_field_flag = sps->mb_adaptive_frame_field_flag;
   vsps->flags.frame_mbs_only_flag = sps->frame_mbs_only_flag;
   vsps->flags.delta_pic_order_always_zero_flag = sps->delta_pic_order_always_zero_flag;
   vsps->flags.separate_colour_plane_flag = sps->separate_colour_plane_flag;

   vsps->bit_depth_luma_minus8 = sps->bit_depth_luma_minus8;
   vsps->bit_depth_chroma_minus8 = sps->bit_depth_chroma_minus8;
   vsps->log2_max_frame_num_minus4 = sps->log2_max_frame_num_minus4;
   vsps->pic_order_cnt_type = sps->pic_order_cnt_type;
   vsps->log2_max_pic_order_cnt_lsb_minus4 = sps->log2_max_pic_order_cnt_lsb_minus4;
   vsps->max_num_ref_frames = h264->num_ref_frames;
   vsps->chroma_format_idc = sps->chroma_format_idc;
   vsps->pic_width_in_mbs_minus1 = sps->pic_width_in_mbs_minus1;
   /* no interlace support */
   vsps->pic_height_in_map_units_minus1 = sps->pic_height_in_mbs_minus1;
}

/* The serial of the last picture data created */
static uint64_t surf_serial;

static struct zink_video_surf_data *
create_surf(struct zink_video_codec *zvc,
            struct pipe_video_buffer *target,
            int dpb_index)
{
   struct zink_resource *luma = (struct zink_resource *)((struct vl_video_buffer *)target)->resources[0];
   struct zink_screen *screen = (struct zink_screen *)zvc->screen;
   struct zink_video_surf_data *surf = calloc(sizeof(*surf), 1);

   surf->screen = screen;
   surf->serial = p_atomic_inc_return(&surf_serial);
   surf->resource.sType = VK_STRUCTURE_TYPE_VIDEO_PICTURE_RESOURCE_INFO_KHR;
   surf->resource.pNext = NULL;
   surf->resource.codedExtent.width = target->width;
   surf->resource.codedExtent.height = target->height;
   surf->resource.imageViewBinding = VK_NULL_HANDLE;

   surf->dpb_index = dpb_index;

   struct pipe_surface psurf = {};
   psurf.format = target->buffer_format;
   VkImageViewCreateInfo ivci = create_ivci(screen, luma, &psurf, PIPE_TEXTURE_2D);
   ivci.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
   /* The view only decodes: the surface's other usage (sampling and
    * rendering) is its planes' views', which the multi-planar format of this
    * one doesn't have.
    */
   VkImageViewUsageCreateInfo usage_info = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_USAGE_CREATE_INFO,
      .pNext = ivci.pNext,
      .usage = luma->obj->vkusage & (VK_IMAGE_USAGE_VIDEO_DECODE_DST_BIT_KHR |
                                     VK_IMAGE_USAGE_VIDEO_DECODE_DPB_BIT_KHR),
   };
   ivci.pNext = &usage_info;

   VKSCR(CreateImageView)(screen->dev, &ivci, NULL, &surf->resource.imageViewBinding);

   return surf;
}

/* Uploads the frame's bitstream after the previous frames', padded to the
 * decoder's size alignment.
 */
static void
upload_bitstream(struct zink_context *ctx, struct zink_video_codec *zvc)
{
   struct pipe_resource *pres = NULL;
   void *ptr;

   zvc->bs_range = align64(zvc->bs_size, zvc->srcbuf_align);
   u_upload_alloc_ref(zvc->bitstream_mgr, 0, zvc->bs_range,
                      zvc->srcbuf_offset_align, &zvc->bs_offset, &pres, &ptr);
   memcpy(ptr, zvc->bs_data, zvc->bs_size);
   memset((uint8_t *)ptr + zvc->bs_size, 0, zvc->bs_range - zvc->bs_size);
   u_upload_unmap(zvc->bitstream_mgr);

   zvc->bitstream_res = zink_resource(pres);
   zink_batch_reference_resource_rw(ctx, zvc->bitstream_res, true);
}

static void
end_coding(struct zink_context *ctx,
           struct zink_video_codec *zvc,
           VkCommandBuffer cmdbuf)
{
   VkVideoEndCodingInfoKHR eci = { 0 };
   eci.sType = VK_STRUCTURE_TYPE_VIDEO_END_CODING_INFO_KHR;

   VKCTX(CmdEndVideoCodingKHR)(cmdbuf, &eci);

   struct zink_resource *zbs = zvc->bitstream_res;
   VkBufferMemoryBarrier2KHR bitstream_bmb;
   bitstream_bmb.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2_KHR;
   bitstream_bmb.pNext = NULL;
   bitstream_bmb.srcStageMask = VK_PIPELINE_STAGE_2_VIDEO_DECODE_BIT_KHR;
   bitstream_bmb.srcAccessMask = VK_ACCESS_2_VIDEO_DECODE_READ_BIT_KHR;
   bitstream_bmb.dstStageMask = VK_PIPELINE_STAGE_2_NONE_KHR;
   bitstream_bmb.dstAccessMask = VK_ACCESS_2_NONE_KHR;
   bitstream_bmb.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
   bitstream_bmb.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
   bitstream_bmb.buffer = zbs->obj->buffer;
   bitstream_bmb.size = zvc->bs_range;
   bitstream_bmb.offset = zvc->bs_offset;

   VkDependencyInfoKHR di = { 0 };
   di.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO_KHR;
   di.dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;
   di.bufferMemoryBarrierCount = 1;
   di.pBufferMemoryBarriers = &bitstream_bmb;
   VKCTX(CmdPipelineBarrier2KHR)(cmdbuf, &di);

   ctx->bs->video_params = zvc->params;
   ctx->bs->has_work = true;
   ctx->base.flush(&ctx->base, NULL, 0);
   zink_resource_reference(&zvc->bitstream_res, NULL);
}

static void
video_image_barrier(struct zink_context *ctx, VkCommandBuffer cmdbuf, struct zink_resource *res, VkImageLayout layout)
{
   struct zink_screen *screen = zink_screen(ctx->base.screen);
   bool is_foreign = res->queue == VK_QUEUE_FAMILY_FOREIGN_EXT;
   zink_batch_reference_resource_rw(ctx, res, true);
   if (res->layout == layout && !is_foreign)
      return;

   VkImageMemoryBarrier2KHR imb = { 0 };
   imb.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2_KHR;
   imb.srcStageMask = VK_PIPELINE_STAGE_2_NONE_KHR;
   imb.srcAccessMask = VK_ACCESS_2_NONE_KHR;
   imb.dstStageMask = VK_PIPELINE_STAGE_2_VIDEO_DECODE_BIT_KHR;
   imb.dstAccessMask = VK_ACCESS_2_VIDEO_DECODE_READ_BIT_KHR | VK_ACCESS_2_VIDEO_DECODE_WRITE_BIT_KHR;
   imb.oldLayout = res->layout;
   imb.newLayout = layout;
   imb.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
   imb.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
   if (is_foreign) {
      imb.srcQueueFamilyIndex = VK_QUEUE_FAMILY_FOREIGN_EXT;
      imb.dstQueueFamilyIndex = screen->video_decode_queue;
      res->queue = VK_QUEUE_FAMILY_IGNORED;
   }
   imb.image = res->obj->image;
   imb.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
   imb.subresourceRange.levelCount = VK_REMAINING_MIP_LEVELS;
   imb.subresourceRange.layerCount = VK_REMAINING_ARRAY_LAYERS;

   VkDependencyInfoKHR di = { 0 };
   di.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO_KHR;
   di.imageMemoryBarrierCount = 1;
   di.pImageMemoryBarriers = &imb;
   VKCTX(CmdPipelineBarrier2KHR)(cmdbuf, &di);

   res->layout = layout;
   res->obj->access = 0;
   res->obj->access_stage = 0;
}

static struct zink_resource *
video_buffer_resource(struct pipe_video_buffer *buf)
{
   return zink_resource(((struct vl_video_buffer *)buf)->resources[0]);
}

static int
zink_end_frame_h264(struct pipe_video_codec *codec,
                    struct pipe_video_buffer *target,
                    struct pipe_picture_desc *picture)
{
   struct zink_video_codec *zvc = (struct zink_video_codec *)codec;
   struct zink_context *ctx = zink_context(zvc->decode_ctx);
   struct zink_screen *screen = (struct zink_screen *)zvc->screen;
   struct pipe_h264_picture_desc *h264 = (struct pipe_h264_picture_desc *)picture;

   int i, j;

   upload_bitstream(ctx, zvc);

   struct zink_video_surf_data *surf = NULL;

   for (i = 0; i < ARRAY_SIZE(zvc->render_pic_list); i++) {
      for (j = 0; (h264->ref[j] != NULL) && (j < ARRAY_SIZE(zvc->render_pic_list)); j++) {
            if (zvc->render_pic_list[i] == h264->ref[j])
               break;
            if (j == ARRAY_SIZE(zvc->render_pic_list) - 1)
               zvc->render_pic_list[i] = NULL;
            else if (h264->ref[j + 1] == NULL)
               zvc->render_pic_list[i] = NULL;
      }
   }
   for (i = 0; i < ARRAY_SIZE(zvc->render_pic_list); ++i) {
      if (zvc->render_pic_list[i] && zvc->render_pic_list[i] == target) {
         if (target->codec != NULL) {
            surf = vl_video_buffer_get_associated_data(target, &zvc->base);
         } else {
            surf = create_surf(zvc, target, i);
            vl_video_buffer_set_associated_data(target, &zvc->base, (void *)surf,
                  &zink_destroy_associated_data);
         }
         break;
      }
   }
   if (i == ARRAY_SIZE(zvc->render_pic_list)) {
      for (i = 0; i < ARRAY_SIZE(zvc->render_pic_list); ++i) {
         if (!zvc->render_pic_list[i]) {
            zvc->render_pic_list[i] = target;
            surf = create_surf(zvc, target, i);
            vl_video_buffer_set_associated_data(target, &zvc->base, (void *)surf,
                  &zink_destroy_associated_data);
            break;
         }
      }
   }

   StdVideoDecodeH264ReferenceInfo h264_ref_info[17] = { 0 };
   VkVideoDecodeH264DpbSlotInfoKHR h264_dpb_info[17] = { 0 };
   VkVideoReferenceSlotInfoKHR ref_info[17] = { 0 };
   int num_ref_frames = 0;
   for (unsigned i = 0; i < h264->num_ref_frames; i++) {
      struct pipe_video_buffer *ref = h264->ref[i];

      if (!ref)
         break;
      num_ref_frames++;
      struct zink_video_surf_data *rsurf = vl_video_buffer_get_associated_data(ref, &zvc->base);

      h264_ref_info[i].PicOrderCnt[0] = h264->field_order_cnt_list[i][0];
      h264_ref_info[i].PicOrderCnt[1] = h264->field_order_cnt_list[i][1];
      h264_ref_info[i].FrameNum = h264->frame_num_list[i];
      h264_ref_info[i].flags.top_field_flag = h264->top_is_reference[i] && !h264->bottom_is_reference[i];
      h264_ref_info[i].flags.bottom_field_flag = h264->bottom_is_reference[i] && !h264->top_is_reference[i];
      h264_ref_info[i].flags.used_for_long_term_reference = h264->is_long_term[i];

      h264_dpb_info[i].sType = VK_STRUCTURE_TYPE_VIDEO_DECODE_H264_DPB_SLOT_INFO_KHR;
      h264_dpb_info[i].pStdReferenceInfo = &h264_ref_info[i];

      ref_info[i].sType = VK_STRUCTURE_TYPE_VIDEO_REFERENCE_SLOT_INFO_KHR;
      ref_info[i].slotIndex = rsurf->dpb_index;
      ref_info[i].pNext = &h264_dpb_info[i];
      ref_info[i].pPictureResource = zvc->coincide_dpb ? &rsurf->resource : &zvc->dpb_resources[rsurf->dpb_index];
   }

   StdVideoDecodeH264ReferenceInfo h264_setup_info = { 0 };
   VkVideoDecodeH264DpbSlotInfoKHR h264_dpb_setup_info = { 0 };
   VkVideoReferenceSlotInfoKHR setup_info;
   h264_setup_info.PicOrderCnt[0] = h264->field_order_cnt[0];
   h264_setup_info.PicOrderCnt[1] = h264->field_order_cnt[1];
   h264_setup_info.FrameNum = h264->frame_num;

   h264_dpb_setup_info.sType = VK_STRUCTURE_TYPE_VIDEO_DECODE_H264_DPB_SLOT_INFO_KHR;
   h264_dpb_setup_info.pStdReferenceInfo = &h264_setup_info;

   setup_info.sType = VK_STRUCTURE_TYPE_VIDEO_REFERENCE_SLOT_INFO_KHR;
   setup_info.slotIndex = surf->dpb_index;
   setup_info.pPictureResource = zvc->coincide_dpb ? &surf->resource : &zvc->dpb_resources[surf->dpb_index];
   setup_info.pNext = &h264_dpb_setup_info;
   /* this must be added to the list of 'bound resources' */
   ref_info[num_ref_frames] = setup_info;
   ref_info[num_ref_frames++].slotIndex = -1;

   StdVideoH264SequenceParameterSet sps = { 0 };
   StdVideoH264PictureParameterSet pps = { 0 };
   StdVideoH264ScalingLists pps_scaling_lists = { 0 };

   pps.pScalingLists = &pps_scaling_lists;
   convert_pps_sps(picture, &sps, &pps, &pps_scaling_lists);
   pps.num_ref_idx_l0_default_active_minus1 = h264->num_ref_idx_l0_active_minus1;
   pps.num_ref_idx_l1_default_active_minus1 = h264->num_ref_idx_l1_active_minus1;
   VkVideoDecodeH264SessionParametersAddInfoKHR h264add = { 0 };
   h264add.sType = VK_STRUCTURE_TYPE_VIDEO_DECODE_H264_SESSION_PARAMETERS_ADD_INFO_KHR;
   h264add.stdSPSCount = 1;
   h264add.pStdSPSs = &sps;
   h264add.stdPPSCount = 1;
   h264add.pStdPPSs = &pps;

   VkVideoDecodeH264SessionParametersCreateInfoKHR h264_create = { 0 };
   h264_create.sType = VK_STRUCTURE_TYPE_VIDEO_DECODE_H264_SESSION_PARAMETERS_CREATE_INFO_KHR;
   h264_create.maxStdSPSCount = 1;
   h264_create.maxStdPPSCount = 1;
   h264_create.pParametersAddInfo = &h264add;
   VkVideoSessionParametersCreateInfoKHR pci = { 0 };
   pci.sType = VK_STRUCTURE_TYPE_VIDEO_SESSION_PARAMETERS_CREATE_INFO_KHR;
   pci.pNext = &h264_create;
   pci.videoSession = zvc->session;

   VKSCR(CreateVideoSessionParametersKHR)(screen->dev, &pci, NULL, &zvc->params);

   VkCommandBuffer cmdbuf = ctx->bs->cmdbuf;

   if (zvc->coincide_dpb) {
      for (unsigned i = 0; i < h264->num_ref_frames && h264->ref[i]; i++)
         video_image_barrier(ctx, cmdbuf, video_buffer_resource(h264->ref[i]), VK_IMAGE_LAYOUT_VIDEO_DECODE_DPB_KHR);
      video_image_barrier(ctx, cmdbuf, video_buffer_resource(target), VK_IMAGE_LAYOUT_VIDEO_DECODE_DPB_KHR);
   } else {
      if (zvc->dpb_array) {
         video_image_barrier(ctx, cmdbuf, zink_resource(zvc->dpb_res[0]), VK_IMAGE_LAYOUT_VIDEO_DECODE_DPB_KHR);
      } else {
         for (unsigned i = 0; i < h264->num_ref_frames && h264->ref[i]; i++) {
            struct zink_video_surf_data *rsurf = vl_video_buffer_get_associated_data(h264->ref[i], &zvc->base);
            video_image_barrier(ctx, cmdbuf, zink_resource(zvc->dpb_res[rsurf->dpb_index]), VK_IMAGE_LAYOUT_VIDEO_DECODE_DPB_KHR);
         }
         video_image_barrier(ctx, cmdbuf, zink_resource(zvc->dpb_res[surf->dpb_index]), VK_IMAGE_LAYOUT_VIDEO_DECODE_DPB_KHR);
      }
      video_image_barrier(ctx, cmdbuf, video_buffer_resource(target), VK_IMAGE_LAYOUT_VIDEO_DECODE_DST_KHR);
   }

   VkVideoBeginCodingInfoKHR bci = { 0 };
   bci.sType = VK_STRUCTURE_TYPE_VIDEO_BEGIN_CODING_INFO_KHR;
   bci.videoSession = zvc->session;
   bci.videoSessionParameters = zvc->params;
   bci.referenceSlotCount = num_ref_frames;
   bci.pReferenceSlots = ref_info;

   VKSCR(CmdBeginVideoCodingKHR)(cmdbuf, &bci);

   if (!zvc->reset_sent) {
      VkVideoCodingControlInfoKHR cc = {0};
      cc.sType = VK_STRUCTURE_TYPE_VIDEO_CODING_CONTROL_INFO_KHR;
      cc.flags = VK_VIDEO_CODING_CONTROL_RESET_BIT_KHR;
      VKSCR(CmdControlVideoCodingKHR)(cmdbuf, &cc);
      zvc->reset_sent = true;
   }

   struct zink_resource *zbs = zvc->bitstream_res;
   VkBufferMemoryBarrier2KHR bitstream_bmb;
   bitstream_bmb.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2_KHR;
   bitstream_bmb.pNext = NULL;
   bitstream_bmb.srcStageMask = VK_PIPELINE_STAGE_2_HOST_BIT_KHR;
   bitstream_bmb.srcAccessMask = VK_ACCESS_2_HOST_WRITE_BIT_KHR;
   bitstream_bmb.dstStageMask = VK_PIPELINE_STAGE_2_VIDEO_DECODE_BIT_KHR;
   bitstream_bmb.dstAccessMask = VK_ACCESS_2_VIDEO_DECODE_READ_BIT_KHR;
   bitstream_bmb.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
   bitstream_bmb.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
   bitstream_bmb.buffer = zbs->obj->buffer;
   bitstream_bmb.size = zvc->bs_range;
   bitstream_bmb.offset = zvc->bs_offset;

   VkDependencyInfoKHR di = { 0 };
   di.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO_KHR;
   di.dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;
   di.bufferMemoryBarrierCount = 1;
   di.pBufferMemoryBarriers = &bitstream_bmb;
   VKSCR(CmdPipelineBarrier2KHR)(cmdbuf, &di);

   StdVideoDecodeH264PictureInfo pi = { 0 };
   pi.frame_num = h264->frame_num;
   pi.PicOrderCnt[0] = h264->field_order_cnt[0];
   pi.PicOrderCnt[1] = h264->field_order_cnt[1];
   pi.flags.field_pic_flag = h264->field_pic_flag;
   pi.flags.bottom_field_flag = h264->bottom_field_flag;
   pi.flags.is_reference = h264->is_reference;

   VkVideoDecodeH264PictureInfoKHR hpi = { 0 };
   hpi.sType = VK_STRUCTURE_TYPE_VIDEO_DECODE_H264_PICTURE_INFO_KHR;
   hpi.pStdPictureInfo = &pi;
   hpi.sliceCount = h264->slice_count;
   uint32_t slice_data_offsets[16] = {};
   for (unsigned i = 0; i < h264->slice_count; i++) {
      for (unsigned j = i + 1; j < h264->slice_count; j++) {
         slice_data_offsets[j] += h264->slice_parameter.slice_data_size[i] + 3;
      }
   }
   hpi.pSliceOffsets = slice_data_offsets;

   VkVideoDecodeInfoKHR vdi = { 0 };
   vdi.sType = VK_STRUCTURE_TYPE_VIDEO_DECODE_INFO_KHR;
   vdi.pNext = &hpi;
   vdi.referenceSlotCount = num_ref_frames - 1;
   vdi.pReferenceSlots = ref_info;
   vdi.pSetupReferenceSlot = &setup_info;
   vdi.dstPictureResource = surf->resource;


   vdi.srcBuffer = zbs->obj->buffer;
   vdi.srcBufferOffset = zvc->bs_offset;
   vdi.srcBufferRange = zvc->bs_range;
   VKSCR(CmdDecodeVideoKHR)(cmdbuf, &vdi);

   end_coding(ctx, zvc, cmdbuf);

   return 0;
}

/* The segmentation features of VP9, as FeatureEnabled and FeatureData take
 * them.
 */
enum vp9_seg_lvl {
   VP9_SEG_LVL_ALT_Q,
   VP9_SEG_LVL_ALT_L,
   VP9_SEG_LVL_REF_FRAME,
   VP9_SEG_LVL_SKIP,
};

/* The data zink keeps with a picture: its view and the size of the frame
 * decoded to it. It belongs to the picture, not to the codec which decoded
 * it, as a picture may be the reference of a frame another codec decodes:
 * VA-API decoders are created again when a VP9 stream changes size, with the
 * pictures of the old size as references.
 */
static struct zink_video_surf_data *
picture_data(struct pipe_video_buffer *buf)
{
   return buf->destroy_associated_data == zink_destroy_associated_data ?
          buf->associated_data : NULL;
}

/* The data of the picture buf, which it gets if it has none yet. */
static struct zink_video_surf_data *
video_buffer_surf(struct zink_video_codec *zvc, struct pipe_video_buffer *buf)
{
   struct zink_video_surf_data *surf = picture_data(buf);
   if (!surf) {
      surf = create_surf(zvc, buf, 0);
      vl_video_buffer_set_associated_data(buf, &zvc->base, surf,
                                          &zink_destroy_associated_data);
   }
   return surf;
}

/* Whether a DPB slot holds the picture surf. */
static bool
in_dpb(const struct zink_video_codec *zvc,
       const struct zink_video_surf_data *surf)
{
   for (unsigned i = 0; surf && i < ARRAY_SIZE(zvc->dpb_pics); i++) {
      if (zvc->dpb_pics[i] == surf->serial)
         return true;
   }
   return false;
}

/* Gives a free DPB slot to the picture surf. */
static void
take_dpb_slot(struct zink_video_codec *zvc, struct zink_video_surf_data *surf)
{
   unsigned slot = 0;
   while (slot < zvc->max_dpb_slots - 1 && zvc->dpb_pics[slot])
      slot++;
   assert(!zvc->dpb_pics[slot]);
   zvc->dpb_pics[slot] = surf->serial;
   surf->dpb_index = slot;
}

/* Has the DPB slots hold the pictures of refs, which other codecs may have
 * decoded, and target. The pictures keep their data.
 */
static struct zink_video_surf_data *
assign_dpb_slots(struct zink_video_codec *zvc, struct pipe_video_buffer *target,
                 struct pipe_video_buffer *const *refs, unsigned num_refs)
{
   for (unsigned i = 0; i < ARRAY_SIZE(zvc->dpb_pics); i++) {
      bool held = false;
      for (unsigned j = 0; j < num_refs; j++) {
         const struct zink_video_surf_data *rsurf =
            refs[j] ? picture_data(refs[j]) : NULL;
         held |= rsurf && zvc->dpb_pics[i] == rsurf->serial;
      }
      if (!held)
         zvc->dpb_pics[i] = 0;
   }

   /* Another codec's pictures are in no DPB slot of this one yet: with
    * separate DPB images, their picture is in the other codec's.
    */
   if (zvc->coincide_dpb) {
      for (unsigned j = 0; j < num_refs; j++) {
         struct zink_video_surf_data *rsurf = refs[j] ? picture_data(refs[j]) : NULL;
         if (rsurf && !in_dpb(zvc, rsurf))
            take_dpb_slot(zvc, rsurf);
      }
   }

   struct zink_video_surf_data *surf = video_buffer_surf(zvc, target);
   take_dpb_slot(zvc, surf);
   return surf;
}

/* The picture resource of the DPB slot of the picture surf, of the size of
 * the frame decoded to it.
 */
static VkVideoPictureResourceInfoKHR
dpb_picture_resource(const struct zink_video_codec *zvc,
                     const struct zink_video_surf_data *surf)
{
   VkVideoPictureResourceInfoKHR res = zvc->coincide_dpb ?
      surf->resource : zvc->dpb_resources[surf->dpb_index];
   res.codedExtent = surf->resource.codedExtent;
   return res;
}

static int
zink_end_frame_vp9(struct pipe_video_codec *codec,
                   struct pipe_video_buffer *target,
                   struct pipe_picture_desc *picture)
{
   struct zink_video_codec *zvc = (struct zink_video_codec *)codec;
   struct zink_context *ctx = zink_context(zvc->decode_ctx);
   struct zink_screen *screen = (struct zink_screen *)zvc->screen;
   struct pipe_vp9_picture_desc *vp9 = (struct pipe_vp9_picture_desc *)picture;
   const bool intra = vp9->picture_parameter.pic_fields.frame_type == 0 || vp9->picture_parameter.pic_fields.intra_only;

   upload_bitstream(ctx, zvc);

   /* The pictures of the 8 reference frames stay in the DPB. */
   struct zink_video_surf_data *surf =
      assign_dpb_slots(zvc, target, vp9->ref, STD_VIDEO_VP9_NUM_REF_FRAMES);
   surf->resource.codedExtent.width = vp9->picture_parameter.frame_width;
   surf->resource.codedExtent.height = vp9->picture_parameter.frame_height;
   fit_dpb_image(zvc, surf);

   /* The references of the frame, LAST, GOLDEN and ALTREF, which may be the
    * same picture, and the setup slot for the frame.
    */
   const unsigned ref_frames[STD_VIDEO_VP9_REFS_PER_FRAME] = {
      vp9->picture_parameter.pic_fields.last_ref_frame,
      vp9->picture_parameter.pic_fields.golden_ref_frame,
      vp9->picture_parameter.pic_fields.alt_ref_frame,
   };
   struct zink_video_surf_data *ref_surfs[STD_VIDEO_VP9_REFS_PER_FRAME];
   VkVideoPictureResourceInfoKHR ref_res[STD_VIDEO_VP9_REFS_PER_FRAME + 1];
   VkVideoReferenceSlotInfoKHR ref_info[STD_VIDEO_VP9_REFS_PER_FRAME + 1] = { 0 };
   VkVideoDecodeVP9PictureInfoKHR vpi = { 0 };
   vpi.sType = VK_STRUCTURE_TYPE_VIDEO_DECODE_VP9_PICTURE_INFO_KHR;
   unsigned num_refs = 0;
   for (unsigned i = 0; i < STD_VIDEO_VP9_REFS_PER_FRAME; i++) {
      vpi.referenceNameSlotIndices[i] = -1;
      ref_surfs[i] = NULL;
      if (intra || !vp9->ref[ref_frames[i]])
         continue;

      struct zink_video_surf_data *rsurf = picture_data(vp9->ref[ref_frames[i]]);
      if (!in_dpb(zvc, rsurf))
         continue;
      ref_surfs[i] = rsurf;
      vpi.referenceNameSlotIndices[i] = rsurf->dpb_index;

      bool listed = false;
      for (unsigned j = 0; j < num_refs; j++)
         listed |= ref_info[j].slotIndex == rsurf->dpb_index;
      if (listed)
         continue;

      ref_res[num_refs] = dpb_picture_resource(zvc, rsurf);
      ref_info[num_refs].sType = VK_STRUCTURE_TYPE_VIDEO_REFERENCE_SLOT_INFO_KHR;
      ref_info[num_refs].slotIndex = rsurf->dpb_index;
      ref_info[num_refs].pPictureResource = &ref_res[num_refs];
      num_refs++;
   }

   ref_res[num_refs] = dpb_picture_resource(zvc, surf);
   VkVideoReferenceSlotInfoKHR setup_info = { 0 };
   setup_info.sType = VK_STRUCTURE_TYPE_VIDEO_REFERENCE_SLOT_INFO_KHR;
   setup_info.slotIndex = surf->dpb_index;
   setup_info.pPictureResource = &ref_res[num_refs];
   /* this must be added to the list of 'bound resources' */
   ref_info[num_refs] = setup_info;
   ref_info[num_refs].slotIndex = -1;

   VkCommandBuffer cmdbuf = ctx->bs->cmdbuf;

   if (zvc->coincide_dpb) {
      for (unsigned i = 0; i < STD_VIDEO_VP9_REFS_PER_FRAME; i++) {
         if (ref_surfs[i])
            video_image_barrier(ctx, cmdbuf, video_buffer_resource(vp9->ref[ref_frames[i]]),
                                VK_IMAGE_LAYOUT_VIDEO_DECODE_DPB_KHR);
      }
      video_image_barrier(ctx, cmdbuf, video_buffer_resource(target), VK_IMAGE_LAYOUT_VIDEO_DECODE_DPB_KHR);
   } else {
      if (zvc->dpb_array) {
         video_image_barrier(ctx, cmdbuf, zink_resource(zvc->dpb_res[0]), VK_IMAGE_LAYOUT_VIDEO_DECODE_DPB_KHR);
      } else {
         for (unsigned i = 0; i < STD_VIDEO_VP9_REFS_PER_FRAME; i++) {
            if (ref_surfs[i])
               video_image_barrier(ctx, cmdbuf, zink_resource(zvc->dpb_res[ref_surfs[i]->dpb_index]),
                                   VK_IMAGE_LAYOUT_VIDEO_DECODE_DPB_KHR);
         }
         video_image_barrier(ctx, cmdbuf, zink_resource(zvc->dpb_res[surf->dpb_index]), VK_IMAGE_LAYOUT_VIDEO_DECODE_DPB_KHR);
      }
      video_image_barrier(ctx, cmdbuf, video_buffer_resource(target), VK_IMAGE_LAYOUT_VIDEO_DECODE_DST_KHR);
   }

   VkVideoBeginCodingInfoKHR bci = { 0 };
   bci.sType = VK_STRUCTURE_TYPE_VIDEO_BEGIN_CODING_INFO_KHR;
   bci.videoSession = zvc->session;
   bci.referenceSlotCount = num_refs + 1;
   bci.pReferenceSlots = ref_info;

   VKSCR(CmdBeginVideoCodingKHR)(cmdbuf, &bci);

   if (!zvc->reset_sent) {
      VkVideoCodingControlInfoKHR cc = {0};
      cc.sType = VK_STRUCTURE_TYPE_VIDEO_CODING_CONTROL_INFO_KHR;
      cc.flags = VK_VIDEO_CODING_CONTROL_RESET_BIT_KHR;
      VKSCR(CmdControlVideoCodingKHR)(cmdbuf, &cc);
      zvc->reset_sent = true;
   }

   struct zink_resource *zbs = zvc->bitstream_res;
   VkBufferMemoryBarrier2KHR bitstream_bmb;
   bitstream_bmb.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2_KHR;
   bitstream_bmb.pNext = NULL;
   bitstream_bmb.srcStageMask = VK_PIPELINE_STAGE_2_HOST_BIT_KHR;
   bitstream_bmb.srcAccessMask = VK_ACCESS_2_HOST_WRITE_BIT_KHR;
   bitstream_bmb.dstStageMask = VK_PIPELINE_STAGE_2_VIDEO_DECODE_BIT_KHR;
   bitstream_bmb.dstAccessMask = VK_ACCESS_2_VIDEO_DECODE_READ_BIT_KHR;
   bitstream_bmb.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
   bitstream_bmb.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
   bitstream_bmb.buffer = zbs->obj->buffer;
   bitstream_bmb.size = zvc->bs_range;
   bitstream_bmb.offset = zvc->bs_offset;

   VkDependencyInfoKHR di = { 0 };
   di.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO_KHR;
   di.dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;
   di.bufferMemoryBarrierCount = 1;
   di.pBufferMemoryBarriers = &bitstream_bmb;
   VKSCR(CmdPipelineBarrier2KHR)(cmdbuf, &di);

   StdVideoVP9ColorConfig color = { 0 };
   color.flags.color_range = vp9->picture_parameter.color_range;
   color.BitDepth = vp9->picture_parameter.bit_depth ? vp9->picture_parameter.bit_depth : 8;
   color.subsampling_x = vp9->picture_parameter.pic_fields.subsampling_x;
   color.subsampling_y = vp9->picture_parameter.pic_fields.subsampling_y;
   color.color_space = vp9->picture_parameter.color_space;

   StdVideoVP9LoopFilter lf = { 0 };
   lf.flags.loop_filter_delta_enabled = vp9->picture_parameter.mode_ref_delta_enabled;
   lf.flags.loop_filter_delta_update = vp9->picture_parameter.mode_ref_delta_update;
   lf.loop_filter_level = vp9->picture_parameter.filter_level;
   lf.loop_filter_sharpness = vp9->picture_parameter.sharpness_level;
   lf.update_ref_delta = vp9->picture_parameter.update_ref_delta;
   lf.update_mode_delta = vp9->picture_parameter.update_mode_delta;
   memcpy(lf.loop_filter_ref_deltas, vp9->picture_parameter.ref_deltas, sizeof(lf.loop_filter_ref_deltas));
   memcpy(lf.loop_filter_mode_deltas, vp9->picture_parameter.mode_deltas, sizeof(lf.loop_filter_mode_deltas));

   StdVideoVP9Segmentation seg = { 0 };
   seg.flags.segmentation_update_map = vp9->picture_parameter.pic_fields.segmentation_update_map;
   seg.flags.segmentation_temporal_update = vp9->picture_parameter.pic_fields.segmentation_temporal_update;
   seg.flags.segmentation_update_data = vp9->picture_parameter.pic_fields.segmentation_update_data;
   seg.flags.segmentation_abs_or_delta_update = vp9->picture_parameter.abs_delta;
   memcpy(seg.segmentation_tree_probs, vp9->picture_parameter.mb_segment_tree_probs, sizeof(seg.segmentation_tree_probs));
   memcpy(seg.segmentation_pred_prob, vp9->picture_parameter.segment_pred_probs, sizeof(seg.segmentation_pred_prob));
   for (unsigned i = 0; i < STD_VIDEO_VP9_MAX_SEGMENTS; i++) {
      const struct vp9_segment_parameter *sp = &vp9->slice_parameter.seg_param[i];
      seg.FeatureEnabled[i] =
         sp->alt_quant_enabled << VP9_SEG_LVL_ALT_Q |
         sp->alt_lf_enabled << VP9_SEG_LVL_ALT_L |
         sp->segment_flags.segment_reference_enabled << VP9_SEG_LVL_REF_FRAME |
         sp->segment_flags.segment_reference_skipped << VP9_SEG_LVL_SKIP;
      seg.FeatureData[i][VP9_SEG_LVL_ALT_Q] = sp->alt_quant;
      seg.FeatureData[i][VP9_SEG_LVL_ALT_L] = sp->alt_lf;
      seg.FeatureData[i][VP9_SEG_LVL_REF_FRAME] = sp->segment_flags.segment_reference;
   }

   StdVideoDecodeVP9PictureInfo pi = { 0 };
   pi.flags.error_resilient_mode = vp9->picture_parameter.pic_fields.error_resilient_mode;
   pi.flags.intra_only = vp9->picture_parameter.pic_fields.intra_only;
   pi.flags.allow_high_precision_mv = vp9->picture_parameter.pic_fields.allow_high_precision_mv;
   pi.flags.refresh_frame_context = vp9->picture_parameter.pic_fields.refresh_frame_context;
   pi.flags.frame_parallel_decoding_mode = vp9->picture_parameter.pic_fields.frame_parallel_decoding_mode;
   pi.flags.segmentation_enabled = vp9->picture_parameter.pic_fields.segmentation_enabled;
   pi.flags.show_frame = vp9->picture_parameter.pic_fields.show_frame;
   pi.flags.UsePrevFrameMvs = vp9->picture_parameter.pic_fields.use_prev_frame_mvs;
   pi.profile = (StdVideoVP9Profile)vp9->picture_parameter.profile;
   pi.frame_type = vp9->picture_parameter.pic_fields.frame_type ? STD_VIDEO_VP9_FRAME_TYPE_NON_KEY :
                                               STD_VIDEO_VP9_FRAME_TYPE_KEY;
   pi.frame_context_idx = vp9->picture_parameter.pic_fields.frame_context_idx;
   pi.reset_frame_context = vp9->picture_parameter.pic_fields.reset_frame_context;
   pi.refresh_frame_flags = vp9->picture_parameter.refresh_frame_flags;
   pi.ref_frame_sign_bias_mask =
      vp9->picture_parameter.pic_fields.last_ref_frame_sign_bias << STD_VIDEO_VP9_REFERENCE_NAME_LAST_FRAME |
      vp9->picture_parameter.pic_fields.golden_ref_frame_sign_bias << STD_VIDEO_VP9_REFERENCE_NAME_GOLDEN_FRAME |
      vp9->picture_parameter.pic_fields.alt_ref_frame_sign_bias << STD_VIDEO_VP9_REFERENCE_NAME_ALTREF_FRAME;
   pi.interpolation_filter = (StdVideoVP9InterpolationFilter)vp9->picture_parameter.pic_fields.mcomp_filter_type;
   pi.base_q_idx = vp9->picture_parameter.base_qindex;
   pi.delta_q_y_dc = vp9->picture_parameter.y_dc_delta_q;
   pi.delta_q_uv_dc = vp9->picture_parameter.uv_dc_delta_q;
   pi.delta_q_uv_ac = vp9->picture_parameter.uv_ac_delta_q;
   pi.tile_cols_log2 = vp9->picture_parameter.log2_tile_columns;
   pi.tile_rows_log2 = vp9->picture_parameter.log2_tile_rows;
   pi.pColorConfig = &color;
   pi.pLoopFilter = &lf;
   pi.pSegmentation = vp9->picture_parameter.pic_fields.segmentation_enabled ? &seg : NULL;

   vpi.pStdPictureInfo = &pi;
   /* The frame is in the bitstream buffer whole: its uncompressed header,
    * compressed header and tiles.
    */
   vpi.uncompressedHeaderOffset = 0;
   vpi.compressedHeaderOffset = vp9->picture_parameter.frame_header_length_in_bytes;
   vpi.tilesOffset = vp9->picture_parameter.frame_header_length_in_bytes + vp9->picture_parameter.first_partition_size;

   VkVideoDecodeInfoKHR vdi = { 0 };
   vdi.sType = VK_STRUCTURE_TYPE_VIDEO_DECODE_INFO_KHR;
   vdi.pNext = &vpi;
   vdi.referenceSlotCount = num_refs;
   vdi.pReferenceSlots = ref_info;
   vdi.pSetupReferenceSlot = &setup_info;
   vdi.dstPictureResource = surf->resource;
   vdi.srcBuffer = zbs->obj->buffer;
   vdi.srcBufferOffset = zvc->bs_offset;
   vdi.srcBufferRange = zvc->bs_range;
   VKSCR(CmdDecodeVideoKHR)(cmdbuf, &vdi);

   end_coding(ctx, zvc, cmdbuf);

   return 0;
}

/* Whether the driver applies the film grain of AV1 frames, which the
 * profile of the codec, and those of the pictures, then say.
 */
static bool
av1_film_grain(struct zink_screen *screen)
{
   if (!screen->video_av1_film_grain_queried) {
      VkVideoDecodeAV1ProfileInfoKHR av1 = {
         .sType = VK_STRUCTURE_TYPE_VIDEO_DECODE_AV1_PROFILE_INFO_KHR,
         .stdProfile = STD_VIDEO_AV1_PROFILE_MAIN,
         .filmGrainSupport = VK_TRUE,
      };
      VkVideoProfileInfoKHR profile = {
         .sType = VK_STRUCTURE_TYPE_VIDEO_PROFILE_INFO_KHR,
         .pNext = &av1,
         .videoCodecOperation = VK_VIDEO_CODEC_OPERATION_DECODE_AV1_BIT_KHR,
         .chromaSubsampling = VK_VIDEO_CHROMA_SUBSAMPLING_420_BIT_KHR,
         .lumaBitDepth = VK_VIDEO_COMPONENT_BIT_DEPTH_8_BIT_KHR,
         .chromaBitDepth = VK_VIDEO_COMPONENT_BIT_DEPTH_8_BIT_KHR,
      };
      VkVideoDecodeAV1CapabilitiesKHR av1_caps = {
         .sType = VK_STRUCTURE_TYPE_VIDEO_DECODE_AV1_CAPABILITIES_KHR,
      };
      VkVideoDecodeCapabilitiesKHR dec_caps = {
         .sType = VK_STRUCTURE_TYPE_VIDEO_DECODE_CAPABILITIES_KHR,
         .pNext = &av1_caps,
      };
      VkVideoCapabilitiesKHR caps = {
         .sType = VK_STRUCTURE_TYPE_VIDEO_CAPABILITIES_KHR,
         .pNext = &dec_caps,
      };
      screen->video_av1_film_grain =
         VKSCR(GetPhysicalDeviceVideoCapabilitiesKHR)(screen->pdev, &profile,
                                                      &caps) == VK_SUCCESS &&
         (dec_caps.flags & VK_VIDEO_DECODE_CAPABILITY_DPB_AND_OUTPUT_DISTINCT_BIT_KHR);
      screen->video_av1_film_grain_queried = true;
   }
   return screen->video_av1_film_grain;
}

/* The distance from order hint b to order hint a (get_relative_dist() of the
 * AV1 specification).
 */
static int
av1_relative_dist(const struct pipe_av1_picture_desc *av1, int a, int b)
{
   if (!av1->picture_parameter.seq_info_fields.enable_order_hint)
      return 0;

   const int m = 1 << av1->picture_parameter.order_hint_bits_minus_1;
   const int diff = a - b;
   return (diff & (m - 1)) - (diff & m);
}

/* The references of skip mode, by name (skip_mode_params() of the AV1
 * specification): the nearest forward reference, and the nearest backward
 * one, or the second nearest forward one.
 */
static void
av1_skip_mode_frames(const struct pipe_av1_picture_desc *av1,
                     const uint8_t order_hints[STD_VIDEO_AV1_NUM_REF_FRAMES],
                     uint8_t frames[STD_VIDEO_AV1_SKIP_MODE_FRAMES])
{
   const int order_hint = av1->picture_parameter.order_hint;
   int forward = -1, backward = -1, second_forward = -1;
   int forward_hint = 0, backward_hint = 0, second_forward_hint = 0;

   for (int i = 0; i < STD_VIDEO_AV1_REFS_PER_FRAME; i++) {
      const int hint = order_hints[STD_VIDEO_AV1_REFERENCE_NAME_LAST_FRAME + i];
      if (av1_relative_dist(av1, hint, order_hint) < 0) {
         if (forward < 0 || av1_relative_dist(av1, hint, forward_hint) > 0) {
            forward = i;
            forward_hint = hint;
         }
      } else if (av1_relative_dist(av1, hint, order_hint) > 0) {
         if (backward < 0 || av1_relative_dist(av1, hint, backward_hint) < 0) {
            backward = i;
            backward_hint = hint;
         }
      }
   }

   frames[0] = frames[1] = 0;
   if (forward < 0)
      return;

   if (backward >= 0) {
      frames[0] = STD_VIDEO_AV1_REFERENCE_NAME_LAST_FRAME + MIN2(forward, backward);
      frames[1] = STD_VIDEO_AV1_REFERENCE_NAME_LAST_FRAME + MAX2(forward, backward);
      return;
   }

   for (int i = 0; i < STD_VIDEO_AV1_REFS_PER_FRAME; i++) {
      const int hint = order_hints[STD_VIDEO_AV1_REFERENCE_NAME_LAST_FRAME + i];
      if (av1_relative_dist(av1, hint, forward_hint) < 0 &&
          (second_forward < 0 ||
           av1_relative_dist(av1, hint, second_forward_hint) > 0)) {
         second_forward = i;
         second_forward_hint = hint;
      }
   }
   if (second_forward >= 0) {
      frames[0] = STD_VIDEO_AV1_REFERENCE_NAME_LAST_FRAME + MIN2(forward, second_forward);
      frames[1] = STD_VIDEO_AV1_REFERENCE_NAME_LAST_FRAME + MAX2(forward, second_forward);
   }
}

/* Copies the picture src to dst, plane by plane, after the work of the video
 * context ctx so far.
 *
 * The video queue runs no transfers, so the copy context copies: its
 * submission waits for the video queue's last one, as submissions to another
 * queue than the last submission's do.
 */
static void
copy_video_buffer(struct zink_context *ctx, struct pipe_video_buffer *dst,
                  struct pipe_video_buffer *src)
{
   struct zink_screen *screen = zink_screen(ctx->base.screen);
   struct pipe_resource *src_planes[VL_NUM_COMPONENTS] = { 0 };
   struct pipe_resource *dst_planes[VL_NUM_COMPONENTS] = { 0 };
   src->get_resources(src, src_planes);
   dst->get_resources(dst, dst_planes);

   /* The planes of the pictures' size, which chroma planes are a part of. */
   const unsigned width = MIN2(src->width, dst->width);
   const unsigned height = MIN2(src->height, dst->height);

   ctx->base.flush(&ctx->base, NULL, 0);
   zink_screen_lock_context(screen);
   struct pipe_context *pctx = &screen->copy_context->base;
   for (unsigned i = 0; i < VL_NUM_COMPONENTS; i++) {
      if (!src_planes[i] || !dst_planes[i])
         continue;

      unsigned w = width, h = height;
      vl_video_buffer_adjust_size(&w, &h, i,
                                  pipe_format_to_chroma_format(src->buffer_format),
                                  false);
      struct pipe_box box;
      u_box_3d(0, 0, 0, w, h, 1, &box);
      pctx->resource_copy_region(pctx, dst_planes[i], 0, 0, 0, 0,
                                 src_planes[i], 0, &box);
   }
   pctx->flush(pctx, NULL, 0);
   zink_screen_unlock_context(screen);
}

static int
zink_end_frame_av1(struct pipe_video_codec *codec,
                   struct pipe_video_buffer *target,
                   struct pipe_picture_desc *picture)
{
   struct zink_video_codec *zvc = (struct zink_video_codec *)codec;
   struct zink_context *ctx = zink_context(zvc->decode_ctx);
   struct zink_screen *screen = (struct zink_screen *)zvc->screen;
   struct pipe_av1_picture_desc *av1 = (struct pipe_av1_picture_desc *)picture;
   const StdVideoAV1FrameType frame_type = av1->picture_parameter.pic_info_fields.frame_type;
   const bool intra = frame_type == STD_VIDEO_AV1_FRAME_TYPE_KEY ||
                      frame_type == STD_VIDEO_AV1_FRAME_TYPE_INTRA_ONLY;
   const unsigned sb_mi_log2 = av1->picture_parameter.seq_info_fields.use_128x128_superblock ? 5 : 4;

   upload_bitstream(ctx, zvc);

   /* The pictures of the 8 reference frames stay in the DPB. */
   struct zink_video_surf_data *surf =
      assign_dpb_slots(zvc, target, av1->ref, STD_VIDEO_AV1_NUM_REF_FRAMES);
   surf->resource.codedExtent.width = av1->picture_parameter.frame_width;
   surf->resource.codedExtent.height = av1->picture_parameter.frame_height;

   /* The picture shown, when the frame has film grain: the decoder outputs
    * the frame with its grain to it, if it applies the grain.
    */
   const bool apply_grain =
      av1->picture_parameter.film_grain_info.film_grain_info_fields.apply_grain &&
      av1->film_grain_target && av1->film_grain_target != target;
   const bool decoder_grain = apply_grain && av1_film_grain(screen);
   struct pipe_video_buffer *output = decoder_grain ? av1->film_grain_target : target;
   VkVideoPictureResourceInfoKHR output_res = surf->resource;
   if (decoder_grain) {
      output_res = video_buffer_surf(zvc, output)->resource;
      output_res.codedExtent = surf->resource.codedExtent;
   }

   /* The 7 references of the frame by name, which may be the same picture,
    * the order hints of the frames they hold, and the setup slot for the
    * frame.
    */
   uint8_t order_hints[STD_VIDEO_AV1_NUM_REF_FRAMES] = { 0 };
   struct zink_video_surf_data *ref_surfs[STD_VIDEO_AV1_REFS_PER_FRAME];
   VkVideoPictureResourceInfoKHR ref_res[STD_VIDEO_AV1_REFS_PER_FRAME + 1];
   VkVideoDecodeAV1DpbSlotInfoKHR ref_dpb[STD_VIDEO_AV1_REFS_PER_FRAME + 1];
   VkVideoReferenceSlotInfoKHR ref_info[STD_VIDEO_AV1_REFS_PER_FRAME + 1] = { 0 };
   VkVideoDecodeAV1PictureInfoKHR avpi = { 0 };
   avpi.sType = VK_STRUCTURE_TYPE_VIDEO_DECODE_AV1_PICTURE_INFO_KHR;
   unsigned num_refs = 0;
   for (unsigned i = 0; i < STD_VIDEO_AV1_REFS_PER_FRAME; i++) {
      struct pipe_video_buffer *ref = av1->ref[av1->picture_parameter.ref_frame_idx[i]];
      avpi.referenceNameSlotIndices[i] = -1;
      ref_surfs[i] = NULL;
      if (intra || !ref)
         continue;

      struct zink_video_surf_data *rsurf = picture_data(ref);
      if (!in_dpb(zvc, rsurf))
         continue;
      ref_surfs[i] = rsurf;
      avpi.referenceNameSlotIndices[i] = rsurf->dpb_index;
      order_hints[STD_VIDEO_AV1_REFERENCE_NAME_LAST_FRAME + i] = rsurf->av1.OrderHint;

      bool listed = false;
      for (unsigned j = 0; j < num_refs; j++)
         listed |= ref_info[j].slotIndex == rsurf->dpb_index;
      if (listed)
         continue;

      ref_res[num_refs] = dpb_picture_resource(zvc, rsurf);
      ref_dpb[num_refs] = (VkVideoDecodeAV1DpbSlotInfoKHR) {
         .sType = VK_STRUCTURE_TYPE_VIDEO_DECODE_AV1_DPB_SLOT_INFO_KHR,
         .pStdReferenceInfo = &rsurf->av1,
      };
      ref_info[num_refs].sType = VK_STRUCTURE_TYPE_VIDEO_REFERENCE_SLOT_INFO_KHR;
      ref_info[num_refs].pNext = &ref_dpb[num_refs];
      ref_info[num_refs].slotIndex = rsurf->dpb_index;
      ref_info[num_refs].pPictureResource = &ref_res[num_refs];
      num_refs++;
   }

   /* What the frames referencing this one will need of it. */
   StdVideoDecodeAV1ReferenceInfo *info = &surf->av1;
   memset(info, 0, sizeof(*info));
   info->flags.disable_frame_end_update_cdf = av1->picture_parameter.pic_info_fields.disable_frame_end_update_cdf;
   info->flags.segmentation_enabled = av1->picture_parameter.seg_info.segment_info_fields.enabled;
   info->frame_type = frame_type;
   info->OrderHint = av1->picture_parameter.order_hint;
   if (!intra) {
      memcpy(info->SavedOrderHints, order_hints, sizeof(order_hints));
      for (unsigned i = 0; i < STD_VIDEO_AV1_REFS_PER_FRAME; i++) {
         const unsigned name = STD_VIDEO_AV1_REFERENCE_NAME_LAST_FRAME + i;
         if (av1_relative_dist(av1, order_hints[name], av1->picture_parameter.order_hint) > 0)
            info->RefFrameSignBias |= 1 << name;
      }
   }

   ref_res[num_refs] = dpb_picture_resource(zvc, surf);
   ref_dpb[num_refs] = (VkVideoDecodeAV1DpbSlotInfoKHR) {
      .sType = VK_STRUCTURE_TYPE_VIDEO_DECODE_AV1_DPB_SLOT_INFO_KHR,
      .pStdReferenceInfo = info,
   };
   VkVideoReferenceSlotInfoKHR setup_info = { 0 };
   setup_info.sType = VK_STRUCTURE_TYPE_VIDEO_REFERENCE_SLOT_INFO_KHR;
   setup_info.pNext = &ref_dpb[num_refs];
   setup_info.slotIndex = surf->dpb_index;
   setup_info.pPictureResource = &ref_res[num_refs];
   /* this must be added to the list of 'bound resources' */
   ref_info[num_refs] = setup_info;
   ref_info[num_refs].slotIndex = -1;

   /* The sequence header, which VA-API gives with each frame. */
   StdVideoAV1ColorConfig color = { 0 };
   color.flags.mono_chrome = av1->picture_parameter.seq_info_fields.mono_chrome;
   color.flags.separate_uv_delta_q = av1->picture_parameter.u_dc_delta_q != av1->picture_parameter.v_dc_delta_q ||
                                     av1->picture_parameter.u_ac_delta_q != av1->picture_parameter.v_ac_delta_q;
   color.BitDepth = av1->picture_parameter.bit_depth_idx == 0 ? 8 : av1->picture_parameter.bit_depth_idx == 1 ? 10 : 12;
   color.subsampling_x = av1->picture_parameter.seq_info_fields.subsampling_x;
   color.subsampling_y = av1->picture_parameter.seq_info_fields.subsampling_y;
   color.matrix_coefficients = av1->picture_parameter.matrix_coefficients;

   StdVideoAV1SequenceHeader seq = { 0 };
   seq.flags.use_128x128_superblock = av1->picture_parameter.seq_info_fields.use_128x128_superblock;
   seq.flags.enable_filter_intra = av1->picture_parameter.seq_info_fields.enable_filter_intra;
   seq.flags.enable_intra_edge_filter = av1->picture_parameter.seq_info_fields.enable_intra_edge_filter;
   seq.flags.enable_interintra_compound = av1->picture_parameter.seq_info_fields.enable_interintra_compound;
   seq.flags.enable_masked_compound = av1->picture_parameter.seq_info_fields.enable_masked_compound;
   /* VA-API doesn't give these sequence flags, which enable what the frame
    * headers then give: allow_warped_motion, use_superres and the loop
    * restoration types.
    */
   seq.flags.enable_warped_motion = true;
   seq.flags.enable_superres = true;
   seq.flags.enable_restoration = true;
   seq.flags.enable_dual_filter = av1->picture_parameter.seq_info_fields.enable_dual_filter;
   seq.flags.enable_order_hint = av1->picture_parameter.seq_info_fields.enable_order_hint;
   seq.flags.enable_jnt_comp = av1->picture_parameter.seq_info_fields.enable_jnt_comp;
   seq.flags.enable_ref_frame_mvs = av1->picture_parameter.seq_info_fields.ref_frame_mvs;
   seq.flags.enable_cdef = av1->picture_parameter.seq_info_fields.enable_cdef;
   seq.flags.film_grain_params_present = av1->picture_parameter.seq_info_fields.film_grain_params_present;
   seq.seq_profile = av1->picture_parameter.profile;
   seq.frame_width_bits_minus_1 = MAX2(util_last_bit(av1->picture_parameter.max_width - 1), 1) - 1;
   seq.frame_height_bits_minus_1 = MAX2(util_last_bit(av1->picture_parameter.max_height - 1), 1) - 1;
   seq.max_frame_width_minus_1 = av1->picture_parameter.max_width - 1;
   seq.max_frame_height_minus_1 = av1->picture_parameter.max_height - 1;
   seq.order_hint_bits_minus_1 = av1->picture_parameter.order_hint_bits_minus_1;
   seq.seq_force_integer_mv = STD_VIDEO_AV1_SELECT_INTEGER_MV;
   seq.seq_force_screen_content_tools = STD_VIDEO_AV1_SELECT_SCREEN_CONTENT_TOOLS;
   seq.pColorConfig = &color;

   VkVideoDecodeAV1SessionParametersCreateInfoKHR av1_create = { 0 };
   av1_create.sType = VK_STRUCTURE_TYPE_VIDEO_DECODE_AV1_SESSION_PARAMETERS_CREATE_INFO_KHR;
   av1_create.pStdSequenceHeader = &seq;
   VkVideoSessionParametersCreateInfoKHR pci = { 0 };
   pci.sType = VK_STRUCTURE_TYPE_VIDEO_SESSION_PARAMETERS_CREATE_INFO_KHR;
   pci.pNext = &av1_create;
   pci.videoSession = zvc->session;
   VKSCR(CreateVideoSessionParametersKHR)(screen->dev, &pci, NULL, &zvc->params);

   VkCommandBuffer cmdbuf = ctx->bs->cmdbuf;

   if (zvc->coincide_dpb) {
      for (unsigned i = 0; i < STD_VIDEO_AV1_REFS_PER_FRAME; i++) {
         if (ref_surfs[i])
            video_image_barrier(ctx, cmdbuf, video_buffer_resource(av1->ref[av1->picture_parameter.ref_frame_idx[i]]),
                                VK_IMAGE_LAYOUT_VIDEO_DECODE_DPB_KHR);
      }
      video_image_barrier(ctx, cmdbuf, video_buffer_resource(target), VK_IMAGE_LAYOUT_VIDEO_DECODE_DPB_KHR);
      if (decoder_grain)
         video_image_barrier(ctx, cmdbuf, video_buffer_resource(output), VK_IMAGE_LAYOUT_VIDEO_DECODE_DST_KHR);
   } else {
      if (zvc->dpb_array) {
         video_image_barrier(ctx, cmdbuf, zink_resource(zvc->dpb_res[0]), VK_IMAGE_LAYOUT_VIDEO_DECODE_DPB_KHR);
      } else {
         for (unsigned i = 0; i < STD_VIDEO_AV1_REFS_PER_FRAME; i++) {
            if (ref_surfs[i])
               video_image_barrier(ctx, cmdbuf, zink_resource(zvc->dpb_res[ref_surfs[i]->dpb_index]),
                                   VK_IMAGE_LAYOUT_VIDEO_DECODE_DPB_KHR);
         }
         video_image_barrier(ctx, cmdbuf, zink_resource(zvc->dpb_res[surf->dpb_index]), VK_IMAGE_LAYOUT_VIDEO_DECODE_DPB_KHR);
      }
      video_image_barrier(ctx, cmdbuf, video_buffer_resource(output), VK_IMAGE_LAYOUT_VIDEO_DECODE_DST_KHR);
   }

   VkVideoBeginCodingInfoKHR bci = { 0 };
   bci.sType = VK_STRUCTURE_TYPE_VIDEO_BEGIN_CODING_INFO_KHR;
   bci.videoSession = zvc->session;
   bci.videoSessionParameters = zvc->params;
   bci.referenceSlotCount = num_refs + 1;
   bci.pReferenceSlots = ref_info;

   VKSCR(CmdBeginVideoCodingKHR)(cmdbuf, &bci);

   if (!zvc->reset_sent) {
      VkVideoCodingControlInfoKHR cc = {0};
      cc.sType = VK_STRUCTURE_TYPE_VIDEO_CODING_CONTROL_INFO_KHR;
      cc.flags = VK_VIDEO_CODING_CONTROL_RESET_BIT_KHR;
      VKSCR(CmdControlVideoCodingKHR)(cmdbuf, &cc);
      zvc->reset_sent = true;
   }

   struct zink_resource *zbs = zvc->bitstream_res;
   VkBufferMemoryBarrier2KHR bitstream_bmb;
   bitstream_bmb.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2_KHR;
   bitstream_bmb.pNext = NULL;
   bitstream_bmb.srcStageMask = VK_PIPELINE_STAGE_2_HOST_BIT_KHR;
   bitstream_bmb.srcAccessMask = VK_ACCESS_2_HOST_WRITE_BIT_KHR;
   bitstream_bmb.dstStageMask = VK_PIPELINE_STAGE_2_VIDEO_DECODE_BIT_KHR;
   bitstream_bmb.dstAccessMask = VK_ACCESS_2_VIDEO_DECODE_READ_BIT_KHR;
   bitstream_bmb.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
   bitstream_bmb.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
   bitstream_bmb.buffer = zbs->obj->buffer;
   bitstream_bmb.size = zvc->bs_range;
   bitstream_bmb.offset = zvc->bs_offset;

   VkDependencyInfoKHR di = { 0 };
   di.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO_KHR;
   di.dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;
   di.bufferMemoryBarrierCount = 1;
   di.pBufferMemoryBarriers = &bitstream_bmb;
   VKSCR(CmdPipelineBarrier2KHR)(cmdbuf, &di);

   /* The tiles, from the starts VA-API gives in superblocks. */
   uint16_t mi_col_starts[STD_VIDEO_AV1_MAX_TILE_COLS + 1];
   uint16_t mi_row_starts[STD_VIDEO_AV1_MAX_TILE_ROWS + 1];
   uint16_t width_in_sbs_minus_1[STD_VIDEO_AV1_MAX_TILE_COLS];
   uint16_t height_in_sbs_minus_1[STD_VIDEO_AV1_MAX_TILE_ROWS];
   for (unsigned i = 0; i <= av1->picture_parameter.tile_cols; i++)
      mi_col_starts[i] = av1->picture_parameter.tile_col_start_sb[i] << sb_mi_log2;
   for (unsigned i = 0; i <= av1->picture_parameter.tile_rows; i++)
      mi_row_starts[i] = av1->picture_parameter.tile_row_start_sb[i] << sb_mi_log2;
   for (unsigned i = 0; i < av1->picture_parameter.tile_cols; i++)
      width_in_sbs_minus_1[i] = av1->picture_parameter.tile_col_start_sb[i + 1] - av1->picture_parameter.tile_col_start_sb[i] - 1;
   for (unsigned i = 0; i < av1->picture_parameter.tile_rows; i++)
      height_in_sbs_minus_1[i] = av1->picture_parameter.tile_row_start_sb[i + 1] - av1->picture_parameter.tile_row_start_sb[i] - 1;

   StdVideoAV1TileInfo tile_info = { 0 };
   tile_info.flags.uniform_tile_spacing_flag = av1->picture_parameter.pic_info_fields.uniform_tile_spacing_flag;
   tile_info.TileCols = av1->picture_parameter.tile_cols;
   tile_info.TileRows = av1->picture_parameter.tile_rows;
   tile_info.context_update_tile_id = av1->picture_parameter.context_update_tile_id;
   /* VA-API gives where each tile is, not the size of the tile sizes. */
   tile_info.tile_size_bytes_minus_1 = 3;
   tile_info.pMiColStarts = mi_col_starts;
   tile_info.pMiRowStarts = mi_row_starts;
   tile_info.pWidthInSbsMinus1 = width_in_sbs_minus_1;
   tile_info.pHeightInSbsMinus1 = height_in_sbs_minus_1;

   StdVideoAV1Quantization quant = { 0 };
   quant.flags.using_qmatrix = av1->picture_parameter.qmatrix_fields.using_qmatrix;
   quant.flags.diff_uv_delta = color.flags.separate_uv_delta_q;
   quant.base_q_idx = av1->picture_parameter.base_qindex;
   quant.DeltaQYDc = av1->picture_parameter.y_dc_delta_q;
   quant.DeltaQUDc = av1->picture_parameter.u_dc_delta_q;
   quant.DeltaQUAc = av1->picture_parameter.u_ac_delta_q;
   quant.DeltaQVDc = av1->picture_parameter.v_dc_delta_q;
   quant.DeltaQVAc = av1->picture_parameter.v_ac_delta_q;
   quant.qm_y = av1->picture_parameter.qmatrix_fields.qm_y;
   quant.qm_u = av1->picture_parameter.qmatrix_fields.qm_u;
   quant.qm_v = av1->picture_parameter.qmatrix_fields.qm_v;

   StdVideoAV1Segmentation seg = { 0 };
   memcpy(seg.FeatureEnabled, av1->picture_parameter.seg_info.feature_mask, sizeof(seg.FeatureEnabled));
   memcpy(seg.FeatureData, av1->picture_parameter.seg_info.feature_data, sizeof(seg.FeatureData));

   StdVideoAV1LoopFilter lf = { 0 };
   lf.flags.loop_filter_delta_enabled = av1->picture_parameter.loop_filter_info_fields.mode_ref_delta_enabled;
   lf.flags.loop_filter_delta_update = av1->picture_parameter.loop_filter_info_fields.mode_ref_delta_update;
   lf.loop_filter_level[0] = av1->picture_parameter.filter_level[0];
   lf.loop_filter_level[1] = av1->picture_parameter.filter_level[1];
   lf.loop_filter_level[2] = av1->picture_parameter.filter_level_u;
   lf.loop_filter_level[3] = av1->picture_parameter.filter_level_v;
   lf.loop_filter_sharpness = av1->picture_parameter.loop_filter_info_fields.sharpness_level;
   memcpy(lf.loop_filter_ref_deltas, av1->picture_parameter.ref_deltas, sizeof(lf.loop_filter_ref_deltas));
   memcpy(lf.loop_filter_mode_deltas, av1->picture_parameter.mode_deltas, sizeof(lf.loop_filter_mode_deltas));

   /* VA-API packs each CDEF strength as primary << 2 | secondary. */
   StdVideoAV1CDEF cdef = { 0 };
   cdef.cdef_damping_minus_3 = av1->picture_parameter.cdef_damping_minus_3;
   cdef.cdef_bits = av1->picture_parameter.cdef_bits;
   for (unsigned i = 0; i < STD_VIDEO_AV1_MAX_CDEF_FILTER_STRENGTHS; i++) {
      cdef.cdef_y_pri_strength[i] = av1->picture_parameter.cdef_y_strengths[i] >> 2;
      cdef.cdef_y_sec_strength[i] = av1->picture_parameter.cdef_y_strengths[i] & 3;
      cdef.cdef_uv_pri_strength[i] = av1->picture_parameter.cdef_uv_strengths[i] >> 2;
      cdef.cdef_uv_sec_strength[i] = av1->picture_parameter.cdef_uv_strengths[i] & 3;
   }

   StdVideoAV1LoopRestoration lr = { 0 };
   lr.FrameRestorationType[0] = av1->picture_parameter.loop_restoration_fields.yframe_restoration_type;
   lr.FrameRestorationType[1] = av1->picture_parameter.loop_restoration_fields.cbframe_restoration_type;
   lr.FrameRestorationType[2] = av1->picture_parameter.loop_restoration_fields.crframe_restoration_type;
   /* Vulkan takes the unit sizes as log2 of the size minus 5. */
   for (unsigned i = 0; i < STD_VIDEO_AV1_MAX_NUM_PLANES; i++)
      lr.LoopRestorationSize[i] = util_logbase2(av1->picture_parameter.lr_unit_size[i]) - 5;

   /* VA-API gives the warp models of the references, from LAST_FRAME. */
   StdVideoAV1GlobalMotion gm = { 0 };
   for (unsigned i = 0; i < STD_VIDEO_AV1_REFS_PER_FRAME; i++) {
      const unsigned name = STD_VIDEO_AV1_REFERENCE_NAME_LAST_FRAME + i;
      gm.GmType[name] = av1->picture_parameter.wm[i].wmtype;
      for (unsigned j = 0; j < STD_VIDEO_AV1_GLOBAL_MOTION_PARAMS; j++)
         gm.gm_params[name][j] = av1->picture_parameter.wm[i].wmmat[j];
   }

   /* The film grain. VA-API gives the auto-regressive coefficients minus
    * 128, Vulkan plus 128, as coded.
    */
   StdVideoAV1FilmGrain fg = { 0 };
   if (decoder_grain) {
      fg.flags.chroma_scaling_from_luma =
         av1->picture_parameter.film_grain_info.film_grain_info_fields.chroma_scaling_from_luma;
      fg.flags.overlap_flag =
         av1->picture_parameter.film_grain_info.film_grain_info_fields.overlap_flag;
      fg.flags.clip_to_restricted_range =
         av1->picture_parameter.film_grain_info.film_grain_info_fields.clip_to_restricted_range;
      fg.flags.update_grain = true;
      fg.grain_scaling_minus_8 =
         av1->picture_parameter.film_grain_info.film_grain_info_fields.grain_scaling_minus_8;
      fg.ar_coeff_lag =
         av1->picture_parameter.film_grain_info.film_grain_info_fields.ar_coeff_lag;
      fg.ar_coeff_shift_minus_6 =
         av1->picture_parameter.film_grain_info.film_grain_info_fields.ar_coeff_shift_minus_6;
      fg.grain_scale_shift =
         av1->picture_parameter.film_grain_info.film_grain_info_fields.grain_scale_shift;
      fg.grain_seed = av1->picture_parameter.film_grain_info.grain_seed;
      fg.num_y_points = av1->picture_parameter.film_grain_info.num_y_points;
      memcpy(fg.point_y_value, av1->picture_parameter.film_grain_info.point_y_value,
             sizeof(fg.point_y_value));
      memcpy(fg.point_y_scaling, av1->picture_parameter.film_grain_info.point_y_scaling,
             sizeof(fg.point_y_scaling));
      fg.num_cb_points = av1->picture_parameter.film_grain_info.num_cb_points;
      memcpy(fg.point_cb_value, av1->picture_parameter.film_grain_info.point_cb_value,
             sizeof(fg.point_cb_value));
      memcpy(fg.point_cb_scaling, av1->picture_parameter.film_grain_info.point_cb_scaling,
             sizeof(fg.point_cb_scaling));
      fg.num_cr_points = av1->picture_parameter.film_grain_info.num_cr_points;
      memcpy(fg.point_cr_value, av1->picture_parameter.film_grain_info.point_cr_value,
             sizeof(fg.point_cr_value));
      memcpy(fg.point_cr_scaling, av1->picture_parameter.film_grain_info.point_cr_scaling,
             sizeof(fg.point_cr_scaling));
      for (unsigned i = 0; i < STD_VIDEO_AV1_MAX_NUM_POS_LUMA; i++)
         fg.ar_coeffs_y_plus_128[i] = av1->picture_parameter.film_grain_info.ar_coeffs_y[i] + 128;
      for (unsigned i = 0; i < STD_VIDEO_AV1_MAX_NUM_POS_CHROMA; i++) {
         fg.ar_coeffs_cb_plus_128[i] = av1->picture_parameter.film_grain_info.ar_coeffs_cb[i] + 128;
         fg.ar_coeffs_cr_plus_128[i] = av1->picture_parameter.film_grain_info.ar_coeffs_cr[i] + 128;
      }
      fg.cb_mult = av1->picture_parameter.film_grain_info.cb_mult;
      fg.cb_luma_mult = av1->picture_parameter.film_grain_info.cb_luma_mult;
      fg.cb_offset = av1->picture_parameter.film_grain_info.cb_offset;
      fg.cr_mult = av1->picture_parameter.film_grain_info.cr_mult;
      fg.cr_luma_mult = av1->picture_parameter.film_grain_info.cr_luma_mult;
      fg.cr_offset = av1->picture_parameter.film_grain_info.cr_offset;
   }

   StdVideoDecodeAV1PictureInfo pi = { 0 };
   pi.flags.error_resilient_mode = av1->picture_parameter.pic_info_fields.error_resilient_mode;
   pi.flags.disable_cdf_update = av1->picture_parameter.pic_info_fields.disable_cdf_update;
   pi.flags.use_superres = av1->picture_parameter.pic_info_fields.use_superres;
   pi.flags.allow_screen_content_tools = av1->picture_parameter.pic_info_fields.allow_screen_content_tools;
   pi.flags.is_filter_switchable = av1->picture_parameter.interp_filter == STD_VIDEO_AV1_INTERPOLATION_FILTER_SWITCHABLE;
   pi.flags.force_integer_mv = av1->picture_parameter.pic_info_fields.force_integer_mv;
   pi.flags.allow_intrabc = av1->picture_parameter.pic_info_fields.allow_intrabc;
   pi.flags.allow_high_precision_mv = av1->picture_parameter.pic_info_fields.allow_high_precision_mv;
   pi.flags.is_motion_mode_switchable = av1->picture_parameter.pic_info_fields.is_motion_mode_switchable;
   pi.flags.use_ref_frame_mvs = av1->picture_parameter.pic_info_fields.use_ref_frame_mvs;
   pi.flags.disable_frame_end_update_cdf = av1->picture_parameter.pic_info_fields.disable_frame_end_update_cdf;
   pi.flags.allow_warped_motion = av1->picture_parameter.pic_info_fields.allow_warped_motion;
   pi.flags.reduced_tx_set = av1->picture_parameter.mode_control_fields.reduced_tx_set_used;
   pi.flags.reference_select = av1->picture_parameter.mode_control_fields.reference_select;
   pi.flags.skip_mode_present = av1->picture_parameter.mode_control_fields.skip_mode_present;
   pi.flags.delta_q_present = av1->picture_parameter.mode_control_fields.delta_q_present_flag;
   pi.flags.delta_lf_present = av1->picture_parameter.mode_control_fields.delta_lf_present_flag;
   pi.flags.delta_lf_multi = av1->picture_parameter.mode_control_fields.delta_lf_multi;
   pi.flags.segmentation_enabled = av1->picture_parameter.seg_info.segment_info_fields.enabled;
   pi.flags.segmentation_update_map = av1->picture_parameter.seg_info.segment_info_fields.update_map;
   pi.flags.segmentation_temporal_update = av1->picture_parameter.seg_info.segment_info_fields.temporal_update;
   pi.flags.segmentation_update_data = av1->picture_parameter.seg_info.segment_info_fields.update_data;
   pi.flags.UsesLr = lr.FrameRestorationType[0] || lr.FrameRestorationType[1] ||
                     lr.FrameRestorationType[2];
   pi.flags.usesChromaLr = lr.FrameRestorationType[1] || lr.FrameRestorationType[2];
   pi.flags.apply_grain = decoder_grain;
   pi.frame_type = frame_type;
   pi.current_frame_id = av1->picture_parameter.current_frame_id;
   pi.OrderHint = av1->picture_parameter.order_hint;
   pi.primary_ref_frame = av1->picture_parameter.primary_ref_frame;
   pi.refresh_frame_flags = av1->picture_parameter.refresh_frame_flags;
   pi.interpolation_filter = av1->picture_parameter.interp_filter;
   pi.TxMode = av1->picture_parameter.mode_control_fields.tx_mode;
   pi.delta_q_res = av1->picture_parameter.mode_control_fields.log2_delta_q_res;
   pi.delta_lf_res = av1->picture_parameter.mode_control_fields.log2_delta_lf_res;
   if (pi.flags.skip_mode_present)
      av1_skip_mode_frames(av1, order_hints, pi.SkipModeFrame);
   pi.coded_denom = av1->picture_parameter.pic_info_fields.use_superres ?
                    av1->picture_parameter.superres_scale_denominator - 9 : 0;
   memcpy(pi.OrderHints, order_hints, sizeof(pi.OrderHints));
   pi.pTileInfo = &tile_info;
   pi.pQuantization = &quant;
   pi.pSegmentation = &seg;
   pi.pLoopFilter = &lf;
   pi.pCDEF = &cdef;
   pi.pLoopRestoration = &lr;
   pi.pGlobalMotion = &gm;
   pi.pFilmGrain = decoder_grain ? &fg : NULL;

   avpi.pStdPictureInfo = &pi;
   /* The bitstream buffer has the tile data of the frame, where VA-API says
    * each tile is.
    */
   avpi.frameHeaderOffset = 0;
   avpi.tileCount = av1->slice_parameter.slice_count;
   avpi.pTileOffsets = av1->slice_parameter.slice_data_offset;
   avpi.pTileSizes = av1->slice_parameter.slice_data_size;

   VkVideoDecodeInfoKHR vdi = { 0 };
   vdi.sType = VK_STRUCTURE_TYPE_VIDEO_DECODE_INFO_KHR;
   vdi.pNext = &avpi;
   vdi.referenceSlotCount = num_refs;
   vdi.pReferenceSlots = ref_info;
   vdi.pSetupReferenceSlot = &setup_info;
   vdi.dstPictureResource = output_res;
   vdi.srcBuffer = zbs->obj->buffer;
   vdi.srcBufferOffset = zvc->bs_offset;
   vdi.srcBufferRange = zvc->bs_range;
   VKSCR(CmdDecodeVideoKHR)(cmdbuf, &vdi);

   end_coding(ctx, zvc, cmdbuf);

   /* The picture shown gets the reconstructed one when the decoder doesn't
    * apply the film grain.
    */
   if (apply_grain && !decoder_grain)
      copy_video_buffer(ctx, av1->film_grain_target, target);

   return 0;
}

static void
zink_video_flush(struct pipe_video_codec *codec) {
}

static struct pipe_video_codec *
zink_create_video_codec(struct pipe_context *pctx, const struct pipe_video_codec *templat)
{
   struct zink_video_codec *zvc = CALLOC_STRUCT(zink_video_codec);
   if (!zvc) {
      return NULL;
   }

   zvc->base = *templat;
   zvc->base.destroy = zink_destroy_video_codec;
   zvc->base.begin_frame = zink_begin_frame;
   zvc->base.decode_macroblock = zink_decode_macroblock;
   zvc->base.decode_bitstream = zink_decode_bitstream;
   switch (u_reduce_video_profile(templat->profile)) {
   case PIPE_VIDEO_FORMAT_VP9:
      zvc->base.end_frame = zink_end_frame_vp9;
      break;
   case PIPE_VIDEO_FORMAT_AV1:
      zvc->base.end_frame = zink_end_frame_av1;
      break;
   default:
      zvc->base.end_frame = zink_end_frame_h264;
      break;
   }
   zvc->base.flush = zink_video_flush;
   zvc->base.context = pctx;

   zvc->decode_ctx = pctx->screen->context_create(pctx->screen, NULL,
                                                  ZINK_CONTEXT_VIDEO_QUEUE);
   if (!zvc->decode_ctx) {
      FREE(zvc);
      return NULL;
   }

   zvc->screen = pctx->screen;
   util_dynarray_init(&zvc->retired_views, NULL);
   /* On the decode context, which uses the buffers: buffers released while
    * still in use are kept by the batch of the context releasing them.
    */
   zvc->bitstream_mgr = u_upload_create(zvc->decode_ctx, BITSTREAM_UPLOAD_SIZE,
                                        ZINK_BIND_VIDEO, PIPE_USAGE_STAGING, 0);
   if (!zvc->bitstream_mgr) {
      zvc->decode_ctx->destroy(zvc->decode_ctx);
      FREE(zvc);
      return NULL;
   }

   return &zvc->base;
}

static struct pipe_video_buffer *
zink_video_buffer_create(struct pipe_context *pctx, const struct pipe_video_buffer *templ)
{
   struct zink_screen *screen = zink_screen(pctx->screen);
   struct pipe_video_buffer vidbuf = *templ;
   uint64_t *modifiers = NULL;
   int modifiers_count = 0;

   vidbuf.bind |= ZINK_BIND_VIDEO;

   /* All the modifiers of the format: the driver picks one the decoder can
    * write, which may be any of them.
    */
   if (pctx->screen->resource_create_with_modifiers) {
      pctx->screen->query_dmabuf_modifiers(pctx->screen, templ->buffer_format,
                                           0, NULL, NULL, &modifiers_count);
      if (modifiers_count > 0) {
         modifiers = malloc(modifiers_count * sizeof(*modifiers));
         if (!modifiers)
            return NULL;
         pctx->screen->query_dmabuf_modifiers(pctx->screen,
                                              templ->buffer_format,
                                              modifiers_count, modifiers,
                                              NULL, &modifiers_count);
      }
   }

   struct zink_video_caps_info caps_info = { 0 };
   zink_video_fill_caps(screen, PIPE_VIDEO_PROFILE_MPEG4_AVC_HIGH,
                        PIPE_VIDEO_ENTRYPOINT_BITSTREAM, false,
                        &caps_info);
   if (caps_info.dec_caps.flags & VK_VIDEO_DECODE_CAPABILITY_DPB_AND_OUTPUT_COINCIDE_BIT_KHR) {
      vidbuf.flags = ZINK_RESOURCE_FLAG_VIDEO_DPB | ZINK_RESOURCE_FLAG_VIDEO_OUTPUT;
   } else {
      vidbuf.flags = ZINK_RESOURCE_FLAG_VIDEO_OUTPUT;
   }

   struct pipe_video_buffer *buf =
      vl_video_buffer_create_as_resource(pctx, &vidbuf, modifiers,
                                         modifiers_count);
   free(modifiers);
   return buf;
}

static StdVideoH264ProfileIdc
h264_profile_conv(enum pipe_video_profile profile)
{
   switch (profile) {
   case PIPE_VIDEO_PROFILE_MPEG4_AVC_MAIN:
      return STD_VIDEO_H264_PROFILE_IDC_MAIN;
   case PIPE_VIDEO_PROFILE_MPEG4_AVC_HIGH:
      return STD_VIDEO_H264_PROFILE_IDC_HIGH;
   case PIPE_VIDEO_PROFILE_MPEG4_AVC_BASELINE:
   case PIPE_VIDEO_PROFILE_MPEG4_AVC_CONSTRAINED_BASELINE:
      return STD_VIDEO_H264_PROFILE_IDC_BASELINE;
   default:
      assert(0);
      return STD_VIDEO_H264_PROFILE_IDC_MAIN;
   }

}

static VkVideoCodecOperationFlagsKHR
convert_decode_op(enum pipe_video_format vid_format)
{
   switch (vid_format) {
   case PIPE_VIDEO_FORMAT_MPEG4_AVC:
      return VK_VIDEO_CODEC_OPERATION_DECODE_H264_BIT_KHR;
   case PIPE_VIDEO_FORMAT_VP9:
      return VK_VIDEO_CODEC_OPERATION_DECODE_VP9_BIT_KHR;
   case PIPE_VIDEO_FORMAT_AV1:
      return VK_VIDEO_CODEC_OPERATION_DECODE_AV1_BIT_KHR;
   default:
      assert(0);
      return VK_VIDEO_CODEC_OPERATION_NONE_KHR;
   }
}

bool
zink_video_fill_single_profile(struct zink_screen *screen,
                               enum pipe_video_profile profile,
                               uint32_t luma_depth,
                               struct zink_video_profile *out_prof)
{
   VkVideoComponentBitDepthFlagBitsKHR bit_depth;
   enum pipe_video_format vid_format = u_reduce_video_profile(profile);

   switch (luma_depth) {
   case 10:
      bit_depth = VK_VIDEO_COMPONENT_BIT_DEPTH_10_BIT_KHR;
      break;
   default:
      bit_depth = VK_VIDEO_COMPONENT_BIT_DEPTH_8_BIT_KHR;
      break;
   }

   if (vid_format == PIPE_VIDEO_FORMAT_MPEG4_AVC &&
       !screen->info.have_KHR_video_decode_h264)
      return false;
   if (vid_format == PIPE_VIDEO_FORMAT_VP9 &&
       !screen->info.have_KHR_video_decode_vp9)
      return false;
   if (vid_format == PIPE_VIDEO_FORMAT_AV1 &&
       !screen->info.have_KHR_video_decode_av1)
      return false;

   out_prof->profile.sType = VK_STRUCTURE_TYPE_VIDEO_PROFILE_INFO_KHR;

   out_prof->profile.videoCodecOperation = convert_decode_op(vid_format);
   out_prof->profile.chromaSubsampling = VK_VIDEO_CHROMA_SUBSAMPLING_420_BIT_KHR;
   out_prof->profile.lumaBitDepth = bit_depth;
   out_prof->profile.chromaBitDepth = bit_depth;

   switch (vid_format) {
   case PIPE_VIDEO_FORMAT_MPEG4_AVC:
      out_prof->h264.sType = VK_STRUCTURE_TYPE_VIDEO_DECODE_H264_PROFILE_INFO_KHR;
      out_prof->h264.stdProfileIdc = h264_profile_conv(profile);
      out_prof->h264.pictureLayout = VK_VIDEO_DECODE_H264_PICTURE_LAYOUT_PROGRESSIVE_KHR;
      out_prof->profile.pNext = &out_prof->h264;
      break;
   case PIPE_VIDEO_FORMAT_VP9:
      out_prof->vp9.sType = VK_STRUCTURE_TYPE_VIDEO_DECODE_VP9_PROFILE_INFO_KHR;
      out_prof->vp9.stdProfile = profile == PIPE_VIDEO_PROFILE_VP9_PROFILE2 ?
         STD_VIDEO_VP9_PROFILE_2 : STD_VIDEO_VP9_PROFILE_0;
      out_prof->profile.pNext = &out_prof->vp9;
      break;
   case PIPE_VIDEO_FORMAT_AV1:
      out_prof->av1.sType = VK_STRUCTURE_TYPE_VIDEO_DECODE_AV1_PROFILE_INFO_KHR;
      out_prof->av1.stdProfile = STD_VIDEO_AV1_PROFILE_MAIN;
      out_prof->av1.filmGrainSupport = av1_film_grain(screen);
      out_prof->profile.pNext = &out_prof->av1;
      break;
   default:
      return false;
   }
   return true;
}

void
zink_video_fill_profiles(struct zink_screen *screen,
                         struct zink_video_profile_info *profiles,
                         enum pipe_video_profile profile,
                         uint32_t luma_depth)
{
   int profile_count = 0;
   enum pipe_video_format vid_format = u_reduce_video_profile(profile);

   if (screen->info.have_KHR_video_decode_h264 && (vid_format == PIPE_VIDEO_FORMAT_MPEG4_AVC ||
                                                   vid_format == PIPE_VIDEO_FORMAT_UNKNOWN)) {
      if (vid_format == PIPE_VIDEO_FORMAT_UNKNOWN)
         profile = PIPE_VIDEO_PROFILE_MPEG4_AVC_HIGH;
      if (zink_video_fill_single_profile(screen, profile, luma_depth, &profiles->h264)) {
         profiles->profiles[profile_count] = profiles->h264.profile;
         profile_count++;
      }
   }

   if (screen->info.have_KHR_video_decode_vp9 && (vid_format == PIPE_VIDEO_FORMAT_VP9 ||
                                                  vid_format == PIPE_VIDEO_FORMAT_UNKNOWN)) {
      enum pipe_video_profile vp9_profile = profile;
      if (vid_format == PIPE_VIDEO_FORMAT_UNKNOWN)
         vp9_profile = luma_depth > 8 ? PIPE_VIDEO_PROFILE_VP9_PROFILE2 :
                                        PIPE_VIDEO_PROFILE_VP9_PROFILE0;
      if (zink_video_fill_single_profile(screen, vp9_profile, luma_depth, &profiles->vp9)) {
         profiles->profiles[profile_count] = profiles->vp9.profile;
         profile_count++;
      }
   }

   if (screen->info.have_KHR_video_decode_av1 && (vid_format == PIPE_VIDEO_FORMAT_AV1 ||
                                                  vid_format == PIPE_VIDEO_FORMAT_UNKNOWN)) {
      enum pipe_video_profile av1_profile = profile;
      if (vid_format == PIPE_VIDEO_FORMAT_UNKNOWN)
         av1_profile = PIPE_VIDEO_PROFILE_AV1_MAIN;
      if (zink_video_fill_single_profile(screen, av1_profile, luma_depth, &profiles->av1)) {
         profiles->profiles[profile_count] = profiles->av1.profile;
         profile_count++;
      }
   }

   profiles->list.sType = VK_STRUCTURE_TYPE_VIDEO_PROFILE_LIST_INFO_KHR;
   profiles->list.profileCount = profile_count;
   profiles->list.pProfiles = profiles->profiles;
}

void
zink_video_init(struct zink_context *ctx)
{
   ctx->base.create_video_codec = zink_create_video_codec;
   ctx->base.create_video_buffer = zink_video_buffer_create;
}

bool
zink_video_fill_caps(struct zink_screen *screen,
                     enum pipe_video_profile profile,
                     enum pipe_video_entrypoint entrypoint,
                     bool bit_depth_10,
                     struct zink_video_caps_info *caps_info)
{
   struct zink_video_profile vk_profile = { 0 };
   if (!zink_video_fill_single_profile(screen, profile, bit_depth_10 ? 10 : 8, &vk_profile))
      return false;

   caps_info->caps.sType = VK_STRUCTURE_TYPE_VIDEO_CAPABILITIES_KHR;
   caps_info->caps.pNext = &caps_info->dec_caps;
   caps_info->dec_caps.sType = VK_STRUCTURE_TYPE_VIDEO_DECODE_CAPABILITIES_KHR;

   switch (u_reduce_video_profile(profile)) {
   case PIPE_VIDEO_FORMAT_VP9:
      caps_info->dec_caps.pNext = &caps_info->vp9_dec_caps;
      caps_info->vp9_dec_caps.sType = VK_STRUCTURE_TYPE_VIDEO_DECODE_VP9_CAPABILITIES_KHR;
      break;
   case PIPE_VIDEO_FORMAT_AV1:
      caps_info->dec_caps.pNext = &caps_info->av1_dec_caps;
      caps_info->av1_dec_caps.sType = VK_STRUCTURE_TYPE_VIDEO_DECODE_AV1_CAPABILITIES_KHR;
      break;
   default:
      caps_info->dec_caps.pNext = &caps_info->h264_dec_caps;
      caps_info->h264_dec_caps.sType = VK_STRUCTURE_TYPE_VIDEO_DECODE_H264_CAPABILITIES_KHR;
      break;
   }

   return VKSCR(GetPhysicalDeviceVideoCapabilitiesKHR)(screen->pdev, &vk_profile.profile, &caps_info->caps) == VK_SUCCESS;
}

VkResult
zink_fill_video_format_props(struct zink_screen *screen,
                             VkImageUsageFlags usage,
                             enum pipe_video_profile profile,
                             uint32_t bit_depth,
                             struct zink_video_format_prop *props)
{
   struct zink_video_profile_info profiles = {0};
   zink_video_fill_profiles(screen, &profiles, profile, bit_depth);

   VkPhysicalDeviceVideoFormatInfoKHR video_format_info = {};
   video_format_info.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VIDEO_FORMAT_INFO_KHR;
   video_format_info.pNext = &profiles.list;
   video_format_info.imageUsage = usage;

   props->pVideoFormatProperties = NULL;
   VkResult ret = VKSCR(GetPhysicalDeviceVideoFormatPropertiesKHR)(screen->pdev, &video_format_info,
                                                                   &props->videoFormatPropertyCount,
                                                                   NULL);
   if (ret != VK_SUCCESS)
      return ret;
   props->pVideoFormatProperties = calloc(props->videoFormatPropertyCount, sizeof(VkVideoFormatPropertiesKHR));
   for (unsigned i = 0; i < props->videoFormatPropertyCount; i++)
      props->pVideoFormatProperties[i].sType = VK_STRUCTURE_TYPE_VIDEO_FORMAT_PROPERTIES_KHR;
   VKSCR(GetPhysicalDeviceVideoFormatPropertiesKHR)(screen->pdev, &video_format_info,
                                                    &props->videoFormatPropertyCount,
                                                    props->pVideoFormatProperties);
   return VK_SUCCESS;
}

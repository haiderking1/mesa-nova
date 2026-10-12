/*
 * Copyright © 2023 Collabora, Ltd. and Red Hat Inc.
 * SPDX-License-Identifier: MIT
 */
#include "nvk_video_session.h"

#include "vk_alloc.h"
#include "nvk_device.h"
#include "nvk_cmd_buffer.h"
#include "nvk_physical_device.h"
#include "nvk_rust.h"
#include "nvk_entrypoints.h"


VKAPI_ATTR VkResult VKAPI_CALL
nvk_CreateVideoSessionKHR(VkDevice _device,
                          const VkVideoSessionCreateInfoKHR *pCreateInfo,
                          const VkAllocationCallbacks *pAllocator,
                          VkVideoSessionKHR *pVideoSession)
{
   VK_FROM_HANDLE(nvk_device, dev, _device);

   struct nvk_video_session *vid =
      vk_zalloc2(&dev->vk.alloc, pAllocator, sizeof(*vid), 8,
                 VK_SYSTEM_ALLOCATION_SCOPE_OBJECT);
   if (!vid)
      return vk_error(dev, VK_ERROR_OUT_OF_HOST_MEMORY);

   VkResult result = vk_video_session_init(&dev->vk, &vid->vk, pCreateInfo);
   if (result != VK_SUCCESS) {
      vk_free2(&dev->vk.alloc, pAllocator, vid);
      return result;
   }

   assert(util_bitcount(vid->vk.op) == 1);
   switch (vid->vk.op) {
   case VK_VIDEO_CODEC_OPERATION_DECODE_H264_BIT_KHR: {
      uint64_t max_width_in_mb = vid->vk.max_coded.width / 16;
      uint64_t max_height_in_mb = vid->vk.max_coded.height / 16;
      uint64_t coloc_size = align(align(max_height_in_mb, 2) * (max_width_in_mb * 64) - 63, 0x100);
      coloc_size *= vid->vk.max_active_ref_pics + 1; /* Max number of references frames, plus current frame */
      uint64_t mbhist_size  = align(max_width_in_mb * 104, 0x100);
      uint64_t history_size = align(max_width_in_mb * 0x300, 0x200);

      vid->mems[0] = (struct nvk_vid_mem) {
         .size_B = coloc_size,
         .align_B = 256,
      };
      vid->mems[1] = (struct nvk_vid_mem) {
         .size_B = mbhist_size,
         .align_B = 256,
      };
      vid->mems[2] = (struct nvk_vid_mem) {
         .size_B = history_size,
         .align_B = 256,
      };
      break;
   }

   case VK_VIDEO_CODEC_OPERATION_DECODE_VP9_BIT_KHR: {
      /* Sizes in 64x64 superblocks, as the decoder needs for the largest
       * picture, matching what NVIDIA's driver asks for.
       */
      uint64_t sb_cols = DIV_ROUND_UP(vid->vk.max_coded.width, 64);
      uint64_t sb_rows = DIV_ROUND_UP(vid->vk.max_coded.height, 64);
      uint64_t sbs = sb_cols * sb_rows;
      /* Tile columns are at least 4 superblocks wide. */
      uint64_t tile_cols = DIV_ROUND_UP(sb_cols, 4);

      /* Two of each, the decoder writing one as it reads the other. */
      const uint64_t seg_map_B =
         align64(NVK_VP9_SEG_MAP_SB_B * sbs, NVK_VP9_BUF_ALIGN_B);
      const uint64_t col_mv_B =
         align64(NVK_VP9_COL_MV_SB_B * sbs, NVK_VP9_BUF_ALIGN_B);
      vid->mems[NVK_VP9_MEM_SEG_MV] = (struct nvk_vid_mem) {
         .size_B = align64(2 * seg_map_B + 2 * col_mv_B, 4096),
         .align_B = 4096,
      };
      vid->mems[NVK_VP9_MEM_FILTER] = (struct nvk_vid_mem) {
         .size_B = align64(NVK_VP9_INTRA_TOP_B + 5776 * sb_cols +
                           3328 * MAX2(tile_cols - 1, 1) * sb_rows, 4096),
         .align_B = 4096,
      };
      vid->mems[NVK_VP9_MEM_PROBS] = (struct nvk_vid_mem) {
         .size_B = NVK_VP9_PROBS_MEM_B,
         .align_B = 4096,
      };
      break;
   }

   case VK_VIDEO_CODEC_OPERATION_DECODE_AV1_BIT_KHR: {
      const struct nvk_video_av1_mem_layout layout =
         nvk_video_av1_mem_layout(vid->vk.max_coded.width,
                                  vid->vk.max_coded.height);
      vid->mems[NVK_AV1_MEM_MV_SEG] = (struct nvk_vid_mem) {
         .size_B = layout.mv_seg_mem_B,
         .align_B = 4096,
      };
      vid->mems[NVK_AV1_MEM_FILTER] = (struct nvk_vid_mem) {
         .size_B = layout.filter_mem_B,
         .align_B = 4096,
      };
      vid->mems[NVK_AV1_MEM_PROBS] = (struct nvk_vid_mem) {
         .size_B = NVK_AV1_PROBS_MEM_B,
         .align_B = 4096,
      };
      break;
   }

   default:
      vk_video_session_finish(&vid->vk);
      vk_free2(&dev->vk.alloc, pAllocator, vid);
      return vk_error(dev, VK_ERROR_VIDEO_PROFILE_OPERATION_NOT_SUPPORTED_KHR);
   }

   nvk_video_create_video_session(vid);

   *pVideoSession = nvk_video_session_to_handle(vid);

   return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL
nvk_DestroyVideoSessionKHR(VkDevice _device, VkVideoSessionKHR videoSession,
                           const VkAllocationCallbacks *pAllocator)
{
   VK_FROM_HANDLE(nvk_device, dev, _device);
   VK_FROM_HANDLE(nvk_video_session, vid, videoSession);

   if (vid == NULL)
      return;

   nvk_video_destroy_video_session(vid);

   vk_video_session_finish(&vid->vk);
   vk_free2(&dev->vk.alloc, pAllocator, vid);
}

VKAPI_ATTR VkResult VKAPI_CALL
nvk_GetVideoSessionMemoryRequirementsKHR(VkDevice _device,
   VkVideoSessionKHR videoSession,
   uint32_t *pMemoryRequirementsCount,
   VkVideoSessionMemoryRequirementsKHR *pMemoryRequirements)
{
   VK_FROM_HANDLE(nvk_device, dev, _device);
   VK_FROM_HANDLE(nvk_video_session, vid, videoSession);
   const struct nvk_physical_device *pdev = nvk_device_physical(dev);
   const uint32_t memory_type_bits = BITFIELD_MASK(pdev->mem_type_count);

   VK_OUTARRAY_MAKE_TYPED(VkVideoSessionMemoryRequirementsKHR, out,
                          pMemoryRequirements, pMemoryRequirementsCount);

   for (unsigned i = 0; i < ARRAY_SIZE(vid->mems); i++) {
      if (vid->mems[i].size_B == 0)
         continue;

      vk_outarray_append_typed(VkVideoSessionMemoryRequirementsKHR, &out, m) {
         m->memoryBindIndex = i;
         m->memoryRequirements.size = vid->mems[i].size_B;
         m->memoryRequirements.alignment = vid->mems[i].align_B;
         m->memoryRequirements.memoryTypeBits = memory_type_bits;
      }
   }

   return vk_outarray_status(&out);
}

/* Gives the probability tables of a VP9 session their default probabilities,
 * and clears the symbol counters and the decoder's state of the previous
 * picture, through the upload queue, which the decodes submitted later wait
 * for. The decoder adapts the 4 frame contexts from then on, and resets them
 * from the defaults on key frames and as frames ask.
 */
static VkResult
nvk_video_vp9_init_probs_mem(struct nvk_device *dev,
                             const struct nvk_video_session *vid)
{
   uint8_t *data = vk_zalloc(&dev->vk.alloc, NVK_VP9_PROBS_MEM_B, 8,
                             VK_SYSTEM_ALLOCATION_SCOPE_COMMAND);
   if (data == NULL)
      return vk_error(dev, VK_ERROR_OUT_OF_HOST_MEMORY);

   for (unsigned i = 0; i <= NVK_VP9_DEFAULT_PROBS; i++)
      nvk_video_vp9_default_probs(data + i * NVK_VP9_PROBS_B, NVK_VP9_PROBS_B);

   VkResult result = nvk_upload_queue_upload(dev, &dev->upload,
                                             vid->mems[NVK_VP9_MEM_PROBS].addr,
                                             data, NVK_VP9_PROBS_MEM_B);
   vk_free(&dev->vk.alloc, data);

   return result;
}

/* Gives an AV1 session its default probability tables, and its segment id
 * map of zeros, through the upload queue, which the decodes submitted later
 * wait for.
 */
static VkResult
nvk_video_av1_init_mem(struct nvk_device *dev,
                       const struct nvk_video_session *vid,
                       uint32_t bind_index)
{
   switch (bind_index) {
   case NVK_AV1_MEM_PROBS: {
      const uint32_t size_B = NVK_AV1_DEFAULT_PROBS * NVK_AV1_PROBS_B;
      uint8_t *data = vk_zalloc(&dev->vk.alloc, size_B, 8,
                                VK_SYSTEM_ALLOCATION_SCOPE_COMMAND);
      if (data == NULL)
         return vk_error(dev, VK_ERROR_OUT_OF_HOST_MEMORY);

      for (unsigned i = 0; i < NVK_AV1_DEFAULT_PROBS; i++) {
         nvk_video_av1_default_probs(i, data + i * NVK_AV1_PROBS_B,
                                     NVK_AV1_PROBS_B);
      }

      VkResult result =
         nvk_upload_queue_upload(dev, &dev->upload,
                                 vid->mems[NVK_AV1_MEM_PROBS].addr,
                                 data, size_B);
      vk_free(&dev->vk.alloc, data);
      return result;
   }

   case NVK_AV1_MEM_MV_SEG: {
      const struct nvk_video_av1_mem_layout layout =
         nvk_video_av1_mem_layout(vid->vk.max_coded.width,
                                  vid->vk.max_coded.height);
      return nvk_upload_queue_fill(dev, &dev->upload,
                                   vid->mems[NVK_AV1_MEM_MV_SEG].addr +
                                   layout.zero_seg_offset_B,
                                   0, layout.seg_B);
   }

   default:
      return VK_SUCCESS;
   }
}

VKAPI_ATTR VkResult VKAPI_CALL
nvk_BindVideoSessionMemoryKHR(VkDevice _device,
   VkVideoSessionKHR videoSession,
   uint32_t videoSessionBindMemoryCount,
   const VkBindVideoSessionMemoryInfoKHR *pBindSessionMemoryInfos)
{
   VK_FROM_HANDLE(nvk_device, dev, _device);
   VK_FROM_HANDLE(nvk_video_session, vid, videoSession);

   for (unsigned i = 0; i < videoSessionBindMemoryCount; i++) {
      const VkBindVideoSessionMemoryInfoKHR *info = &pBindSessionMemoryInfos[i];
      VK_FROM_HANDLE(nvk_device_memory, mem, info->memory);

      vid->mems[info->memoryBindIndex].addr =
         mem->mem->va->addr + info->memoryOffset;

      if (vid->vk.op == VK_VIDEO_CODEC_OPERATION_DECODE_VP9_BIT_KHR &&
          info->memoryBindIndex == NVK_VP9_MEM_PROBS) {
         VkResult result = nvk_video_vp9_init_probs_mem(dev, vid);
         if (result != VK_SUCCESS)
            return result;
      }

      if (vid->vk.op == VK_VIDEO_CODEC_OPERATION_DECODE_AV1_BIT_KHR) {
         VkResult result =
            nvk_video_av1_init_mem(dev, vid, info->memoryBindIndex);
         if (result != VK_SUCCESS)
            return result;
      }
   }
   return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL
nvk_GetPhysicalDeviceVideoCapabilitiesKHR(VkPhysicalDevice physicalDevice,
                                          const VkVideoProfileInfoKHR *pVideoProfile,
                                          VkVideoCapabilitiesKHR *pCapabilities)
{
   pCapabilities->flags = 0;
   pCapabilities->minBitstreamBufferOffsetAlignment = 256;
   pCapabilities->minBitstreamBufferSizeAlignment = 256;

   struct VkVideoDecodeCapabilitiesKHR *dec_caps =
      vk_find_struct(pCapabilities->pNext, VIDEO_DECODE_CAPABILITIES_KHR);
   if (dec_caps)
      dec_caps->flags = VK_VIDEO_DECODE_CAPABILITY_DPB_AND_OUTPUT_COINCIDE_BIT_KHR;

   /* H264 allows different luma and chroma bit depths */
   if (pVideoProfile->lumaBitDepth != pVideoProfile->chromaBitDepth)
      return VK_ERROR_VIDEO_PROFILE_FORMAT_NOT_SUPPORTED_KHR;

   if (pVideoProfile->chromaSubsampling != VK_VIDEO_CHROMA_SUBSAMPLING_420_BIT_KHR)
      return VK_ERROR_VIDEO_PROFILE_FORMAT_NOT_SUPPORTED_KHR;

   switch (pVideoProfile->videoCodecOperation) {
   case VK_VIDEO_CODEC_OPERATION_DECODE_H264_BIT_KHR: {
      const struct VkVideoDecodeH264ProfileInfoKHR *h264_profile =
         vk_find_struct_const(pVideoProfile->pNext,
                              VIDEO_DECODE_H264_PROFILE_INFO_KHR);
      struct VkVideoDecodeH264CapabilitiesKHR *ext =
         vk_find_struct(pCapabilities->pNext,
                        VIDEO_DECODE_H264_CAPABILITIES_KHR);

      if (h264_profile->stdProfileIdc != STD_VIDEO_H264_PROFILE_IDC_BASELINE &&
          h264_profile->stdProfileIdc != STD_VIDEO_H264_PROFILE_IDC_MAIN &&
          h264_profile->stdProfileIdc != STD_VIDEO_H264_PROFILE_IDC_HIGH)
         return VK_ERROR_VIDEO_PROFILE_OPERATION_NOT_SUPPORTED_KHR;

      if (pVideoProfile->lumaBitDepth != VK_VIDEO_COMPONENT_BIT_DEPTH_8_BIT_KHR)
         return VK_ERROR_VIDEO_PROFILE_FORMAT_NOT_SUPPORTED_KHR;

      pCapabilities->pictureAccessGranularity.width = VK_VIDEO_H264_MACROBLOCK_WIDTH;
      pCapabilities->pictureAccessGranularity.height = VK_VIDEO_H264_MACROBLOCK_HEIGHT;
      pCapabilities->minCodedExtent.width = 48;
      pCapabilities->minCodedExtent.height = VK_VIDEO_H264_MACROBLOCK_HEIGHT;
      pCapabilities->maxCodedExtent.width = 4096;
      pCapabilities->maxCodedExtent.height = 4096;
      pCapabilities->flags = VK_VIDEO_CAPABILITY_SEPARATE_REFERENCE_IMAGES_BIT_KHR;
      pCapabilities->maxDpbSlots = 17;
      pCapabilities->maxActiveReferencePictures = 16;
      ext->fieldOffsetGranularity.x = 0;
      ext->fieldOffsetGranularity.y = 0;
      ext->maxLevelIdc = STD_VIDEO_H264_LEVEL_IDC_5_2;
      strcpy(pCapabilities->stdHeaderVersion.extensionName,
             VK_STD_VULKAN_VIDEO_CODEC_H264_DECODE_EXTENSION_NAME);
      pCapabilities->stdHeaderVersion.specVersion =
         VK_STD_VULKAN_VIDEO_CODEC_H264_DECODE_SPEC_VERSION;
      break;
   }

   case VK_VIDEO_CODEC_OPERATION_DECODE_VP9_BIT_KHR: {
      VK_FROM_HANDLE(nvk_physical_device, pdev, physicalDevice);
      const struct VkVideoDecodeVP9ProfileInfoKHR *vp9_profile =
         vk_find_struct_const(pVideoProfile->pNext,
                              VIDEO_DECODE_VP9_PROFILE_INFO_KHR);
      struct VkVideoDecodeVP9CapabilitiesKHR *ext =
         vk_find_struct(pCapabilities->pNext,
                        VIDEO_DECODE_VP9_CAPABILITIES_KHR);

      if (!pdev->vk.supported_extensions.KHR_video_decode_vp9)
         return VK_ERROR_VIDEO_PROFILE_CODEC_NOT_SUPPORTED_KHR;

      /* Profile 0: 8-bit 4:2:0. */
      if (vp9_profile->stdProfile != STD_VIDEO_VP9_PROFILE_0)
         return VK_ERROR_VIDEO_PROFILE_OPERATION_NOT_SUPPORTED_KHR;

      if (pVideoProfile->lumaBitDepth != VK_VIDEO_COMPONENT_BIT_DEPTH_8_BIT_KHR)
         return VK_ERROR_VIDEO_PROFILE_FORMAT_NOT_SUPPORTED_KHR;

      /* The limits NVIDIA's driver reports for NVDEC on Ampere. */
      pCapabilities->pictureAccessGranularity.width = 16;
      pCapabilities->pictureAccessGranularity.height = 16;
      pCapabilities->minCodedExtent.width = 128;
      pCapabilities->minCodedExtent.height = 128;
      pCapabilities->maxCodedExtent.width = 8192;
      pCapabilities->maxCodedExtent.height = 8192;
      pCapabilities->flags = VK_VIDEO_CAPABILITY_SEPARATE_REFERENCE_IMAGES_BIT_KHR;
      /* The 8 reference frames, and the picture decoded, which is also its
       * own reconstructed picture.
       */
      pCapabilities->maxDpbSlots = STD_VIDEO_VP9_NUM_REF_FRAMES + 1;
      pCapabilities->maxActiveReferencePictures =
         VK_MAX_VIDEO_VP9_REFERENCES_PER_FRAME_KHR;
      ext->maxLevel = STD_VIDEO_VP9_LEVEL_6_2;
      strcpy(pCapabilities->stdHeaderVersion.extensionName,
             VK_STD_VULKAN_VIDEO_CODEC_VP9_DECODE_EXTENSION_NAME);
      pCapabilities->stdHeaderVersion.specVersion =
         VK_STD_VULKAN_VIDEO_CODEC_VP9_DECODE_SPEC_VERSION;
      break;
   }

   case VK_VIDEO_CODEC_OPERATION_DECODE_AV1_BIT_KHR: {
      VK_FROM_HANDLE(nvk_physical_device, pdev, physicalDevice);
      const struct VkVideoDecodeAV1ProfileInfoKHR *av1_profile =
         vk_find_struct_const(pVideoProfile->pNext,
                              VIDEO_DECODE_AV1_PROFILE_INFO_KHR);
      struct VkVideoDecodeAV1CapabilitiesKHR *ext =
         vk_find_struct(pCapabilities->pNext,
                        VIDEO_DECODE_AV1_CAPABILITIES_KHR);

      if (!pdev->vk.supported_extensions.KHR_video_decode_av1)
         return VK_ERROR_VIDEO_PROFILE_CODEC_NOT_SUPPORTED_KHR;

      /* Main profile, 8-bit 4:2:0. */
      if (av1_profile->stdProfile != STD_VIDEO_AV1_PROFILE_MAIN)
         return VK_ERROR_VIDEO_PROFILE_OPERATION_NOT_SUPPORTED_KHR;

      /* The decoder outputs the frames with their film grain to another
       * picture than the reconstructed one, which it writes too.
       */
      if (av1_profile->filmGrainSupport && dec_caps)
         dec_caps->flags |= VK_VIDEO_DECODE_CAPABILITY_DPB_AND_OUTPUT_DISTINCT_BIT_KHR;

      if (pVideoProfile->lumaBitDepth != VK_VIDEO_COMPONENT_BIT_DEPTH_8_BIT_KHR)
         return VK_ERROR_VIDEO_PROFILE_FORMAT_NOT_SUPPORTED_KHR;

      /* The limits NVIDIA's driver reports for NVDEC on Ampere. */
      pCapabilities->pictureAccessGranularity.width = 16;
      pCapabilities->pictureAccessGranularity.height = 16;
      pCapabilities->minCodedExtent.width = 128;
      pCapabilities->minCodedExtent.height = 128;
      pCapabilities->maxCodedExtent.width = 8192;
      pCapabilities->maxCodedExtent.height = 8192;
      pCapabilities->flags = VK_VIDEO_CAPABILITY_SEPARATE_REFERENCE_IMAGES_BIT_KHR;
      /* The 8 reference frames, and the picture decoded, which is also its
       * own reconstructed picture.
       */
      pCapabilities->maxDpbSlots = STD_VIDEO_AV1_NUM_REF_FRAMES + 1;
      pCapabilities->maxActiveReferencePictures =
         STD_VIDEO_AV1_REFS_PER_FRAME;
      ext->maxLevel = STD_VIDEO_AV1_LEVEL_6_0;
      strcpy(pCapabilities->stdHeaderVersion.extensionName,
             VK_STD_VULKAN_VIDEO_CODEC_AV1_DECODE_EXTENSION_NAME);
      pCapabilities->stdHeaderVersion.specVersion =
         VK_STD_VULKAN_VIDEO_CODEC_AV1_DECODE_SPEC_VERSION;
      break;
   }

   default:
      return VK_ERROR_VIDEO_PROFILE_OPERATION_NOT_SUPPORTED_KHR;
   }

   return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL
nvk_GetPhysicalDeviceVideoFormatPropertiesKHR(VkPhysicalDevice physicalDevice,
   const VkPhysicalDeviceVideoFormatInfoKHR *pVideoFormatInfo,
   uint32_t *pVideoFormatPropertyCount,
   VkVideoFormatPropertiesKHR *pVideoFormatProperties)
{
   VK_OUTARRAY_MAKE_TYPED(VkVideoFormatPropertiesKHR, out,
                          pVideoFormatProperties, pVideoFormatPropertyCount);

   const VkImageUsageFlags decode_usage =
      VK_IMAGE_USAGE_VIDEO_DECODE_DST_BIT_KHR |
      VK_IMAGE_USAGE_VIDEO_DECODE_DPB_BIT_KHR;

   const VkImageUsageFlags video_usage = pVideoFormatInfo->imageUsage &
      (decode_usage |
       VK_IMAGE_USAGE_VIDEO_DECODE_SRC_BIT_KHR |
       VK_IMAGE_USAGE_VIDEO_ENCODE_SRC_BIT_KHR |
       VK_IMAGE_USAGE_VIDEO_ENCODE_DST_BIT_KHR |
       VK_IMAGE_USAGE_VIDEO_ENCODE_DPB_BIT_KHR);

   /* imageUsage names the usages the returned formats have to support, so a
    * usage we cannot serve has to fail the query: there is no encode support,
    * and decode sources are buffers, not images.
    */
   if (video_usage & ~decode_usage)
      return VK_ERROR_IMAGE_USAGE_NOT_SUPPORTED_KHR;

   if (video_usage == 0)
      return VK_ERROR_IMAGE_USAGE_NOT_SUPPORTED_KHR;

   /* Optimal images, and images of DRM format modifiers, for the decoded
    * pictures to be shared: those of the block linear modifier the decoder
    * writes (see nil_drm_format_mod_supports_video()).
    */
   const VkImageTiling tilings[] = {
      VK_IMAGE_TILING_OPTIMAL,
      VK_IMAGE_TILING_DRM_FORMAT_MODIFIER_EXT,
   };
   for (unsigned i = 0; i < ARRAY_SIZE(tilings); i++) {
      vk_outarray_append_typed(VkVideoFormatPropertiesKHR, &out, p) {
         p->format = VK_FORMAT_G8_B8R8_2PLANE_420_UNORM;
         p->imageType = VK_IMAGE_TYPE_2D;
         p->imageTiling = tilings[i];
         /* Report the create flags video images actually support. */
         p->imageCreateFlags = VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT |
                               VK_IMAGE_CREATE_EXTENDED_USAGE_BIT |
                               VK_IMAGE_CREATE_ALIAS_BIT;
         /* Only VK_VIDEO_DECODE_CAPABILITY_DPB_AND_OUTPUT_COINCIDE_BIT_KHR
          * is advertised, so DPB and decode output usage always come
          * together on the same image.
          */
         p->imageUsageFlags = pVideoFormatInfo->imageUsage | decode_usage |
                              VK_IMAGE_USAGE_SAMPLED_BIT |
                              VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                              VK_IMAGE_USAGE_TRANSFER_DST_BIT;
      }
   }

   return vk_outarray_status(&out);
}

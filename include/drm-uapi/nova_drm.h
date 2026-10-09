/* SPDX-License-Identifier: MIT */

#ifndef __NOVA_DRM_H__
#define __NOVA_DRM_H__

#include "drm.h"

/* DISCLAIMER: Do not use, this is not a stable uAPI.
 *
 * This uAPI serves only testing purposes as long as this driver is still in
 * development. It is required to implement and test infrastructure which is
 * upstreamed in the context of this driver. See also [1].
 *
 * [1] https://lore.kernel.org/dri-devel/Zfsj0_tb-0-tNrJy@cassiopeiae/T/#u
 */

#if defined(__cplusplus)
extern "C" {
#endif

/**
 * enum drm_nova_architecture - GPU architecture identifier
 */
enum drm_nova_architecture {
	NOVA_DRM_ARCHITECTURE_TURING		= 0x16,
	NOVA_DRM_ARCHITECTURE_AMPERE		= 0x17,
	NOVA_DRM_ARCHITECTURE_HOPPER		= 0x18,
	NOVA_DRM_ARCHITECTURE_ADA		= 0x19,
	NOVA_DRM_ARCHITECTURE_BLACKWELL_GB10X	= 0x1a,
	NOVA_DRM_ARCHITECTURE_BLACKWELL_GB20X	= 0x1b,
};

/**
 * enum drm_nova_chipid - opaque GPU chip identifier
 *
 * These values identify the chip a GPU is based on. They are otherwise
 * opaque: userspace must not assume the values carry any particular meaning
 * or encoding, only that they may be compared against this enum.
 */
enum drm_nova_chipid {
	/* Turing */
	NOVA_DRM_CHIPID_TU102			= 0x162,
	NOVA_DRM_CHIPID_TU104			= 0x164,
	NOVA_DRM_CHIPID_TU106			= 0x166,
	NOVA_DRM_CHIPID_TU117			= 0x167,
	NOVA_DRM_CHIPID_TU116			= 0x168,
	/* Ampere */
	NOVA_DRM_CHIPID_GA100			= 0x170,
	NOVA_DRM_CHIPID_GA102			= 0x172,
	NOVA_DRM_CHIPID_GA103			= 0x173,
	NOVA_DRM_CHIPID_GA104			= 0x174,
	NOVA_DRM_CHIPID_GA106			= 0x176,
	NOVA_DRM_CHIPID_GA107			= 0x177,
	/* Hopper */
	NOVA_DRM_CHIPID_GH100			= 0x180,
	/* Ada */
	NOVA_DRM_CHIPID_AD102			= 0x192,
	NOVA_DRM_CHIPID_AD103			= 0x193,
	NOVA_DRM_CHIPID_AD104			= 0x194,
	NOVA_DRM_CHIPID_AD106			= 0x196,
	NOVA_DRM_CHIPID_AD107			= 0x197,
	/* Blackwell GB10x */
	NOVA_DRM_CHIPID_GB100			= 0x1a0,
	NOVA_DRM_CHIPID_GB102			= 0x1a2,
	/* Blackwell GB20x */
	NOVA_DRM_CHIPID_GB202			= 0x1b2,
	NOVA_DRM_CHIPID_GB203			= 0x1b3,
	NOVA_DRM_CHIPID_GB205			= 0x1b5,
	NOVA_DRM_CHIPID_GB206			= 0x1b6,
	NOVA_DRM_CHIPID_GB207			= 0x1b7,
};

/**
 * DRM_NOVA_GEM_VRAM - the memory of the object is in VRAM
 *
 * Otherwise, it is in system memory, which the GPU accesses coherently with
 * the CPU's caches, and which the CPU maps cached.
 */
#define DRM_NOVA_GEM_VRAM		(1 << 0)

/**
 * DRM_NOVA_GEM_CPU_VISIBLE - the CPU can map the object
 *
 * The CPU maps VRAM objects write-combined, through the GPU's VRAM BAR. System
 * memory objects are always visible to the CPU.
 */
#define DRM_NOVA_GEM_CPU_VISIBLE	(1 << 1)

/**
 * struct drm_nova_gem_create - create a new DRM GEM object
 *
 * The memory of the object is zeroed.
 */
struct drm_nova_gem_create {
	/**
	 * @handle: The handle of the new DRM GEM object.
	 */
	__u32 handle;

	/**
	 * @flags: Where the memory of the object is and how it is accessed:
	 * DRM_NOVA_GEM_* flags. 0 is system memory.
	 */
	__u32 flags;

	/**
	 * @size: The size of the new DRM GEM object. It is rounded up to the
	 * page size, or to 64 KiB, the size of the GPU's large pages, in VRAM.
	 * &struct drm_nova_gem_info returns the rounded size.
	 */
	__u64 size;
};

/**
 * struct drm_nova_gem_info - query DRM GEM object metadata
 */
struct drm_nova_gem_info {
	/**
	 * @handle: The handle of the DRM GEM object to query.
	 */
	__u32 handle;

	/**
	 * @flags: Where the memory of the object is and how it is accessed:
	 * DRM_NOVA_GEM_* flags.
	 */
	__u32 flags;

	/**
	 * @size: The size of the DRM GEM obejct.
	 */
	__u64 size;

	/**
	 * @mmap_offset: The offset to pass to mmap() on the DRM file to map
	 * the object, 0 if the CPU can't map it.
	 */
	__u64 mmap_offset;
};

/**
 * struct drm_nova_info - query device information
 */
struct drm_nova_info {
	/**
	 * @id: The identifier of the information to query.
	 */
	__u32 id;

	/**
	 * @size: The amount of space allocated by userspace at @data. The kernel
	 * will return the number of bytes it wrote, or the size of the queried
	 * information structure if @data is NULL.
	 */
	__u32 size;

	/**
	 * @data: Pointer to the userspace buffer into which the queried
	 * information will be written. May be NULL, in which case nothing is
	 * written and @size is set to the size of the information structure so
	 * userspace can allocate a suitably sized buffer.
	 */
	__u64 data;
};

/**
 * DRM_NOVA_INFO_GPU
 *
 * Query GPU information. The result is returned in a
 * &struct drm_nova_info_gpu.
 */
#define DRM_NOVA_INFO_GPU		0x00

/**
 * struct drm_nova_info_gpu - GPU information
 */
struct drm_nova_info_gpu {
	/**
	 * @architecture: GPU architecture identifier.
	 *
	 * See &enum drm_nova_architecture for currently known architectures.
	 */
	__u32 architecture;

	/**
	 * @chipid: Opaque GPU chip identifier.
	 *
	 * See &enum drm_nova_chipid for currently known chips.
	 */
	__u32 chipid;

	/**
	 * @vram_size: Amount of usable FB, excluding GSP carveouts and protected
	 * regions.
	 */
	__u64 vram_size;

	/**
	 * @vram_bar_size: Size of the PCI BAR aperture used to access VRAM, in bytes.
	 */
	__u64 vram_bar_size;

	/**
	 * @gpu_name: NUL-terminated full GPU name.
	 */
	__u8 gpu_name[64];

	/**
	 * @gpu_short_name: NUL-terminated short GPU name.
	 */
	__u8 gpu_short_name[64];

	/**
	 * @gpu_gid: 16-byte SHA-1 GPU identifier supplied by GSP-RM.
	 */
	__u8 gpu_gid[16];
};

/**
 * DRM_NOVA_INFO_GRAPHICS
 *
 * Query what the graphics engine is made of, and the classes of the channels
 * and of the engine objects on them that userspace builds work for. The result
 * is returned in a &struct drm_nova_info_graphics.
 *
 * Fails with -ENODEV if the graphics engine isn't available.
 */
#define DRM_NOVA_INFO_GRAPHICS		0x01

/**
 * struct drm_nova_info_graphics - graphics engine information
 */
struct drm_nova_info_graphics {
	/**
	 * @sm_major: Major version of the shader instruction set, as the 8 of
	 * sm_86.
	 */
	__u8 sm_major;

	/**
	 * @sm_minor: Minor version of the shader instruction set, as the 6 of
	 * sm_86.
	 */
	__u8 sm_minor;

	/**
	 * @pad: 16 bit padding, 0.
	 */
	__u16 pad;

	/**
	 * @gpc_count: Number of enabled GPCs.
	 */
	__u32 gpc_count;

	/**
	 * @tpc_count: Number of enabled TPCs, of all GPCs.
	 */
	__u32 tpc_count;

	/**
	 * @sm_per_tpc: Number of SMs of a TPC.
	 */
	__u32 sm_per_tpc;

	/**
	 * @max_warps_per_sm: Number of warps an SM runs at most at once.
	 */
	__u32 max_warps_per_sm;

	/**
	 * @class_gpfifo: Class of the GPFIFO channels.
	 */
	__u32 class_gpfifo;

	/**
	 * @class_copy: Class of the copy engine objects.
	 */
	__u32 class_copy;

	/**
	 * @class_2d: Class of the 2D engine objects.
	 */
	__u32 class_2d;

	/**
	 * @class_3d: Class of the 3D engine objects.
	 */
	__u32 class_3d;

	/**
	 * @class_m2mf: Class of the inline-to-memory engine objects, which write
	 * data from the push buffer to memory.
	 */
	__u32 class_m2mf;

	/**
	 * @class_compute: Class of the compute engine objects.
	 */
	__u32 class_compute;
};

/**
 * DRM_NOVA_INFO_VM
 *
 * Query the GPU virtual address spaces of userspace. The result is returned in
 * a &struct drm_nova_info_vm.
 */
#define DRM_NOVA_INFO_VM		0x02

/**
 * struct drm_nova_info_vm - GPU virtual address space information
 */
struct drm_nova_info_vm {
	/**
	 * @va_size: Size of the GPU virtual address spaces of userspace, in
	 * bytes. Their addresses start at 0.
	 */
	__u64 va_size;
};

/**
 * struct drm_nova_vm_init - create the GPU virtual address space of the file
 *
 * Userspace maps memory in the address space at the addresses it chooses, with
 * &struct drm_nova_vm_bind, but in the kernel managed range, where the kernel
 * maps what it needs for userspace, such as the memory of its channels. A file
 * has one address space.
 */
struct drm_nova_vm_init {
	/**
	 * @kernel_managed_addr: Start of the kernel managed range, page aligned.
	 */
	__u64 kernel_managed_addr;

	/**
	 * @kernel_managed_size: Size of the kernel managed range, page aligned
	 * and not 0.
	 */
	__u64 kernel_managed_size;
};

/**
 * DRM_NOVA_VM_BIND_OP_MAP - map memory of a GEM object
 *
 * Mapping over existing mappings replaces the parts it covers.
 */
#define DRM_NOVA_VM_BIND_OP_MAP		0x0

/**
 * DRM_NOVA_VM_BIND_OP_UNMAP - unmap memory
 *
 * The parts of mappings outside of the range unmapped stay mapped.
 */
#define DRM_NOVA_VM_BIND_OP_UNMAP	0x1

/**
 * struct drm_nova_vm_bind_op - an operation on the mappings of the GPU virtual
 * address space of the file
 */
struct drm_nova_vm_bind_op {
	/**
	 * @op: The operation, DRM_NOVA_VM_BIND_OP_*.
	 */
	__u32 op;

	/**
	 * @handle: DRM_NOVA_VM_BIND_OP_MAP: the handle of the GEM object to map.
	 */
	__u32 handle;

	/**
	 * @kind: DRM_NOVA_VM_BIND_OP_MAP: the PTE kind of the memory, how it is
	 * laid out (NV_MMU_PTE_KIND_*). Memory isn't compressed: compressible
	 * kinds are mapped as their uncompressed equivalent.
	 */
	__u32 kind;

	/**
	 * @pad: 32 bit padding, should be 0.
	 */
	__u32 pad;

	/**
	 * @addr: Address of the range, page aligned.
	 */
	__u64 addr;

	/**
	 * @bo_offset: DRM_NOVA_VM_BIND_OP_MAP: offset in the GEM object of the
	 * memory mapped, page aligned.
	 */
	__u64 bo_offset;

	/**
	 * @range: Size of the range, page aligned and not 0.
	 */
	__u64 range;
};

/**
 * struct drm_nova_vm_bind - operate on the mappings of the GPU virtual address
 * space of the file
 *
 * The operations are carried out in order, before the ioctl returns. On
 * failure, those before the one that failed are carried out.
 */
struct drm_nova_vm_bind {
	/**
	 * @op_count: Number of operations at @op_ptr.
	 */
	__u32 op_count;

	/**
	 * @flags: Should be 0.
	 */
	__u32 flags;

	/**
	 * @op_ptr: Pointer to an array of &struct drm_nova_vm_bind_op.
	 */
	__u64 op_ptr;
};

/**
 * DRM_NOVA_INFO_EXEC
 *
 * Query the limits of &struct drm_nova_exec. The result is returned in a
 * &struct drm_nova_info_exec.
 */
#define DRM_NOVA_INFO_EXEC		0x03

/**
 * struct drm_nova_info_exec - work submission information
 */
struct drm_nova_info_exec {
	/**
	 * @push_max: Number of push buffers &struct drm_nova_exec runs at most
	 * at once.
	 */
	__u32 push_max;

	/**
	 * @pad: 32 bit padding, 0.
	 */
	__u32 pad;
};

/**
 * DRM_NOVA_INFO_TIMESTAMP
 *
 * Query the GPU's timer, which the timestamps that work writes count with.
 * The result is returned in a &struct drm_nova_info_timestamp.
 */
#define DRM_NOVA_INFO_TIMESTAMP		0x04

/**
 * struct drm_nova_info_timestamp - the GPU's timer
 */
struct drm_nova_info_timestamp {
	/**
	 * @timestamp: The GPU's timer, a count of nanoseconds.
	 */
	__u64 timestamp;
};

/**
 * DRM_NOVA_ENGINE_GRAPHICS - the graphics engine
 *
 * Its channels have objects of the classes &struct drm_nova_channel_create
 * asks for, among the 3D, compute, 2D and inline-to-memory classes
 * (&struct drm_nova_info_graphics), which userspace binds to subchannels
 * itself.
 */
#define DRM_NOVA_ENGINE_GRAPHICS	0x1

/**
 * DRM_NOVA_ENGINE_COPY - a copy engine
 *
 * Its channels have an object of the copy class, which userspace binds to a
 * subchannel itself.
 */
#define DRM_NOVA_ENGINE_COPY		0x2

/*
 * The classes of the objects of a channel on the graphics engine, for
 * &struct drm_nova_channel_create.
 */
#define DRM_NOVA_CLASS_3D		(1 << 0)
#define DRM_NOVA_CLASS_COMPUTE		(1 << 1)
#define DRM_NOVA_CLASS_2D		(1 << 2)
#define DRM_NOVA_CLASS_M2MF		(1 << 3)

/**
 * struct drm_nova_channel_create - create a channel
 *
 * The channel runs in the GPU virtual address space of the file, which must
 * have been created with &struct drm_nova_vm_init, and maps its memory in its
 * kernel managed range.
 */
struct drm_nova_channel_create {
	/**
	 * @engine: The engine the channel runs on, DRM_NOVA_ENGINE_*.
	 */
	__u32 engine;

	/**
	 * @classes: DRM_NOVA_ENGINE_GRAPHICS: the classes of the objects the
	 * channel has, DRM_NOVA_CLASS_* flags, not 0. DRM_NOVA_ENGINE_COPY:
	 * should be 0, the channel has a copy object.
	 */
	__u32 classes;

	/**
	 * @channel: The handle of the new channel.
	 */
	__u32 channel;

	/**
	 * @pad: 32 bit padding, should be 0.
	 */
	__u32 pad;
};

/**
 * struct drm_nova_channel_destroy - destroy a channel
 *
 * The channel is destroyed once the work it runs is done.
 */
struct drm_nova_channel_destroy {
	/**
	 * @channel: The handle of the channel.
	 */
	__u32 channel;

	/**
	 * @pad: 32 bit padding, should be 0.
	 */
	__u32 pad;
};

/**
 * DRM_NOVA_EXEC_PUSH_NO_PREFETCH - fetch the push buffer once the work before
 * it is done, rather than ahead
 */
#define DRM_NOVA_EXEC_PUSH_NO_PREFETCH	(1 << 0)

/**
 * struct drm_nova_exec_push - a push buffer
 */
struct drm_nova_exec_push {
	/**
	 * @va: GPU virtual address of the push buffer, 4-byte aligned.
	 */
	__u64 va;

	/**
	 * @range: Size of the push buffer in bytes, a multiple of 4, not 0.
	 */
	__u32 range;

	/**
	 * @flags: DRM_NOVA_EXEC_PUSH_* flags.
	 */
	__u32 flags;
};

/**
 * DRM_NOVA_SYNC_SYNCOBJ - a binary synchronization object
 */
#define DRM_NOVA_SYNC_SYNCOBJ		0x0

/**
 * DRM_NOVA_SYNC_TIMELINE_SYNCOBJ - a point of the timeline of a
 * synchronization object
 */
#define DRM_NOVA_SYNC_TIMELINE_SYNCOBJ	0x1

/**
 * DRM_NOVA_SYNC_TYPE_MASK - the bits of the type of a &struct drm_nova_sync
 */
#define DRM_NOVA_SYNC_TYPE_MASK		0xf

/**
 * struct drm_nova_sync - a synchronization object to wait for or to signal
 */
struct drm_nova_sync {
	/**
	 * @flags: Its type, one of DRM_NOVA_SYNC_*, in DRM_NOVA_SYNC_TYPE_MASK.
	 */
	__u32 flags;

	/**
	 * @handle: The handle of the synchronization object.
	 */
	__u32 handle;

	/**
	 * @timeline_value: DRM_NOVA_SYNC_TIMELINE_SYNCOBJ: the point of its
	 * timeline, not 0. DRM_NOVA_SYNC_SYNCOBJ: should be 0.
	 */
	__u64 timeline_value;
};

/**
 * struct drm_nova_exec - run push buffers on a channel
 *
 * The push buffers run in order, after the work submitted to the channel
 * before, once the fences of the synchronization objects to wait for signal.
 * The ioctl returns once they are submitted, without waiting for them to be
 * done: the synchronization objects to signal then get the fence that signals
 * once they are.
 *
 * The ioctl waits for the fences to wait for, and for the channel to have room
 * as the work submitted before completes. It fails with -ETIMEDOUT if that work
 * makes no progress for too long, after which the channel runs no more work:
 * its fences signal with an error, and further submissions fail with -ENODEV.
 * It fails with the error a fence to wait for signaled with, without running
 * the push buffers.
 */
struct drm_nova_exec {
	/**
	 * @channel: The handle of the channel.
	 */
	__u32 channel;

	/**
	 * @push_count: Number of push buffers at @push_ptr, at most
	 * &struct drm_nova_info_exec.push_max. With none, the synchronization
	 * objects to signal get the fence that signals once the work submitted to
	 * the channel before is done, and the ioctl fails with -ENODEV if the
	 * channel runs no more work.
	 */
	__u32 push_count;

	/**
	 * @wait_count: Number of synchronization objects at @wait_ptr to wait for.
	 */
	__u32 wait_count;

	/**
	 * @sig_count: Number of synchronization objects at @sig_ptr to signal.
	 */
	__u32 sig_count;

	/**
	 * @wait_ptr: Pointer to an array of &struct drm_nova_sync to wait for.
	 */
	__u64 wait_ptr;

	/**
	 * @sig_ptr: Pointer to an array of &struct drm_nova_sync to signal.
	 */
	__u64 sig_ptr;

	/**
	 * @push_ptr: Pointer to an array of &struct drm_nova_exec_push.
	 */
	__u64 push_ptr;
};

#define DRM_NOVA_INFO			0x00
#define DRM_NOVA_GEM_CREATE		0x01
#define DRM_NOVA_GEM_INFO		0x02
#define DRM_NOVA_VM_INIT		0x03
#define DRM_NOVA_VM_BIND		0x04
#define DRM_NOVA_CHANNEL_CREATE		0x05
#define DRM_NOVA_CHANNEL_DESTROY	0x06
#define DRM_NOVA_EXEC			0x07

/* Note: this is an enum so that it can be resolved by Rust bindgen. */
enum {
	DRM_IOCTL_NOVA_INFO		= DRM_IOWR(DRM_COMMAND_BASE + DRM_NOVA_INFO,
						   struct drm_nova_info),
	DRM_IOCTL_NOVA_GEM_CREATE	= DRM_IOWR(DRM_COMMAND_BASE + DRM_NOVA_GEM_CREATE,
						   struct drm_nova_gem_create),
	DRM_IOCTL_NOVA_GEM_INFO		= DRM_IOWR(DRM_COMMAND_BASE + DRM_NOVA_GEM_INFO,
						   struct drm_nova_gem_info),
	DRM_IOCTL_NOVA_VM_INIT		= DRM_IOW(DRM_COMMAND_BASE + DRM_NOVA_VM_INIT,
						  struct drm_nova_vm_init),
	DRM_IOCTL_NOVA_VM_BIND		= DRM_IOW(DRM_COMMAND_BASE + DRM_NOVA_VM_BIND,
						  struct drm_nova_vm_bind),
	DRM_IOCTL_NOVA_CHANNEL_CREATE	= DRM_IOWR(DRM_COMMAND_BASE + DRM_NOVA_CHANNEL_CREATE,
						   struct drm_nova_channel_create),
	DRM_IOCTL_NOVA_CHANNEL_DESTROY	= DRM_IOW(DRM_COMMAND_BASE + DRM_NOVA_CHANNEL_DESTROY,
						  struct drm_nova_channel_destroy),
	DRM_IOCTL_NOVA_EXEC		= DRM_IOW(DRM_COMMAND_BASE + DRM_NOVA_EXEC,
						  struct drm_nova_exec),
};

#if defined(__cplusplus)
}
#endif

#endif /* __NOVA_DRM_H__ */

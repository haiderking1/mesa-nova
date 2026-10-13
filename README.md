# mesa-nova, haider's Mesa fork

Mesa with the userspace half of my [nova](https://github.com/haiderking1/nova)
stack: NVK (Vulkan) and Zink (OpenGL) running on nova, the Rust NVIDIA kernel
driver, on an RTX 3070 (GA104). Experimental and not upstream; the nova uAPI
it talks to is my fork's own and will change.

For Mesa itself, see `README.rst` and https://mesa3d.org.

## Branches

- **`soka/nova`**: NVK on nova.
  - An nvkmd backend for nova-drm (`src/nouveau/vulkan/nvkmd/nova`).
  - Vulkan Video decode for NVK: **VP9** (profiles 0 and 2) and **AV1** (main
    profile, 8/10-bit, film grain, monochrome), next to upstream's H.264.
    Bit-exact on the libvpx/libaom conformance vectors and passing the Vulkan
    CTS decode tests it supports.
  - Block-linear DRM format modifiers for multi-planar (video) images.
- **`soka/zink-video-2`**: `soka/nova` plus VA-API on Zink.
  - Builds on mesa!31517 (Dave Airlie, Mike Blumenkrantz), which keeps its
    authors.
  - Adds VP9 and AV1 decode through VA-API.
  - Fixes the video path: decoding on a context of the codec's own, plane
    sizes and usages of VA surfaces, device loss hangs, video processing of
    progressive surfaces, VP9 resolution changes.
  - This is what lets Firefox-based browsers (Zen) decode YouTube's VP9/AV1
    on the GPU.

Fixes that apply to upstream are being sent there as merge requests.

## Running it

Needs the nova fork's kernel modules and
[libdrm-nova](https://github.com/haiderking1/libdrm-nova). Then point the
Vulkan loader at the build's NVK ICD, set `NVK_EXPERIMENTAL=video` for video
decode, and for VA-API use `LIBVA_DRIVER_NAME=zink` with `LIBVA_DRIVERS_PATH`
set to the build's `src/gallium/targets/dri`.

## Credits

Mesa is the work of its many developers; commits taken from others keep
their authors. Much of the work in these branches was written with the help
of an AI assistant (Claude), which the commits say.

## License

Mesa's licenses (mostly MIT); see `docs/license.rst`.

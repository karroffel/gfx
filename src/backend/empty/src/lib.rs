//! Dummy backend implementation to test the code for compile errors
//! outside of the graphics development environment.

extern crate gfx_hal as hal;
#[cfg(feature = "winit")]
extern crate winit;

use crate::hal::range::RangeArg;
use crate::hal::{
    buffer, command, device, error, format, image, mapping, memory, pass, pool, pso, query, queue,
};
use std::borrow::Borrow;
use std::ops::Range;

pub type UniqueId = u128;

/// generate a new unique ID
fn gen_id() -> UniqueId {
    let id = uuid::Uuid::new_v4();
    let bytes = id.as_bytes();
    let bytes_ptr = bytes.as_ptr();

    unsafe { std::ptr::read_unaligned(bytes_ptr as *const u128) }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct Backend;

impl hal::Backend for Backend {
    type PhysicalDevice = PhysicalDevice;
    type Device = Device;

    type Surface = Surface;
    type Swapchain = Swapchain;

    type QueueFamily = QueueFamily;
    type CommandQueue = CommandQueue;
    type CommandBuffer = CommandBuffer;

    type ShaderModule = ();
    type RenderPass = ();

    type Framebuffer = ();
    type Memory = ();
    type CommandPool = CommandPool;

    type Buffer = ();
    type BufferView = ();
    type Image = ();
    type ImageView = ();
    type Sampler = ();

    type ComputePipeline = ();
    type GraphicsPipeline = ();
    type PipelineCache = ();
    type PipelineLayout = ();
    type DescriptorPool = DescriptorPool;
    type DescriptorSet = ();
    type DescriptorSetLayout = ();

    type Fence = ();
    type Semaphore = ();
    type QueryPool = ();
}

#[derive(Clone, Debug)]
pub struct PhysicalDevice {
    pub format_properties: image::FormatProperties,
    pub memory_properties: hal::MemoryProperties,
    pub features: hal::Features,
    pub limits: hal::Limits,
}

impl Default for PhysicalDevice {
    fn default() -> Self {
        let format_properties = {
            image::FormatProperties {
                max_extent: hal::image::Extent {
                    width: 65535,
                    height: 65535,
                    depth: 65535,
                },
                max_levels: u8::max_value(),
                max_layers: 65535,
                sample_count_mask: 64,
                max_resource_size: 8 * 1024 * 1024 * 1024, // 8 Go
            }
        };
        let memory_properties = {
            // values taken from a RX 590 (AMD RADV POLARIS10)

            let memory_heaps = {
                let device_local = 8321499136;
                let host_mem = 8589934592;
                let cached = 268435456;

                vec![device_local, cached, host_mem]
            };

            let memory_types = {
                let device_local = hal::MemoryType {
                    heap_index: 0,
                    properties: hal::memory::Properties::DEVICE_LOCAL,
                };

                let host_mem = hal::MemoryType {
                    heap_index: 2,
                    properties: hal::memory::Properties::COHERENT
                        | hal::memory::Properties::CPU_VISIBLE,
                };

                let device_local_cpu_visible = hal::MemoryType {
                    heap_index: 1,
                    properties: hal::memory::Properties::DEVICE_LOCAL
                        | hal::memory::Properties::COHERENT
                        | hal::memory::Properties::CPU_VISIBLE,
                };

                let cpu_cached = hal::MemoryType {
                    heap_index: 2,
                    properties: hal::memory::Properties::CPU_CACHED
                        | hal::memory::Properties::COHERENT
                        | hal::memory::Properties::CPU_VISIBLE,
                };

                vec![device_local, host_mem, device_local_cpu_visible, cpu_cached]
            };

            hal::MemoryProperties {
                memory_types,
                memory_heaps,
            }
        };

        let limits = {
            // taken from a RX 590 (AMD RADV POLARIS10)
            hal::Limits {
                max_image_1d_size: 16384,
                max_image_2d_size: 16384,
                max_image_3d_size: 2048,
                max_image_cube_size: 16384,
                max_image_array_layers: 0,
                max_texel_elements: 134217728,
                max_uniform_buffer_range: 0,
                max_storage_buffer_range: 0,
                max_push_constants_size: 0,
                max_memory_allocation_count: 0,
                max_sampler_allocation_count: 0,
                max_bound_descriptor_sets: 0,
                max_per_stage_descriptor_samplers: 0,
                max_per_stage_descriptor_uniform_buffers: 0,
                max_per_stage_descriptor_storage_buffers: 0,
                max_per_stage_descriptor_sampled_images: 0,
                max_per_stage_descriptor_storage_images: 0,
                max_per_stage_descriptor_input_attachments: 0,
                max_per_stage_resources: 0,
                max_descriptor_set_samplers: 0,
                max_descriptor_set_uniform_buffers: 0,
                max_descriptor_set_uniform_buffers_dynamic: 0,
                max_descriptor_set_storage_buffers: 0,
                max_descriptor_set_storage_buffers_dynamic: 0,
                max_descriptor_set_sampled_images: 0,
                max_descriptor_set_storage_images: 0,
                max_descriptor_set_input_attachments: 0,
                max_vertex_input_attributes: 32,
                max_vertex_input_bindings: 32,
                max_vertex_input_attribute_offset: 2047,
                max_vertex_input_binding_stride: 2048,
                max_vertex_output_components: 128,
                max_patch_size: 32,
                max_geometry_shader_invocations: 0,
                max_geometry_input_components: 0,
                max_geometry_output_components: 0,
                max_geometry_output_vertices: 0,
                max_geometry_total_output_components: 0,
                max_fragment_input_components: 0,
                max_fragment_output_attachments: 0,
                max_fragment_dual_source_attachments: 0,
                max_fragment_combined_output_resources: 0,
                max_compute_shared_memory_size: 0,
                max_compute_work_group_count: [65535, 65535, 65535],
                max_compute_work_group_invocations: 0,
                max_compute_work_group_size: [2048, 2048, 2048],
                max_draw_indexed_index_value: 0,
                max_draw_indirect_count: 0,
                max_sampler_lod_bias: 0.0,
                max_sampler_anisotropy: 16.0,
                max_viewports: 16,
                max_viewport_dimensions: [16384, 16384],
                max_framebuffer_extent: image::Extent {
                    width: 16384,
                    height: 16384,
                    depth: 1024,
                },
                min_memory_map_alignment: 0,
                buffer_image_granularity: 64,
                min_texel_buffer_offset_alignment: 1,
                min_uniform_buffer_offset_alignment: 4,
                min_storage_buffer_offset_alignment: 4,
                framebuffer_color_samples_count: 15,
                framebuffer_depth_samples_count: 15,
                framebuffer_stencil_samples_count: 15,
                max_color_attachments: 8,
                standard_sample_locations: false,
                optimal_buffer_copy_offset_alignment: 128,
                optimal_buffer_copy_pitch_alignment: 128,
                non_coherent_atom_size: 64,
                min_vertex_input_binding_stride_alignment: 1,
            }
        };

        PhysicalDevice {
            format_properties,
            memory_properties,
            features: hal::Features::all(),
            limits,
        }
    }
}

impl hal::PhysicalDevice<Backend> for PhysicalDevice {
    unsafe fn open(
        &self,
        families: &[(&QueueFamily, &[hal::QueuePriority])],
        _: hal::Features,
    ) -> Result<hal::Gpu<Backend>, error::DeviceCreationError> {
        let queues = {
            families
                .into_iter()
                .map(|&(family, priority)| {
                    let fam = family.clone();
                    let mut family_raw = hal::backend::RawQueueGroup::new(fam);

                    for id in 0..priority.len() {
                        let queue = CommandQueue { id };

                        family_raw.add_queue(queue);
                    }

                    family_raw
                })
                .collect()
        };

        Ok(hal::Gpu {
            device: Device::new(),
            queues: queue::Queues::new(queues),
        })
    }

    fn format_properties(&self, _: Option<format::Format>) -> format::Properties {
        Default::default()
    }

    fn image_format_properties(
        &self,
        _: format::Format,
        _dim: u8,
        _: image::Tiling,
        _: image::Usage,
        _: image::ViewCapabilities,
    ) -> Option<image::FormatProperties> {
        Some(self.format_properties)
    }

    fn memory_properties(&self) -> hal::MemoryProperties {
        self.memory_properties.clone()
    }

    fn features(&self) -> hal::Features {
        self.features
    }

    fn limits(&self) -> hal::Limits {
        self.limits
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct CommandQueue {
    pub id: usize,
}

impl queue::RawCommandQueue<Backend> for CommandQueue {
    unsafe fn submit<'a, T, Ic, S, Iw, Is>(
        &mut self,
        _: queue::Submission<Ic, Iw, Is>,
        _: Option<&()>,
    ) where
        T: 'a + Borrow<CommandBuffer>,
        Ic: IntoIterator<Item = &'a T>,
        S: 'a + Borrow<()>,
        Iw: IntoIterator<Item = (&'a S, pso::PipelineStage)>,
        Is: IntoIterator<Item = &'a S>,
    {
        // Do nothing
    }

    unsafe fn present<'a, W, Is, S, Iw>(&mut self, _: Is, _: Iw) -> Result<(), ()>
    where
        W: 'a + Borrow<Swapchain>,
        Is: IntoIterator<Item = (&'a W, hal::SwapImageIndex)>,
        S: 'a + Borrow<()>,
        Iw: IntoIterator<Item = &'a S>,
    {
        // Nothing to do
        Ok(())
    }

    fn wait_idle(&self) -> Result<(), error::HostExecutionError> {
        // Nothing to do
        Ok(())
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct Device;

impl Device {
    pub fn new() -> Self {
        Device
    }
}

impl hal::Device<Backend> for Device {
    unsafe fn create_command_pool(
        &self,
        _: queue::QueueFamilyId,
        _: pool::CommandPoolCreateFlags,
    ) -> Result<CommandPool, device::OutOfMemory> {
        Ok(CommandPool::new())
    }

    unsafe fn destroy_command_pool(&self, _: CommandPool) {}

    unsafe fn allocate_memory(
        &self,
        _: hal::MemoryTypeId,
        _: u64,
    ) -> Result<(), device::AllocationError> {
        unimplemented!()
    }

    unsafe fn create_render_pass<'a, IA, IS, ID>(
        &self,
        _: IA,
        _: IS,
        _: ID,
    ) -> Result<(), device::OutOfMemory>
    where
        IA: IntoIterator,
        IA::Item: Borrow<pass::Attachment>,
        IS: IntoIterator,
        IS::Item: Borrow<pass::SubpassDesc<'a>>,
        ID: IntoIterator,
        ID::Item: Borrow<pass::SubpassDependency>,
    {
        unimplemented!()
    }

    unsafe fn create_pipeline_layout<IS, IR>(&self, _: IS, _: IR) -> Result<(), device::OutOfMemory>
    where
        IS: IntoIterator,
        IS::Item: Borrow<()>,
        IR: IntoIterator,
        IR::Item: Borrow<(pso::ShaderStageFlags, Range<u32>)>,
    {
        unimplemented!()
    }

    unsafe fn create_pipeline_cache(
        &self,
        _data: Option<&[u8]>,
    ) -> Result<(), device::OutOfMemory> {
        unimplemented!()
    }

    unsafe fn get_pipeline_cache_data(&self, _cache: &()) -> Result<Vec<u8>, device::OutOfMemory> {
        unimplemented!()
    }

    unsafe fn destroy_pipeline_cache(&self, _: ()) {
        unimplemented!()
    }

    unsafe fn merge_pipeline_caches<I>(&self, _: &(), _: I) -> Result<(), device::OutOfMemory>
    where
        I: IntoIterator,
        I::Item: Borrow<()>,
    {
        unimplemented!()
    }

    unsafe fn create_framebuffer<I>(
        &self,
        _: &(),
        _: I,
        _: image::Extent,
    ) -> Result<(), device::OutOfMemory>
    where
        I: IntoIterator,
        I::Item: Borrow<()>,
    {
        unimplemented!()
    }

    unsafe fn create_shader_module(&self, _: &[u8]) -> Result<(), device::ShaderError> {
        unimplemented!()
    }

    unsafe fn create_sampler(&self, _: image::SamplerInfo) -> Result<(), device::AllocationError> {
        unimplemented!()
    }
    unsafe fn create_buffer(&self, _: u64, _: buffer::Usage) -> Result<(), buffer::CreationError> {
        unimplemented!()
    }

    unsafe fn get_buffer_requirements(&self, _: &()) -> memory::Requirements {
        unimplemented!()
    }

    unsafe fn bind_buffer_memory(
        &self,
        _: &(),
        _: u64,
        _: &mut (),
    ) -> Result<(), device::BindError> {
        unimplemented!()
    }

    unsafe fn create_buffer_view<R: RangeArg<u64>>(
        &self,
        _: &(),
        _: Option<format::Format>,
        _: R,
    ) -> Result<(), buffer::ViewCreationError> {
        unimplemented!()
    }

    unsafe fn create_image(
        &self,
        _: image::Kind,
        _: image::Level,
        _: format::Format,
        _: image::Tiling,
        _: image::Usage,
        _: image::ViewCapabilities,
    ) -> Result<(), image::CreationError> {
        unimplemented!()
    }

    unsafe fn get_image_requirements(&self, _: &()) -> memory::Requirements {
        unimplemented!()
    }

    unsafe fn get_image_subresource_footprint(
        &self,
        _: &(),
        _: image::Subresource,
    ) -> image::SubresourceFootprint {
        unimplemented!()
    }

    unsafe fn bind_image_memory(
        &self,
        _: &(),
        _: u64,
        _: &mut (),
    ) -> Result<(), device::BindError> {
        unimplemented!()
    }

    unsafe fn create_image_view(
        &self,
        _: &(),
        _: image::ViewKind,
        _: format::Format,
        _: format::Swizzle,
        _: image::SubresourceRange,
    ) -> Result<(), image::ViewError> {
        unimplemented!()
    }

    unsafe fn create_descriptor_pool<I>(
        &self,
        _: usize,
        _: I,
        _: pso::DescriptorPoolCreateFlags,
    ) -> Result<DescriptorPool, device::OutOfMemory>
    where
        I: IntoIterator,
        I::Item: Borrow<pso::DescriptorRangeDesc>,
    {
        unimplemented!()
    }

    unsafe fn create_descriptor_set_layout<I, J>(
        &self,
        _: I,
        _: J,
    ) -> Result<(), device::OutOfMemory>
    where
        I: IntoIterator,
        I::Item: Borrow<pso::DescriptorSetLayoutBinding>,
        J: IntoIterator,
        J::Item: Borrow<()>,
    {
        unimplemented!()
    }

    unsafe fn write_descriptor_sets<'a, I, J>(&self, _: I)
    where
        I: IntoIterator<Item = pso::DescriptorSetWrite<'a, Backend, J>>,
        J: IntoIterator,
        J::Item: Borrow<pso::Descriptor<'a, Backend>>,
    {
        unimplemented!()
    }

    unsafe fn copy_descriptor_sets<'a, I>(&self, _: I)
    where
        I: IntoIterator,
        I::Item: Borrow<pso::DescriptorSetCopy<'a, Backend>>,
    {
        unimplemented!()
    }

    fn create_semaphore(&self) -> Result<(), device::OutOfMemory> {
        unimplemented!()
    }

    fn create_fence(&self, _: bool) -> Result<(), device::OutOfMemory> {
        unimplemented!()
    }

    unsafe fn get_fence_status(&self, _: &()) -> Result<bool, device::DeviceLost> {
        unimplemented!()
    }

    unsafe fn create_query_pool(&self, _: query::Type, _: u32) -> Result<(), query::CreationError> {
        unimplemented!()
    }

    unsafe fn destroy_query_pool(&self, _: ()) {
        unimplemented!()
    }

    unsafe fn get_query_pool_results(
        &self,
        _: &(),
        _: Range<query::Id>,
        _: &mut [u8],
        _: buffer::Offset,
        _: query::ResultFlags,
    ) -> Result<bool, device::OomOrDeviceLost> {
        unimplemented!()
    }

    unsafe fn map_memory<R: RangeArg<u64>>(&self, _: &(), _: R) -> Result<*mut u8, mapping::Error> {
        unimplemented!()
    }

    unsafe fn unmap_memory(&self, _: &()) {
        unimplemented!()
    }

    unsafe fn flush_mapped_memory_ranges<'a, I, R>(&self, _: I) -> Result<(), device::OutOfMemory>
    where
        I: IntoIterator,
        I::Item: Borrow<(&'a (), R)>,
        R: RangeArg<u64>,
    {
        unimplemented!()
    }

    unsafe fn invalidate_mapped_memory_ranges<'a, I, R>(
        &self,
        _: I,
    ) -> Result<(), device::OutOfMemory>
    where
        I: IntoIterator,
        I::Item: Borrow<(&'a (), R)>,
        R: RangeArg<u64>,
    {
        unimplemented!()
    }

    unsafe fn free_memory(&self, _: ()) {
        unimplemented!()
    }

    unsafe fn destroy_shader_module(&self, _: ()) {
        unimplemented!()
    }

    unsafe fn destroy_render_pass(&self, _: ()) {
        unimplemented!()
    }

    unsafe fn destroy_pipeline_layout(&self, _: ()) {
        unimplemented!()
    }
    unsafe fn destroy_graphics_pipeline(&self, _: ()) {
        unimplemented!()
    }
    unsafe fn destroy_compute_pipeline(&self, _: ()) {
        unimplemented!()
    }
    unsafe fn destroy_framebuffer(&self, _: ()) {
        unimplemented!()
    }

    unsafe fn destroy_buffer(&self, _: ()) {
        unimplemented!()
    }
    unsafe fn destroy_buffer_view(&self, _: ()) {
        unimplemented!()
    }
    unsafe fn destroy_image(&self, _: ()) {
        unimplemented!()
    }
    unsafe fn destroy_image_view(&self, _: ()) {
        unimplemented!()
    }
    unsafe fn destroy_sampler(&self, _: ()) {
        unimplemented!()
    }

    unsafe fn destroy_descriptor_pool(&self, _: DescriptorPool) {
        unimplemented!()
    }

    unsafe fn destroy_descriptor_set_layout(&self, _: ()) {
        unimplemented!()
    }

    unsafe fn destroy_fence(&self, _: ()) {
        unimplemented!()
    }

    unsafe fn destroy_semaphore(&self, _: ()) {
        unimplemented!()
    }

    unsafe fn create_swapchain(
        &self,
        _: &mut Surface,
        _: hal::SwapchainConfig,
        _: Option<Swapchain>,
    ) -> Result<(Swapchain, hal::Backbuffer<Backend>), hal::window::CreationError> {
        unimplemented!()
    }

    unsafe fn destroy_swapchain(&self, _: Swapchain) {
        unimplemented!()
    }

    fn wait_idle(&self) -> Result<(), error::HostExecutionError> {
        unimplemented!()
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct QueueFamily {
    pub queue_type: hal::QueueType,
    pub max_queues: usize,
    pub id: queue::QueueFamilyId,
}

impl QueueFamily {
    pub fn new(queue_type: hal::QueueType) -> Self {
        let id = gen_id() as usize;

        QueueFamily {
            queue_type,
            max_queues: 16,
            id: queue::QueueFamilyId(id),
        }
    }
}

impl queue::QueueFamily for QueueFamily {
    fn queue_type(&self) -> hal::QueueType {
        self.queue_type
    }
    fn max_queues(&self) -> usize {
        self.max_queues
    }
    fn id(&self) -> queue::QueueFamilyId {
        self.id
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct CommandPool {
    pub id: UniqueId,
    pub buffers: Vec<CommandBuffer>,
}

impl CommandPool {
    pub fn new() -> Self {
        CommandPool {
            id: gen_id(),
            buffers: vec![],
        }
    }
}

impl pool::RawCommandPool<Backend> for CommandPool {
    unsafe fn reset(&mut self) {
        self.buffers.clear();
    }

    fn allocate_one(&mut self, _level: command::RawLevel) -> CommandBuffer {
        let buf = CommandBuffer::new();

        self.buffers.push(buf.clone());

        buf
    }

    unsafe fn free<I>(&mut self, bufs: I)
    where
        I: IntoIterator<Item = CommandBuffer>,
    {
        for buf in bufs {
            let idx = self
                .buffers
                .iter()
                .enumerate()
                .find(|(_i, cmd)| cmd.id == buf.id);

            if let Some((idx, _)) = idx {
                self.buffers.swap_remove(idx);
            }
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct CommandBuffer {
    pub id: UniqueId,
}

impl CommandBuffer {
    pub fn new() -> Self {
        CommandBuffer { id: gen_id() }
    }
}

impl command::RawCommandBuffer<Backend> for CommandBuffer {
    unsafe fn begin(
        &mut self,
        _: command::CommandBufferFlags,
        _: command::CommandBufferInheritanceInfo<Backend>,
    ) {
    }

    unsafe fn finish(&mut self) {}

    unsafe fn reset(&mut self, _: bool) {}

    unsafe fn pipeline_barrier<'a, T>(
        &mut self,
        _: Range<pso::PipelineStage>,
        _: memory::Dependencies,
        _: T,
    ) where
        T: IntoIterator,
        T::Item: Borrow<memory::Barrier<'a, Backend>>,
    {
    }

    unsafe fn fill_buffer<R>(&mut self, _: &(), _: R, _: u32)
    where
        R: RangeArg<buffer::Offset>,
    {
    }

    unsafe fn update_buffer(&mut self, _: &(), _: buffer::Offset, _: &[u8]) {}

    unsafe fn clear_image<T>(
        &mut self,
        _: &(),
        _: image::Layout,
        _: command::ClearColorRaw,
        _: command::ClearDepthStencilRaw,
        _: T,
    ) where
        T: IntoIterator,
        T::Item: Borrow<image::SubresourceRange>,
    {
    }

    unsafe fn clear_attachments<T, U>(&mut self, _: T, _: U)
    where
        T: IntoIterator,
        T::Item: Borrow<command::AttachmentClear>,
        U: IntoIterator,
        U::Item: Borrow<pso::ClearRect>,
    {
    }

    unsafe fn resolve_image<T>(&mut self, _: &(), _: image::Layout, _: &(), _: image::Layout, _: T)
    where
        T: IntoIterator,
        T::Item: Borrow<command::ImageResolve>,
    {
    }

    unsafe fn blit_image<T>(
        &mut self,
        _: &(),
        _: image::Layout,
        _: &(),
        _: image::Layout,
        _: image::Filter,
        _: T,
    ) where
        T: IntoIterator,
        T::Item: Borrow<command::ImageBlit>,
    {
    }

    unsafe fn bind_index_buffer(&mut self, _: buffer::IndexBufferView<Backend>) {}

    unsafe fn bind_vertex_buffers<I, T>(&mut self, _: u32, _: I)
    where
        I: IntoIterator<Item = (T, buffer::Offset)>,
        T: Borrow<()>,
    {
    }

    unsafe fn set_viewports<T>(&mut self, _: u32, _: T)
    where
        T: IntoIterator,
        T::Item: Borrow<pso::Viewport>,
    {
    }

    unsafe fn set_scissors<T>(&mut self, _: u32, _: T)
    where
        T: IntoIterator,
        T::Item: Borrow<pso::Rect>,
    {
    }

    unsafe fn set_stencil_reference(&mut self, _: pso::Face, _: pso::StencilValue) {}

    unsafe fn set_stencil_read_mask(&mut self, _: pso::Face, _: pso::StencilValue) {}

    unsafe fn set_stencil_write_mask(&mut self, _: pso::Face, _: pso::StencilValue) {}

    unsafe fn set_blend_constants(&mut self, _: pso::ColorValue) {}

    unsafe fn set_depth_bounds(&mut self, _: Range<f32>) {}

    unsafe fn set_line_width(&mut self, _: f32) {}

    unsafe fn set_depth_bias(&mut self, _: pso::DepthBias) {}

    unsafe fn begin_render_pass<T>(
        &mut self,
        _: &(),
        _: &(),
        _: pso::Rect,
        _: T,
        _: command::SubpassContents,
    ) where
        T: IntoIterator,
        T::Item: Borrow<command::ClearValueRaw>,
    {
    }

    unsafe fn next_subpass(&mut self, _: command::SubpassContents) {}

    unsafe fn end_render_pass(&mut self) {}

    unsafe fn bind_graphics_pipeline(&mut self, _: &()) {}

    unsafe fn bind_graphics_descriptor_sets<I, J>(&mut self, _: &(), _: usize, _: I, _: J)
    where
        I: IntoIterator,
        I::Item: Borrow<()>,
        J: IntoIterator,
        J::Item: Borrow<command::DescriptorSetOffset>,
    {
    }

    unsafe fn bind_compute_pipeline(&mut self, _: &()) {}

    unsafe fn bind_compute_descriptor_sets<I, J>(&mut self, _: &(), _: usize, _: I, _: J)
    where
        I: IntoIterator,
        I::Item: Borrow<()>,
        J: IntoIterator,
        J::Item: Borrow<command::DescriptorSetOffset>,
    {
    }

    unsafe fn dispatch(&mut self, _: hal::WorkGroupCount) {}

    unsafe fn dispatch_indirect(&mut self, _: &(), _: buffer::Offset) {}

    unsafe fn copy_buffer<T>(&mut self, _: &(), _: &(), _: T)
    where
        T: IntoIterator,
        T::Item: Borrow<command::BufferCopy>,
    {
    }

    unsafe fn copy_image<T>(&mut self, _: &(), _: image::Layout, _: &(), _: image::Layout, _: T)
    where
        T: IntoIterator,
        T::Item: Borrow<command::ImageCopy>,
    {
    }

    unsafe fn copy_buffer_to_image<T>(&mut self, _: &(), _: &(), _: image::Layout, _: T)
    where
        T: IntoIterator,
        T::Item: Borrow<command::BufferImageCopy>,
    {
    }

    unsafe fn copy_image_to_buffer<T>(&mut self, _: &(), _: image::Layout, _: &(), _: T)
    where
        T: IntoIterator,
        T::Item: Borrow<command::BufferImageCopy>,
    {
    }

    unsafe fn draw(&mut self, _: Range<hal::VertexCount>, _: Range<hal::InstanceCount>) {}

    unsafe fn draw_indexed(
        &mut self,
        _: Range<hal::IndexCount>,
        _: hal::VertexOffset,
        _: Range<hal::InstanceCount>,
    ) {
    }

    unsafe fn draw_indirect(&mut self, _: &(), _: buffer::Offset, _: hal::DrawCount, _: u32) {}

    unsafe fn draw_indexed_indirect(
        &mut self,
        _: &(),
        _: buffer::Offset,
        _: hal::DrawCount,
        _: u32,
    ) {
    }

    unsafe fn begin_query(&mut self, _: query::Query<Backend>, _: query::ControlFlags) {}

    unsafe fn end_query(&mut self, _: query::Query<Backend>) {}

    unsafe fn reset_query_pool(&mut self, _: &(), _: Range<query::Id>) {}

    unsafe fn copy_query_pool_results(
        &mut self,
        _: &(),
        _: Range<query::Id>,
        _: &(),
        _: buffer::Offset,
        _: buffer::Offset,
        _: query::ResultFlags,
    ) {
    }

    unsafe fn write_timestamp(&mut self, _: pso::PipelineStage, _: query::Query<Backend>) {}

    unsafe fn push_graphics_constants(
        &mut self,
        _: &(),
        _: pso::ShaderStageFlags,
        _: u32,
        _: &[u32],
    ) {
    }

    unsafe fn push_compute_constants(&mut self, _: &(), _: u32, _: &[u32]) {}

    unsafe fn execute_commands<'a, T, I>(&mut self, _: I)
    where
        T: 'a + Borrow<CommandBuffer>,
        I: IntoIterator<Item = &'a T>,
    {
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct DescriptorPool;

impl pso::DescriptorPool<Backend> for DescriptorPool {
    unsafe fn free_sets<I>(&mut self, _descriptor_sets: I)
    where
        I: IntoIterator<Item = ()>,
    {
        unimplemented!()
    }

    unsafe fn allocate_set(&mut self, _layout: &()) -> Result<(), pso::AllocationError> {
        unimplemented!()
    }

    unsafe fn reset(&mut self) {
        unimplemented!()
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct Surface {
    pub id: UniqueId,
}

impl Surface {
    pub fn new() -> Self {
        Surface { id: gen_id() }
    }
}

impl hal::Surface<Backend> for Surface {
    fn kind(&self) -> hal::image::Kind {
        unimplemented!()
    }

    fn compatibility(
        &self,
        _: &PhysicalDevice,
    ) -> (
        hal::SurfaceCapabilities,
        Option<Vec<format::Format>>,
        Vec<hal::PresentMode>,
    ) {
        unimplemented!()
    }

    fn supports_queue_family(&self, queue_family: &QueueFamily) -> bool {
        match queue_family.queue_type() {
            QueueType::General | QueueType::Graphics => true,
            _ => false,
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct Swapchain;

impl hal::Swapchain<Backend> for Swapchain {
    unsafe fn acquire_image(
        &mut self,
        _: u64,
        _: Option<&()>,
        _: Option<&()>,
    ) -> Result<hal::SwapImageIndex, hal::AcquireError> {
        unimplemented!()
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct Instance;

impl Instance {
    /// Create instance.
    pub fn create(_name: &str, _version: u32) -> Self {
        Instance
    }

    #[cfg(feature = "winit")]
    pub fn create_surface(&self, _: &winit::Window) -> Surface {
        Surface::new()
    }
}

impl hal::Instance for Instance {
    type Backend = Backend;
    fn enumerate_adapters(&self) -> Vec<hal::Adapter<Backend>> {
        let info = hal::AdapterInfo {
            name: "EmptyDevice".to_string(),
            vendor: 42,
            device: 1337,
            device_type: hal::adapter::DeviceType::DiscreteGpu,
        };

        let queue_families = vec![
            QueueFamily::new(hal::QueueType::General),
            QueueFamily::new(hal::QueueType::Graphics),
            QueueFamily::new(hal::QueueType::Compute),
            QueueFamily::new(hal::QueueType::Transfer),
        ];

        let default_adapter = hal::Adapter {
            physical_device: PhysicalDevice::default(),
            queue_families,
            info,
        };

        vec![default_adapter]
    }
}

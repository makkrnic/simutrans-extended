/*
 * This file is part of the Simutrans-Extended project under the Artistic License.
 * (see LICENSE.txt)
 */

#ifndef SYS_SIMSYS_S2_VULKAN_H
#define SYS_SIMSYS_S2_VULKAN_H


#ifdef ALT_SDL_DIR
#include "SDL.h"
#include "SDL_vulkan.h"
#else
#include <SDL2/SDL.h>
#include <SDL2/SDL_vulkan.h>
#endif

#include <vulkan/vulkan.h>
#include <vector>
#include <chrono>

#include "../display/window.h"
#include "../display/viewport.h"
#include "../display/scr_coord.h"

struct Vertices {
	VkBuffer       buffer;
	VkDeviceMemory memory;
};

struct Indices {
	VkBuffer       buffer;
	VkDeviceMemory memory;
	uint32_t       count;
};

// TODO: Use sim_renderer_t as an API, and implement SDL, GDI etc in separate subclasses?
class sim_renderer_t {
public:
	sim_renderer_t(sim_window_t *_window) : window(_window) {
		run();
	};

	void run();
	void draw_frame();

	void set_viewport(viewport_t *_viewport) { viewport = _viewport; }

	float consume_fps_change() {
		if (!fps_updated) {
			return -1.0;
		}

		fps_updated = false;
		return last_fps;
	}

private:
	int width;
	int height;
	bool fullscreen;

	int frame_counter = 0;
	std::chrono::steady_clock::time_point last_timestamp = std::chrono::steady_clock::now();
	float last_fps = 0;
	bool fps_updated = false;

	viewport_t *viewport = nullptr;

	sim_window_t *window = nullptr;

	VkInstance instance = nullptr;
	VkDebugUtilsMessengerEXT debugMessenger = nullptr;
	VkSurfaceKHR surface = nullptr;

	VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
	VkDevice device;

	VkQueue graphics_queue;
	VkQueue present_queue;

	VkSwapchainKHR swap_chain;
	std::vector<VkImage> swap_chain_images;
	VkFormat swap_chain_image_format;
	VkExtent2D swap_chain_extent;
	std::vector<VkImageView> swap_chain_image_views;
	std::vector<VkFramebuffer> swap_chain_framebuffers;

	VkRenderPass renderPass;
	VkRenderPass renderPass_grid;
	VkDescriptorSetLayout descriptor_set_layout;
	VkPipelineLayout pipeline_layout;
	VkPipeline graphicsPipeline;

	VkCommandPool command_pool;

	VkBuffer vertex_buffer;
	VkDeviceMemory vertex_buffer_memory;
	VkBuffer index_buffer;
	VkDeviceMemory index_buffer_memory;

	Vertices tiles_vertices = { VK_NULL_HANDLE };
	Indices tiles_grid_indices = { VK_NULL_HANDLE };

	std::vector<VkBuffer> uniform_buffers;
	std::vector<VkDeviceMemory> uniform_buffers_memory;

	VkDescriptorPool descriptor_pool;
	std::vector<VkDescriptorSet> descriptor_sets;

	std::vector<VkCommandBuffer> command_buffers;

	std::vector<VkSemaphore> image_available_semaphores;
	std::vector<VkSemaphore> render_finished_semaphores;
	std::vector<VkFence> in_flight_fences;
	std::vector<VkFence> images_in_flight;
	size_t current_frame = 0;

	bool framebuffer_resized = false;

	uint32_t mip_levels;


	void init_vulkan();
	void cleanup();
	void cleanup_swap_chain();


	void create_vulkan_instance();
	void setup_debug_messenger();
	void create_surface();
	void pick_physical_device();
	void create_logical_device();
	void recreate_swap_chain();
	void create_swap_chain();
	void create_image_views();
	void create_render_pass();
	void create_descriptor_set_layout();
	void create_graphics_pipeline();



	void create_framebuffers();
	void create_command_pool();
	void create_vertex_buffer();
	void create_index_buffer();
	void create_uniform_buffers();
	void create_descriptor_pool();
	void create_descriptor_sets();
	void create_command_buffers();
	void create_sync_objects();


	void prepare_tiles_rendering();

	void copy_buffer(VkBuffer src_buffer, VkBuffer dst_buffer, VkDeviceSize size);
};

#endif //SIMSYS_S2_VULKAN_H

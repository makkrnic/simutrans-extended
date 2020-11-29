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
#include <vulkan/vulkan.h>
#include <vector>

#endif

#include "../display/scr_coord.h"

// TODO: Use sim_window_t as an API, and implement SDL, GDI etc in separate subclasses?
class sim_window_t {
public:
	sim_window_t(scr_size window_size, bool _fullscreen);

	void run();
	void render_iteration();
	void draw_frame();

private:
	int width;
	int height;
	bool fullscreen;

	SDL_Window *window = nullptr;

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
	VkDescriptorSetLayout descriptorSetLayout;
	VkPipelineLayout pipeline_layout;
	VkPipeline graphicsPipeline;

	VkCommandPool command_pool;

	std::vector<VkCommandBuffer> command_buffers;

	std::vector<VkSemaphore> image_available_semaphores;
	std::vector<VkSemaphore> render_finished_semaphores;
	std::vector<VkFence> in_flight_fences;
	std::vector<VkFence> images_in_flight;
	size_t current_frame = 0;

	bool framebuffer_resized = false;

	uint32_t mip_levels;


	void init_window();

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

	void create_graphics_pipeline();



	void create_framebuffers();
	void create_command_pool();
	void create_command_buffers();
	void create_sync_objects();
};

#endif //SIMSYS_S2_VULKAN_H

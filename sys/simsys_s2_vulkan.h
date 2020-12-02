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

	void cleanup();
	void init_window();
	void init_vulkan();
	void create_vulkan_instance();
	void setup_debug_messenger();
	void create_surface();
	void pick_physical_device();
	void create_logical_device();
	void create_swap_chain();
};

#endif //SIMSYS_S2_VULKAN_H

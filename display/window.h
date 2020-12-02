/*
 * This file is part of the Simutrans-Extended project under the Artistic License.
 * (see LICENSE.txt)
 */

#ifndef DISPLAY_WINDOW_H
#define DISPLAY_WINDOW_H

#ifdef ALT_SDL_DIR
#	include "SDL.h"
#	include "SDL_vulkan.h"
#else
#	include <SDL2/SDL.h>
#	include <SDL2/SDL_vulkan.h>
#endif
#include <vulkan/vulkan.h>

#include "../sys/simsys.h"

// This is similar what simsys.h and simgraph.h  are doing currently,
// however this approach attempts to get rid of global variables and
// instead wraps windowing in sim_window_t and rendering in
// sim_renderer_t classes.

class sim_window_t {
public:
	sim_window_t(int _width, int _height, int _fullscreen);

	// TODO: ifdef VULKAN etc
	// this function should be only called by a vulkan renderer
	void create_vulkan_surface(VkInstance instance, VkSurfaceKHR *pSurface);

	resolution get_drawable_size();

	static resolution query_screen_resolution();

	struct sys_event_t get_event(bool const wait);

private:
	// TODO: update on resize
	resolution res;
	bool fullscreen;

	SDL_Window *window = nullptr;

	void init_window();
};

#endif //DISPLAY_WINDOW_H

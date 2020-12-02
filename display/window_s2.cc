#include "window.h"

#ifdef ALT_SDL_DIR
#include "SDL.h"
#include "SDL_vulkan.h"
#else
#include <SDL2/SDL.h>
#include <SDL2/SDL_vulkan.h>
#endif

#include <stdexcept>

#include "../simversion.h"
#include "../simdebug.h"


resolution sim_window_t::query_screen_resolution() {
	SDL_DisplayMode mode;
	SDL_GetCurrentDisplayMode( 0, &mode );
	DBG_MESSAGE("dr_query_screen_resolution(SDL2)", "screen resolution width=%d, height=%d", mode.w, mode.h );
	return resolution {
		.w = mode.w,
		.h = mode.h,
	};
}

static int display_error_message(const char *title, const char *message) {
	return SDL_ShowSimpleMessageBox(SDL_MESSAGEBOX_ERROR, title, message, nullptr);
}

sim_window_t::sim_window_t(int _width, int _height, int _fullscreen)
		: res{.w = _width, .h = _height}, fullscreen(_fullscreen) {

	if (SDL_InitSubSystem(SDL_INIT_VIDEO) != 0) {
		display_error_message("SDL_Init", SDL_GetError());
		throw std::runtime_error(SDL_GetError());
	}

	init_window();
};

void sim_window_t::init_window() {
	Uint32 flags = SDL_WINDOW_ALLOW_HIGHDPI | SDL_WINDOW_VULKAN;

	flags |= fullscreen ? SDL_WINDOW_FULLSCREEN : SDL_WINDOW_RESIZABLE;

	window = SDL_CreateWindow(
			SIM_TITLE,
			SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED,
			res.w, res.h,
			flags );

	if(  window == NULL  ) {
		display_error_message("Could not open the window", SDL_GetError());
		dbg->error("sim_renderer_t::show SDL2_vulkan", "Could not open the window: %s", SDL_GetError() );
	}

	SDL_Vulkan_GetDrawableSize(window, &res.w, &res.h);
}


void sim_window_t::create_vulkan_surface(VkInstance instance, VkSurfaceKHR *pSurface) {
	if (!SDL_Vulkan_CreateSurface(window, instance, pSurface)) {
		throw std::runtime_error("failed to create window surface");
	}
}

resolution sim_window_t::get_drawable_size() {
	return res;
}
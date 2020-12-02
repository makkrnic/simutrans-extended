/*
 * This file is part of the Simutrans-Extended project under the Artistic License.
 * (see LICENSE.txt)
 */

#ifndef SYS_SIMSYS_S2_VULKAN_H
#define SYS_SIMSYS_S2_VULKAN_H


#ifdef ALT_SDL_DIR
#include "SDL.h"
#else
#include <SDL2/SDL.h>
#endif


#include "../display/scr_coord.h"


class sim_window_t {
public:
	sim_window_t(scr_size window_size, bool _fullscreen) : width(window_size.w), height(window_size.h), fullscreen(_fullscreen) {};

	void show();

private:
	int width;
	int height;
	bool fullscreen;

	SDL_Window *window;
};

#endif //SIMSYS_S2_VULKAN_H

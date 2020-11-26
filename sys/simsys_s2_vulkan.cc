/*
 * This file is part of the Simutrans-Extended project under the Artistic License.
 * (see LICENSE.txt)
 */


#include "simsys.h"
#include "simsys_s2_vulkan.h"

#include "../simversion.h"
#include "../simdebug.h"

#include <sys/time.h>
#include <csignal>

int display_error_message(const char *title, const char *message) {
	return SDL_ShowSimpleMessageBox(SDL_MESSAGEBOX_ERROR, title, message, nullptr);
}

void sim_window_t::show() {
	if (SDL_InitSubSystem(SDL_INIT_VIDEO) != 0) {
		display_error_message("SDL_Init", SDL_GetError());
		throw std::runtime_error(SDL_GetError());
	}

	Uint32 flags = SDL_WINDOW_ALLOW_HIGHDPI | SDL_WINDOW_VULKAN;

	flags |= fullscreen ? SDL_WINDOW_FULLSCREEN : SDL_WINDOW_RESIZABLE;

	window = SDL_CreateWindow(
		SIM_TITLE,
		SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED,
		width, height,
		flags );

	if(  window == NULL  ) {
		display_error_message("Could not open the window", SDL_GetError());
		dbg->error("sim_window_t::show SDL2_vulkan", "Could not open the window: %s", SDL_GetError() );
	}
}

bool dr_auto_scale(bool)
{
	return false;
}

bool dr_os_init(const int*)
{
	return true;
}

resolution dr_query_screen_resolution()
{
	resolution const res = { 0, 0 };
	return res;
}



void dr_os_close()
{
}

// resizes screen
int dr_textur_resize(unsigned short** const textur, int, int)
{
	*textur = NULL;
	return 1;
}


unsigned short *dr_textur_init()
{
	return NULL;
}

unsigned int get_system_color(unsigned int, unsigned int, unsigned int)
{
	return 1;
}

void dr_prepare_flush()
{
}

void dr_flush()
{
}

void dr_textur(int, int, int, int)
{
}

void move_pointer(int, int)
{
}

void set_pointer(int)
{
}

int dr_screenshot(const char *,int,int,int,int)
{
	return -1;
}

void GetEvents()
{
}


void GetEventsNoWait()
{
}

void show_pointer(int)
{
}


void ex_ord_update_mx_my()
{
}

#ifndef _MSC_VER
static timeval first;
#endif

uint32 dr_time()
{
#ifndef _MSC_VER
	timeval second;
	gettimeofday(&second,NULL);
	if (first.tv_usec > second.tv_usec) {
		// since those are often unsigned
		second.tv_usec += 1000000;
		second.tv_sec--;
	}

	return (second.tv_sec - first.tv_sec)*1000ul + (second.tv_usec - first.tv_usec)/1000ul;
#else
	return timeGetTime();
#endif
}

void dr_sleep(uint32 msec)
{
/*
	// this would be 100% POSIX but is usually not very accurate ...
	if(  msec>0  ) {
		struct timeval tv;
		tv.sec = 0;
		tv.usec = msec*1000;
		select(0, 0, 0, 0, &tv);
	}
*/
#ifdef _WIN32
	Sleep( msec );
#else
	usleep( msec * 1000u );
#endif
}

void dr_start_textinput()
{
}

void dr_stop_textinput()
{
}

void dr_notify_input_pos(int, int)
{
}

static void posix_sigterm(int)
{
}

int main(int argc, char **argv) {
	signal( SIGTERM, posix_sigterm );
#ifndef _MSC_VER
	gettimeofday(&first,NULL);
#endif
	return sysmain(argc, argv);
}
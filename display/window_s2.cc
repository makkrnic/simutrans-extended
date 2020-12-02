#include "window.h"

#ifdef ALT_SDL_DIR
#include "SDL.h"
#include "SDL_vulkan.h"
#else
#include <SDL2/SDL.h>
#include <SDL2/SDL_vulkan.h>
#endif

#include <stdexcept>
#include <sstream>

#include "../simversion.h"
#include "../simdebug.h"
#include "../simevent.h"
#include "../unicode.h"


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

	atexit( SDL_Quit ); // clean up on exit

	init_window();
};

void sim_window_t::init_window() {
	Uint32 flags = SDL_WINDOW_ALLOW_HIGHDPI | SDL_WINDOW_VULKAN;

	flags |= fullscreen ? SDL_WINDOW_FULLSCREEN : SDL_WINDOW_RESIZABLE;

	window = SDL_CreateWindow(
			SIM_TITLE " Vulkan",
			SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED,
			res.w, res.h,
			flags );

	if(  window == NULL  ) {
		display_error_message("Could not open the window", SDL_GetError());
		dbg->error("sim_renderer_t::show SDL2_vulkan", "Could not open the window: %s", SDL_GetError() );
	}

	update_drawable_size();
}

void sim_window_t::update_drawable_size() {
	SDL_Vulkan_GetDrawableSize(window, &res.w, &res.h);
}

void sim_window_t::update_fps_info(float fps) {
	std::ostringstream title;

	title << SIM_TITLE << ' ' << fps << "fps" << "; " << res.w << 'x' << res.h;

	SDL_SetWindowTitle(window, title.str().c_str());
}


void sim_window_t::create_vulkan_surface(VkInstance instance, VkSurfaceKHR *pSurface) {
	if (!SDL_Vulkan_CreateSurface(window, instance, pSurface)) {
		throw std::runtime_error("failed to create window surface");
	}
}

resolution sim_window_t::get_drawable_size() {
	return res;
}

static inline unsigned int ModifierKeys()
{
	SDL_Keymod mod = SDL_GetModState();

	return
			(mod & KMOD_SHIFT ? 1 : 0)
			| (mod & KMOD_CTRL ? 2 : 0)
#ifdef __APPLE__
		// Treat the Command key as a control key.
		| (mod & KMOD_GUI ? 2 : 0)
#endif
			;
}


static int conv_mouse_buttons(Uint8 const state)
{
	return
			(state & SDL_BUTTON_LMASK ? MOUSE_LEFTBUTTON  : 0) |
			(state & SDL_BUTTON_MMASK ? MOUSE_MIDBUTTON   : 0) |
			(state & SDL_BUTTON_RMASK ? MOUSE_RIGHTBUTTON : 0);
}

//
#define SCREEN_TO_TEX_X(x) (x)
#define SCREEN_TO_TEX_Y(y) (y)

struct sys_event_t sim_window_t::get_event(bool const wait)
{
	// Apparently Cocoa SDL posts key events that meant to be used by IM...
	// Ignoring SDL_KEYDOWN during preedit seems to work fine.
	static bool composition_is_underway = false;
	static bool ignore_previous_number = false;

	struct sys_event_t sys_event = {
			.type = SIM_NOEVENT,
			.code = 0,
	};

	SDL_Event event;
	event.type = 1;
	if(  wait  ) {
		int n;
		do {
			SDL_WaitEvent( &event );
			n = SDL_PollEvent( NULL );
		} while(  n != 0  &&  event.type == SDL_MOUSEMOTION  );
	}
	else {
		int n;
		bool got_one = false;
		do {
			n = SDL_PollEvent( &event );
			if(  n != 0  ) {
				got_one = true;
				if(  event.type == SDL_MOUSEMOTION  ) {
					// sys_event.mx   = SCREEN_TO_TEX_X(event.motion.x);
					// sys_event.my   = SCREEN_TO_TEX_Y(event.motion.y);
					sys_event.type = SIM_MOUSE_MOVE;
					sys_event.code = SIM_MOUSE_MOVED;
					sys_event.mb   = conv_mouse_buttons( event.motion.state );
				}
			}
		} while(  n != 0  &&  event.type == SDL_MOUSEMOTION  );
		if(  !got_one  ) {
			return sys_event;
		}
	}

	static char textinput[SDL_TEXTINPUTEVENT_TEXT_SIZE];
	switch(  event.type  ) {
		case SDL_WINDOWEVENT: {
			if(  event.window.event == SDL_WINDOWEVENT_SIZE_CHANGED  ) {
				sys_event.new_window_size.w = SCREEN_TO_TEX_X(event.window.data1);
				sys_event.new_window_size.h = SCREEN_TO_TEX_Y(event.window.data2);
				sys_event.type = SIM_SYSTEM;
				sys_event.code = SYSTEM_RESIZE;

				update_drawable_size();
			}
			// Ignore other window events.
			break;
		}
		case SDL_MOUSEBUTTONDOWN: {
			sys_event.type    = SIM_MOUSE_BUTTONS;
			switch(  event.button.button  ) {
				case SDL_BUTTON_LEFT:   sys_event.code = SIM_MOUSE_LEFTBUTTON;  break;
				case SDL_BUTTON_MIDDLE: sys_event.code = SIM_MOUSE_MIDBUTTON;   break;
				case SDL_BUTTON_RIGHT:  sys_event.code = SIM_MOUSE_RIGHTBUTTON; break;
				case SDL_BUTTON_X1:     sys_event.code = SIM_MOUSE_WHEELUP;     break;
				case SDL_BUTTON_X2:     sys_event.code = SIM_MOUSE_WHEELDOWN;   break;
			}
			sys_event.mx      = SCREEN_TO_TEX_X(event.button.x);
			sys_event.my      = SCREEN_TO_TEX_Y(event.button.y);
			sys_event.mb      = conv_mouse_buttons( SDL_GetMouseState(0, 0) );
			sys_event.key_mod = ModifierKeys();
			break;
		}
		case SDL_MOUSEBUTTONUP: {
			sys_event.type    = SIM_MOUSE_BUTTONS;
			switch(  event.button.button  ) {
				case SDL_BUTTON_LEFT:   sys_event.code = SIM_MOUSE_LEFTUP;  break;
				case SDL_BUTTON_MIDDLE: sys_event.code = SIM_MOUSE_MIDUP;   break;
				case SDL_BUTTON_RIGHT:  sys_event.code = SIM_MOUSE_RIGHTUP; break;
			}
			sys_event.mx      = SCREEN_TO_TEX_X(event.button.x);
			sys_event.my      = SCREEN_TO_TEX_Y(event.button.y);
			sys_event.mb      = conv_mouse_buttons( SDL_GetMouseState(0, 0) );
			sys_event.key_mod = ModifierKeys();
			break;
		}
		case SDL_MOUSEWHEEL: {
			sys_event.type    = SIM_MOUSE_BUTTONS;
			sys_event.code    = event.wheel.y > 0 ? SIM_MOUSE_WHEELUP : SIM_MOUSE_WHEELDOWN;
			SDL_GetMouseState(&sys_event.mx, &sys_event.my);
			sys_event.key_mod = ModifierKeys();
			break;
		}
		case SDL_KEYDOWN: {
			// Hack: when 2 byte character composition is under way, we have to leave the key processing with the IME
			// BUT: if not, we have to do it ourselves, or the cursor or return will not be recognised
			// TODO MAK
			// if(  composition_is_underway  ) {
			// 	if(  gui_component_t *c = win_get_focus()  ) {
			// 		if(  gui_textinput_t *tinp = dynamic_cast<gui_textinput_t *>(c)  ) {
			// 			if(  tinp->get_composition()[0]  ) {
			// 				// pending string, handled by IME
			// 				break;
			// 			}
			// 		}
			// 	}
			// }

			unsigned long code;
 #ifdef _WIN32
			// SDL doesn't set numlock state correctly on startup. Revert to win32 function as workaround.
			const bool key_numlock = ((GetKeyState( VK_NUMLOCK ) & 1) != 0);
 #else
			const bool key_numlock = (SDL_GetModState() & KMOD_NUM);
 #endif
			const bool numlock = key_numlock; // &&  !env_t::numpad_always_moves_map;
			sys_event.key_mod = ModifierKeys();
			SDL_Keycode sym = event.key.keysym.sym;
			bool np = false; // to indicate we converted a numpad key

			switch(  sym  ) {
				case SDLK_BACKSPACE:  code = SIM_KEY_BACKSPACE;             break;
				case SDLK_TAB:        code = SIM_KEY_TAB;                   break;
				case SDLK_RETURN:     code = SIM_KEY_ENTER;                 break;
				case SDLK_ESCAPE:     code = SIM_KEY_ESCAPE;                break;
				case SDLK_DELETE:     code = SIM_KEY_DELETE;                break;
				case SDLK_DOWN:       code = SIM_KEY_DOWN;                  break;
				case SDLK_END:        code = SIM_KEY_END;                   break;
				case SDLK_HOME:       code = SIM_KEY_HOME;                  break;
				case SDLK_F1:         code = SIM_KEY_F1;                    break;
				case SDLK_F2:         code = SIM_KEY_F2;                    break;
				case SDLK_F3:         code = SIM_KEY_F3;                    break;
				case SDLK_F4:         code = SIM_KEY_F4;                    break;
				case SDLK_F5:         code = SIM_KEY_F5;                    break;
				case SDLK_F6:         code = SIM_KEY_F6;                    break;
				case SDLK_F7:         code = SIM_KEY_F7;                    break;
				case SDLK_F8:         code = SIM_KEY_F8;                    break;
				case SDLK_F9:         code = SIM_KEY_F9;                    break;
				case SDLK_F10:        code = SIM_KEY_F10;                   break;
				case SDLK_F11:        code = SIM_KEY_F11;                   break;
				case SDLK_F12:        code = SIM_KEY_F12;                   break;
				case SDLK_F13:        code = SIM_KEY_F13;                   break;
				case SDLK_F14:        code = SIM_KEY_F14;                   break;
				case SDLK_F15:        code = SIM_KEY_F15;                   break;
				case SDLK_KP_0:       SIM_KEY_NUMPAD_BASE+0;                break;
				case SDLK_KP_1:       SIM_KEY_NUMPAD_BASE+1;                break;
				case SDLK_KP_2:       SIM_KEY_NUMPAD_BASE+2;                break;
				case SDLK_KP_3:       SIM_KEY_NUMPAD_BASE+3;                break;
				case SDLK_KP_4:       SIM_KEY_NUMPAD_BASE+4;                break;
				case SDLK_KP_5:       SIM_KEY_NUMPAD_BASE+5;                break;
				case SDLK_KP_6:       SIM_KEY_NUMPAD_BASE+6;                break;
				case SDLK_KP_7:       SIM_KEY_NUMPAD_BASE+7;                break;
				case SDLK_KP_8:       SIM_KEY_NUMPAD_BASE+8;                break;
				case SDLK_KP_9:       SIM_KEY_NUMPAD_BASE+9;                break;
				case SDLK_KP_ENTER:   code = SIM_KEY_ENTER;                 break;
				case SDLK_LEFT:       code = SIM_KEY_LEFT;                  break;
				case SDLK_PAGEDOWN:   code = '<';                           break;
				case SDLK_PAGEUP:     code = '>';                           break;
				case SDLK_RIGHT:      code = SIM_KEY_RIGHT;                 break;
				case SDLK_UP:         code = SIM_KEY_UP;                    break;
				case SDLK_PAUSE:      code = SIM_KEY_PAUSE;                 break;
				default: {
					// Handle CTRL-keys. SDL_TEXTINPUT event handles regular input
					if(  (sys_event.key_mod & 2)  &&  SDLK_a <= sym  &&  sym <= SDLK_z  ) {
						code = event.key.keysym.sym & 31;
					}
					else {
						code = 0;
					}
					break;
				}
			}
			ignore_previous_number = (np  &&   key_numlock);
			sys_event.type    = SIM_KEYBOARD;
			sys_event.code    = code;
			break;
		}

		case SDL_TEXTINPUT: {
			size_t in_pos = 0;
			utf32 uc = utf8_decoder_t::decode((utf8 const*)event.text.text, in_pos);
			if(  event.text.text[in_pos]==0  ) {
				// single character
				// TODO: Handle numpad state outside general event handling
				if( ignore_previous_number ) {
					ignore_previous_number = false;
					break;
				}
				sys_event.type    = SIM_KEYBOARD;
				sys_event.code    = (unsigned long)uc;
			}
			else {
				// string
				strcpy( textinput, event.text.text );
				sys_event.type    = SIM_STRING;
				sys_event.ptr     = (void*)textinput;
			}
			sys_event.key_mod = ModifierKeys();
			composition_is_underway = false;
			break;
		}
 #ifdef USE_SDL_TEXTEDITING
			case SDL_TEXTEDITING: {
			//printf( "SDL_TEXTEDITING {timestamp=%d, \"%s\", start=%d, length=%d}\n", event.edit.timestamp, event.edit.text, event.edit.start, event.edit.length );
			strcpy( textinput, event.edit.text );
			if(  !textinput[0]  ) {
				composition_is_underway = false;
			}
			int i = 0;
			int start = 0;
			for(  ; i<event.edit.start; ++i  ) {
				start = utf8_get_next_char( (utf8 *)event.edit.text, start );
			}
			int end = start;
			for(  ; i<event.edit.start+event.edit.length; ++i  ) {
				end = utf8_get_next_char( (utf8*)event.edit.text, end );
			}

			if(  gui_component_t *c = win_get_focus()  ) {
				if(  gui_textinput_t *tinp = dynamic_cast<gui_textinput_t *>(c)  ) {
					tinp->set_composition_status( textinput, start, end-start );
				}
			}
			composition_is_underway = true;
			break;
		}
 #endif
		case SDL_MOUSEMOTION: {
			sys_event.type    = SIM_MOUSE_MOVE;
			sys_event.code    = SIM_MOUSE_MOVED;
			sys_event.mx      = SCREEN_TO_TEX_X(event.motion.x);
			sys_event.my      = SCREEN_TO_TEX_Y(event.motion.y);
			sys_event.mb      = conv_mouse_buttons( event.motion.state );
			sys_event.key_mod = ModifierKeys();
			break;
		}
		case SDL_KEYUP: {
			sys_event.type = SIM_KEYBOARD;
			sys_event.code = 0;
			break;
		}
		case SDL_QUIT: {
			sys_event.type = SIM_SYSTEM;
			sys_event.code = SYSTEM_QUIT;
			break;
		}
		default: {
			sys_event.type = SIM_IGNORE_EVENT;
			sys_event.code = 0;
			break;
		}
	}

	return sys_event;
}

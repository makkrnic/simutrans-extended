/*
 * This file is part of the Simutrans-Extended project under the Artistic License.
 * (see LICENSE.txt)
 */

#include "../sys/simsys_s2_vulkan.h"
#include "../simconst.h"
#include "../sys/simsys.h"
#include "../descriptor/image.h"

#include "simgraph.h"

void simgraph_init(scr_size window_size, bool fullscreen)
{
}


// TODO MAK remove after implementing vulkan rendering
int default_font_ascent = 0;
int default_font_linespace = 0;
uint8 display_day_lights[LIGHT_COUNT*3] = {
		0x57,	0x65,	0x6F, // Dark windows, lit yellowish at night
		0x7F,	0x9B,	0xF1, // Lighter windows, lit blueish at night
		0xFF,	0xFF,	0x53, // Yellow light
		0xFF,	0x21,	0x1D, // Red light
		0x01,	0xDD,	0x01, // Green light
		0x6B,	0x6B,	0x6B, // Non-darkening grey 1 (menus)
		0x9B,	0x9B,	0x9B, // Non-darkening grey 2 (menus)
		0xB3,	0xB3,	0xB3, // non-darkening grey 3 (menus)
		0xC9,	0xC9,	0xC9, // Non-darkening grey 4 (menus)
		0xDF,	0xDF,	0xDF, // Non-darkening grey 5 (menus)
		0xE3,	0xE3,	0xFF, // Nearly white light at day, yellowish light at night
		0xC1,	0xB1,	0xD1, // Windows, lit yellow
		0x4D,	0x4D,	0x4D, // Windows, lit yellow
		0xE1,	0x00,	0xE1, // purple light for signals
		0x01,	0x01,	0xFF, // blue light
};
uint8 display_night_lights[LIGHT_COUNT*3] = {
		0xD3,	0xC3,	0x80, // Dark windows, lit yellowish at night
		0x80,	0xC3,	0xD3, // Lighter windows, lit blueish at night
		0xFF,	0xFF,	0x53, // Yellow light
		0xFF,	0x21,	0x1D, // Red light
		0x01,	0xDD,	0x01, // Green light
		0x6B,	0x6B,	0x6B, // Non-darkening grey 1 (menus)
		0x9B,	0x9B,	0x9B, // Non-darkening grey 2 (menus)
		0xB3,	0xB3,	0xB3, // non-darkening grey 3 (menus)
		0xC9,	0xC9,	0xC9, // Non-darkening grey 4 (menus)
		0xDF,	0xDF,	0xDF, // Non-darkening grey 5 (menus)
		0xFF,	0xFF,	0xE3, // Nearly white light at day, yellowish light at night
		0xD3,	0xC3,	0x80, // Windows, lit yellow
		0xD3,	0xC3,	0x80, // Windows, lit yellow
		0xE1,	0x00,	0xE1, // purple light for signals
		0x01,	0x01,	0xFF, // blue light
};

KOORD_VAL tile_raster_width = 16; // zoomed
KOORD_VAL base_tile_raster_width = 16; // original


PIXVAL color_idx_to_rgb(PIXVAL idx)
{
	return idx;
}

PIXVAL color_rgb_to_idx(PIXVAL color)
{
	return color;
}


uint32 get_color_rgb(uint8)
{
	return 0;
}

void env_t_rgb_to_system_colors()
{
}

KOORD_VAL display_set_base_raster_width(KOORD_VAL)
{
	return 0;
}

void set_zoom_factor(int)
{
}

int get_zoom_factor()
{
	return 1;
}

int zoom_factor_up()
{
	return false;
}

int zoom_factor_down()
{
	return false;
}

void mark_rect_dirty_wc(KOORD_VAL, KOORD_VAL, KOORD_VAL, KOORD_VAL)
{
}

void mark_rect_dirty_clip(KOORD_VAL, KOORD_VAL, KOORD_VAL, KOORD_VAL  CLIP_NUM_DEF_NOUSE)
{
}

void mark_screen_dirty()
{
}

void display_mark_img_dirty(image_id, KOORD_VAL, KOORD_VAL)
{
}

bool display_load_font(const char*, bool)
{
	return true;
}

sint16 display_get_width()
{
	return 0;
}

sint16 display_get_height()
{
	return 0;
}

void display_set_height(KOORD_VAL)
{
}

void display_set_actual_width(KOORD_VAL)
{
}

void display_day_night_shift(int)
{
}

void display_set_player_color_scheme(const int, const uint8, const uint8)
{
}

void register_image(image_t* image)
{
	image->imageid = 1;
}

void display_snapshot(int, int, int, int)
{
}

void display_get_image_offset(image_id image, KOORD_VAL *xoff, KOORD_VAL *yoff, KOORD_VAL *xw, KOORD_VAL *yw)
{
	if (image < 2) {
		// initialize offsets with dummy values
		*xoff = 0;
		*yoff = 0;
		*xw = 0;
		*yw = 0;
	}
}

void display_get_base_image_offset(image_id image, scr_coord_val *xoff, scr_coord_val *yoff, scr_coord_val *xw, scr_coord_val *yw)
{
    if (image < 2) {
        // initialize offsets with dummy values
        xoff = 0;
        yoff = 0;
        xw = 0;
        yw = 0;
    }
}

clip_dimension display_get_clip_wh(CLIP_NUM_DEF_NOUSE0)
{
	clip_dimension clip_rect;
	clip_rect.x = 0;
	clip_rect.xx = 0;
	clip_rect.w = 0;
	clip_rect.y = 0;
	clip_rect.yy = 0;
	clip_rect.h = 0;
	return clip_rect;
}

void display_set_clip_wh(KOORD_VAL, KOORD_VAL, KOORD_VAL, KOORD_VAL  CLIP_NUM_DEF_NOUSE, bool)
{
}

void display_push_clip_wh(KOORD_VAL, KOORD_VAL, KOORD_VAL, KOORD_VAL  CLIP_NUM_DEF_NOUSE)
{
}

void display_swap_clip_wh(CLIP_NUM_DEF_NOUSE0)
{
}

void display_pop_clip_wh(CLIP_NUM_DEF_NOUSE0)
{
}

void display_scroll_band(const KOORD_VAL, const KOORD_VAL, const KOORD_VAL)
{
}

void display_img_aux(const image_id, KOORD_VAL, KOORD_VAL, const signed char, const int, const int  CLIP_NUM_DEF_NOUSE)
{
}

void display_color_img(const image_id, KOORD_VAL, KOORD_VAL, const signed char, const int, const int  CLIP_NUM_DEF_NOUSE)
{
}

void display_base_img(const image_id, KOORD_VAL, KOORD_VAL, const signed char, const int, const int  CLIP_NUM_DEF_NOUSE)
{
}

void display_fit_img_to_width(const image_id, sint16)
{
}

void display_img_stretch(const stretch_map_t &, scr_rect)
{
}

void display_img_stretch_blend(const stretch_map_t &, scr_rect, FLAGGED_PIXVAL)
{
}

void display_rezoomed_img_blend(const image_id, KOORD_VAL, KOORD_VAL, const signed char, const FLAGGED_PIXVAL, const int, const int  CLIP_NUM_DEF_NOUSE)
{
}

void display_rezoomed_img_alpha(const image_id, const image_id, const unsigned, KOORD_VAL, KOORD_VAL, const signed char, const FLAGGED_PIXVAL, const int, const int  CLIP_NUM_DEF_NOUSE)
{
}

void display_base_img_blend(const image_id, KOORD_VAL, KOORD_VAL, const signed char, const FLAGGED_PIXVAL, const int, const int  CLIP_NUM_DEF_NOUSE)
{
}

void display_base_img_alpha(const image_id, const image_id, const unsigned, KOORD_VAL, KOORD_VAL, const signed char, const FLAGGED_PIXVAL, const int, int  CLIP_NUM_DEF_NOUSE)
{
}

// variables for storing currently used image procedure set and tile raster width
display_image_proc display_normal = display_base_img;
display_image_proc display_color = display_base_img;
display_blend_proc display_blend = display_base_img_blend;
display_alpha_proc display_alpha = display_base_img_alpha;

signed short current_tile_raster_width = 0;

PIXVAL display_blend_colors(PIXVAL, PIXVAL, int)
{
	return 0;
}

void display_blend_wh_rgb(KOORD_VAL, KOORD_VAL, KOORD_VAL, KOORD_VAL, PIXVAL, int )
{
}

void display_vlinear_gradient_wh_rgb(KOORD_VAL, KOORD_VAL, KOORD_VAL, KOORD_VAL, PIXVAL, int, int)
{
}

void display_color_img_with_tooltip(const image_id, KOORD_VAL, KOORD_VAL, sint8, const int, const int, const char* CLIP_NUM_DEF_NOUSE)
{
}

void display_fillbox_wh_rgb(KOORD_VAL, KOORD_VAL, KOORD_VAL, KOORD_VAL, PIXVAL, bool)
{
}


void display_fillbox_wh_clip_rgb(KOORD_VAL, KOORD_VAL, KOORD_VAL, KOORD_VAL, PIXVAL, bool  CLIP_NUM_DEF_NOUSE)
{
}

void display_cylinderbar_wh_clip_rgb(KOORD_VAL, KOORD_VAL, KOORD_VAL, KOORD_VAL, PIXVAL, bool  CLIP_NUM_DEF_NOUSE)
{
}

void display_colorbox_with_tooltip(KOORD_VAL, KOORD_VAL, KOORD_VAL, KOORD_VAL, PIXVAL, bool, const char*)
{
}

void display_veh_form_wh_clip_rgb(KOORD_VAL, KOORD_VAL, KOORD_VAL, PIXVAL, bool, uint8, uint8, bool CLIP_NUM_DEF_NOUSE)
{
}

void display_vline_wh_rgb(KOORD_VAL, KOORD_VAL, KOORD_VAL, PIXVAL, bool)
{
}

void display_vline_wh_clip_rgb(KOORD_VAL, KOORD_VAL, KOORD_VAL, PIXVAL, bool  CLIP_NUM_DEF_NOUSE)
{
}

void display_array_wh(KOORD_VAL, KOORD_VAL, KOORD_VAL, KOORD_VAL, const PIXVAL *)
{
}

void display_filled_roundbox_clip(KOORD_VAL, KOORD_VAL, KOORD_VAL, KOORD_VAL, PIXVAL, bool)
{
}

size_t get_next_char(const char*, size_t pos)
{
	return pos + 1;
}

sint32 get_prev_char(const char*, sint32 pos)
{
	if (pos <= 0) {
		return 0;
	}
	return pos - 1;
}

KOORD_VAL display_get_char_width(utf32)
{
	return 0;
}

KOORD_VAL display_get_char_max_width(const char*, size_t)
{
	return 0;
}

utf32 get_next_char_with_metrics(const char* &, unsigned char &, unsigned char &)
{
	return 0;
}

utf32 get_prev_char_with_metrics(const char* &, const char *const, unsigned char &, unsigned char &)
{
	return 0;
}

bool has_character(utf16)
{
	return false;
}

size_t display_fit_proportional(const char *, scr_coord_val, scr_coord_val)
{
	return 0;
}

int display_calc_proportional_string_len_width(const char*, size_t)
{
	return 0;
}


void display_calc_proportional_multiline_string_len_width( int &xw, int &yh, const char *, size_t )
{
	xw = yh = 0;;
}


int display_text_proportional_len_clip_rgb(KOORD_VAL, KOORD_VAL, const char*, control_alignment_t , const PIXVAL, bool, sint32  CLIP_NUM_DEF_NOUSE)
{
	return 0;
}

void display_outline_proportional_rgb(KOORD_VAL, KOORD_VAL, PIXVAL, PIXVAL, const char *, int, sint32)
{
}

void display_shadow_proportional_rgb(KOORD_VAL, KOORD_VAL, PIXVAL, PIXVAL, const char *, int, sint32)
{
}

void display_ddd_box_rgb(KOORD_VAL, KOORD_VAL, KOORD_VAL, KOORD_VAL, PIXVAL, PIXVAL, bool)
{
}

void display_ddd_box_clip_rgb(KOORD_VAL, KOORD_VAL, KOORD_VAL, KOORD_VAL, PIXVAL, PIXVAL)
{
}

void display_ddd_proportional(KOORD_VAL, KOORD_VAL, KOORD_VAL, KOORD_VAL, FLAGGED_PIXVAL, FLAGGED_PIXVAL, const char *, int)
{
}

void display_ddd_proportional_clip(KOORD_VAL, KOORD_VAL, KOORD_VAL, KOORD_VAL, FLAGGED_PIXVAL, FLAGGED_PIXVAL, const char *, int  CLIP_NUM_DEF_NOUSE)
{
}

int display_multiline_text_rgb(KOORD_VAL, KOORD_VAL, const char *, PIXVAL)
{
	return 0;
}

void display_flush_buffer()
{
}

void display_show_pointer(int)
{
}

void display_set_pointer(int)
{
}

void display_show_load_pointer(int)
{
}

bool is_display_init()
{
	return false;
}

void display_free_all_images_above(image_id)
{
}

void simgraph_exit()
{
	dr_os_close();
}

void simgraph_resize(scr_size)
{
}

void display_direct_line_rgb(const KOORD_VAL, const KOORD_VAL, const KOORD_VAL, const KOORD_VAL, const PIXVAL)
{
}

void display_direct_line_dotted_rgb(const KOORD_VAL, const KOORD_VAL, const KOORD_VAL, const KOORD_VAL, const KOORD_VAL, const KOORD_VAL, const PIXVAL)
{
}

void display_circle_rgb(KOORD_VAL, KOORD_VAL, int, const PIXVAL)
{
}

void display_filled_circle_rgb(KOORD_VAL, KOORD_VAL, int, const PIXVAL)
{
}

void display_right_triangle_rgb(KOORD_VAL, KOORD_VAL, uint8, const PIXVAL, const bool)
{
}

int display_fluctuation_triangle_rgb(KOORD_VAL, KOORD_VAL, uint8, const bool, sint64)
{
	return 0;
}

void draw_bezier_rgb(KOORD_VAL, KOORD_VAL, KOORD_VAL, KOORD_VAL, KOORD_VAL, KOORD_VAL, KOORD_VAL, KOORD_VAL, const PIXVAL, KOORD_VAL, KOORD_VAL)
{
}

void display_img_aligned(const image_id, scr_rect, int, int)
{
}

KOORD_VAL display_proportional_ellipsis_rgb( scr_rect, const char *, int, PIXVAL, bool, bool, PIXVAL)
{
	return 0;
}

image_id get_image_count()
{
	return 0;
}

#ifdef MULTI_THREAD
void add_poly_clip(int, int, int, int, int  CLIP_NUM_DEF_NOUSE)
{
}

void clear_all_poly_clip(const sint8)
{
}

void activate_ribi_clip(int  CLIP_NUM_DEF_NOUSE)
{
}
#else
void add_poly_clip(int, int, int, int, int)
{
}

void clear_all_poly_clip()
{
}

void activate_ribi_clip(int)
{
}
#endif
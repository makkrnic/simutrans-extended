/*
 * This file is part of the Simutrans-Extended project under the Artistic License.
 * (see LICENSE.txt)
 */

#include "gui_label.h"
#include "../gui_frame.h"
#include "../../dataobj/translator.h"
#include "../../utils/simstring.h"
#include "../simwin.h"

#include "../../display/simgraph.h"

/*
 * just displays a text, will be auto-translated
 */

static scr_coord_val separator_width = 0;
static scr_coord_val large_money_width = 0;

gui_label_t::gui_label_t(const char* text, PIXVAL color_, align_t align_) :
	align(align_), tooltip(NULL)
{
	separator_width = proportional_string_width( ",00$" );

	if (get_large_money_string()) {
		cbuffer_t buf;
		buf.printf("%s$", get_large_money_string());
		large_money_width = proportional_string_width((const char*) buf);
	}
	else {
		large_money_width = 0;
	}

	set_size( scr_size( D_BUTTON_WIDTH, D_LABEL_HEIGHT ) );
	init( text, scr_coord (0,0), color_, align_);
	shadowed = false;
	min_size = scr_size(0, 0);
}


void gui_label_t::set_min_size(scr_size s)
{
	min_size = s;
}


void gui_label_t::set_fixed_width(const scr_coord_val width)
{
	set_min_size(scr_size(width, size.h));
	fixed_width = width;
}


scr_size gui_label_t::get_min_size() const
{
	return scr_size( max(min_size.w, text ? display_calc_proportional_string_len_width(text,strlen(text)) : D_BUTTON_WIDTH), D_LABEL_HEIGHT );
}

scr_size gui_label_t::get_max_size() const
{
	if (fixed_width) {
		return get_min_size();
	}
	return align == left  ? scr_size(max(get_min_size().w, size.w), get_min_size().h) : scr_size(scr_size::inf.w, get_min_size().h);
}


void gui_label_t::set_text(const char *text, bool autosize)
{
	if (text != NULL) {
		set_text_pointer(translator::translate(text), autosize);
	}
	else {
		set_text_pointer(NULL, false);
	}
}


void gui_label_t::set_text_pointer(const char *text_par, bool autosize)
{
	text = text_par;

	if (autosize && text && *text != '\0') {
		set_size( scr_size( display_calc_proportional_string_len_width(text,strlen(text)),size.h ) );
	}
}


void gui_label_t::draw(scr_coord offset)
{
	if(  align == money_right) {
		if(text) {
			const char *separator = NULL;
			const bool not_a_number = atol(text)==0  &&  !isdigit(*text);

			scr_coord right = pos + offset;

			if(  !not_a_number  ) {
				// find first letter of large_money_width in text
				if (get_large_money_string()!=NULL) {
					separator = strrchr(text, *(get_large_money_string()) );
					if (separator) {
						right.x += get_size().w - large_money_width;
					}
				}
				// look for fraction_sep (e.g., comma)
				if (separator==NULL) {
					// everything else align at decimal separator
					right.x += get_size().w - separator_width;
					separator = strrchr(text, get_fraction_sep());
				}
			}

			if(separator) {
				display_proportional_clip_rgb(right.x, right.y, separator, ALIGN_LEFT, color, true);
				if(  separator!=text  ) {
					display_text_proportional_len_clip_rgb(right.x, right.y, text, ALIGN_RIGHT | DT_CLIP, color, true, separator-text );
				}
			}
			else {
				// integer or normal text
				display_proportional_clip_rgb(right.x, right.y, text, ALIGN_RIGHT, color, true);
			}
		}
	}

	else if(text) {
		const scr_rect area( offset+pos, size );
		int a = align == left ? ALIGN_LEFT : ( align == right ? ALIGN_RIGHT : ALIGN_CENTER_H);
		display_proportional_ellipsis_rgb( area, text,  a | DT_CLIP, color, true, shadowed, color_shadow );
	}

	if ( tooltip  &&  getroffen(get_mouse_x()-offset.x, get_mouse_y()-offset.y) ) {
		const scr_coord_val by = offset.y + pos.y;
		const scr_coord_val bh = size.h;

		win_set_tooltip(get_mouse_x() + TOOLTIP_MOUSE_OFFSET_X, by + bh + TOOLTIP_MOUSE_OFFSET_Y, tooltip, this);
	}
}

void gui_label_t::set_tooltip(const char * t)
{
	tooltip = t;
}


void gui_label_buf_t::init(PIXVAL color_par, align_t align_par)
{
	gui_label_t::init(NULL, get_pos(), color_par, align_par);
	buf_changed = false;
}


void gui_label_buf_t::update()
{
	buffer_read = buffer_write;
	buffer_write.clear();
	gui_label_t::set_text_pointer( (const char*)buffer_read, false /*no autoresize*/ );
	buf_changed = false;
}


void gui_label_buf_t::draw(scr_coord offset)
{
	if (buf_changed) {
		update();
	}
	gui_label_t::draw(offset);
}


void gui_label_buf_t::append_money(double money)
{
	buffer_write.append_money(money);
	set_color(money >= 0 ? MONEY_PLUS : MONEY_MINUS);
}


void gui_label_updown_t::draw(scr_coord offset)
{
	cbuffer_t text;
	text.clear();
	if (!show_border_value && value == border) {
		text.append("-");
	}
	else{
		text.append(abs(value), align == money_right ? 2 : 0);
	}

	if (align == money_right) {
		const char *separator = NULL;

		// position of separator
		scr_coord right = pos + offset;
		if (align == money_right) {
			right.x += get_size().w - separator_width;
		}

		separator = strrchr(text, get_fraction_sep());
		if (separator == NULL && get_large_money_string() != NULL) {
			separator = strrchr(text, *(get_large_money_string()));
		}

		if (separator) {
			display_proportional_clip_rgb(right.x, right.y, separator, ALIGN_LEFT, color, true);
			if (separator != text) {
				display_text_proportional_len_clip_rgb(right.x, right.y, text, ALIGN_RIGHT | DT_CLIP, color, true, separator - text);
			}
		}
		else {
			// integer numbers without decimals, align at decimal separator
			display_proportional_clip_rgb(right.x, right.y, text, ALIGN_RIGHT, color, true);
		}

		display_fluctuation_triangle_rgb(right.x - LINEASCENT + 3, right.y + LINESPACE/6, LINEASCENT*2/3, true, value);
	}
	else {
		display_fluctuation_triangle_rgb(pos.x + offset.x, pos.y + offset.y + LINESPACE/6, LINEASCENT*2/3, true, value);
		const scr_rect area(offset + pos + scr_coord(LINEASCENT,0), size);
		int a = align == left ? ALIGN_LEFT : (align == right ? ALIGN_RIGHT : ALIGN_CENTER_H);
		display_proportional_ellipsis_rgb(area, text, a | DT_CLIP, color, true);
	}

	if (tooltip  &&  getroffen(get_mouse_x() - offset.x, get_mouse_y() - offset.y)) {
		const scr_coord_val by = offset.y + pos.y;
		const scr_coord_val bh = size.h;

		win_set_tooltip(get_mouse_x() + TOOLTIP_MOUSE_OFFSET_X, by + bh + TOOLTIP_MOUSE_OFFSET_Y, tooltip, this);
	}
}

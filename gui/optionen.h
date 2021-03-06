/*
 * This file is part of the Simutrans-Extended project under the Artistic License.
 * (see LICENSE.txt)
 */

#ifndef GUI_OPTIONEN_H
#define GUI_OPTIONEN_H


/**
 * Settings in the game
 * @author Hj. Malthaner
 */

class gui_frame_t;
class action_listener_t;
class gui_divider_t;
class button_t;
class gui_action_creator_t;

/*
 * Dialog for game options/Main menu
 * Niels Roest, Hj. Malthaner, 2000
 */
class optionen_gui_t : public gui_frame_t, action_listener_t
{
	private:
		gui_divider_t divider;
		button_t      option_buttons[11];

	public:
		optionen_gui_t();

		/**
		 * Set the window associated helptext
		 * @return the filename for the helptext, or NULL
		 * @author Hj. Malthaner
		 */
		const char * get_help_filename() const OVERRIDE {return "options.txt";}

		bool action_triggered(gui_action_creator_t*, value_t) OVERRIDE;
};

#endif

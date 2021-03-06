/*
 * This file is part of the Simutrans-Extended project under the Artistic License.
 * (see LICENSE.txt)
 */

#ifndef GUI_MESSAGE_FRAME_T_H
#define GUI_MESSAGE_FRAME_T_H


#include "../gui/simwin.h"

#include "gui_frame.h"
#include "components/gui_button.h"
#include "components/gui_scrollpane.h"
#include "components/gui_tab_panel.h"
#include "components/gui_textinput.h"

#include "message_stats_t.h"
#include "components/action_listener.h"



/**
 * All messages since the start of the program
 * @author prissi
 */
class message_frame_t : public gui_frame_t, private action_listener_t
{
private:
	char ibuf[256];
	message_stats_t	stats;
	gui_scrollpane_t scrolly;
	gui_tab_panel_t tabs;		// Knightly : tab panel for filtering messages
	gui_textinput_t input;
	button_t option_bt, send_bt;
	vector_tpl<sint32> tab_categories;

public:
	message_frame_t();

	/**
	 * Set the window associated helptext
	 * @return the filename for the helptext, or NULL
	 * @author Hj. Malthaner
	 */
	const char * get_help_filename() const OVERRIDE {return "mailbox.txt";}

	/**
	* resize window in response to a resize event
	* @author Hj. Malthaner
	*/
	void resize(const scr_coord delta) OVERRIDE;

	bool action_triggered(gui_action_creator_t*, value_t) OVERRIDE;

	void rdwr(loadsave_t *) OVERRIDE;

	uint32 get_rdwr_id() OVERRIDE { return magic_messageframe; }
};

#endif

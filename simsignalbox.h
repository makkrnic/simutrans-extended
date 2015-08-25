 /*
 * This file is part of the Simutrans project under the artistic license.
 * (see license.txt)
 */

#ifndef _simsignalbox_h
#define _simsignalbox_h

#include "obj/gebaeude.h"
#include "tpl/slist_tpl.h"

class karte_t;
class signal_t;
class roadsign_besch_t;

class signalbox_t : public gebaeude_t
{
private:

	slist_tpl<koord3d> signals; // The signals controlled by this signalbox.
	inline signal_t* get_signal_from_location(koord3d k);

protected:

	static slist_tpl<signalbox_t *> all_signalboxes;

public:

	signalbox_t(loadsave_t *file);
	signalbox_t(koord3d pos, player_t *player, const haus_tile_besch_t *t);

	~signalbox_t();

	obj_t::typ get_typ() const { return signalbox; }

	void rdwr(loadsave_t *file);

	void rotate90();

	void add_to_world_list(bool lock = false);

	// Adds a signal to this signalbox. Returns whether this succeeds.
	inline bool add_signal(signal_t* s);

	// Checks whether a specific signal can be added without adding it. Returns true if it succeeds.
	bool can_add_signal(signal_t* s) const;
	bool can_add_signal(const roadsign_besch_t* b) const; 
	
	// Check whether any more signals can be added. Returns true if it succeeds.
	// Separate from the above in order to give better error messages.
	inline bool can_add_more_signals() const;

	// Transfers a signal to this box from another box.
	// Returns true if the transfer succeeds, false if not.
	bool transfer_signal(signal_t* s, signalbox_t* sb);

	inline void remove_signal(signal_t* s);

	// Transfer all signals *from* this box to the specified box.
	// (Intended to be a preparation for closure).
	// Returns number of signals transferred successfully, number that failed (x,y).
	koord transfer_all_signals(signalbox_t* sb); 
};

#endif
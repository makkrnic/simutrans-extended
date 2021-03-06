/*
 * This file is part of the Simutrans-Extended project under the Artistic License.
 * (see LICENSE.txt)
 */

#ifndef DATAOBJ_MARKER_H
#define DATAOBJ_MARKER_H


#include "../tpl/ptrhashtable_tpl.h"

#include "../utils/simthread.h"

class grund_t;

/**
 * Class to mark tiles as visited during route search.
 * Singleton.
 */
class marker_t {
	// Hajo: added bit mask, because it allows a more efficient
	// implementation (use & instead of %)
	enum { bit_unit = (8 * sizeof(unsigned char)),
		bit_mask = (8 * sizeof(unsigned char))-1 };

	/// bit-field to mark ground tiles
	unsigned char *bits;

	/// length of field
	int bits_length;

	/// bit-field is made for this x-size
	int cached_size_x;

	/// hashtable to mark non-ground tiles (bridges, tunnels)
	ptrhashtable_tpl <const grund_t *, bool> more;

	/**
	 * Initializes marker. Set all tiles to not marked.
	 * @param world_size_x x-size of map
	 * @param world_size_y y-size of map
	 */
	void init(int world_size_x,int world_size_y);

	/// the instance (single threaded only)
	static marker_t the_instance;

public:

	/// For running multi-threadedly
	static marker_t* markers;

	marker_t() : bits(NULL) { bits_length = 0; init(0, 0); }
	~marker_t();

	/**
	 * Return handle to marker instance.
	 * @param world_size_x x-size of map
	 * @param world_size_y y-size of map
	 * @returns handle to the singleton instance
	 */
	static marker_t& instance(int world_size_x, int world_size_y, uint32 thread_number);

	/**
	 * Marks tile as visited.
	 */
	void mark(const grund_t *gr);

	/**
	 * Unmarks tile as visited.
	 */
	void unmark(const grund_t *gr);

	/**
	 * Checks if tile is visited.
	 * @returns true if tile was already visited
	 */
	bool is_marked(const grund_t *gr) const;

	/**
	 * Checks if tile is visited. Marks tile as visited if not visited before.
	 * @returns true if tile was already visited
	 */
	bool test_and_mark(const grund_t *gr);

	/**
	 * Marks all fields as not visited.
	 */
	void unmark_all();
};

#endif

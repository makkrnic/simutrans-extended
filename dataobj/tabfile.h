/*
 * This file is part of the Simutrans-Extended project under the Artistic License.
 * (see LICENSE.txt)
 */

#ifndef DATAOBJ_TABFILE_H
#define DATAOBJ_TABFILE_H


#include <stdio.h>

#include "../tpl/stringhashtable_tpl.h"

class tabfileobj_t;
class koord;
class scr_coord;
class scr_size;

class obj_info_t
{
public:
	bool retrieved;
	const char *str;
	obj_info_t() { retrieved=false; str=0; }
	obj_info_t(bool b, const char *s ) { retrieved=b; str=s; }
};

/*
 * This class can be used instead of FILE to read a game definition file,
 * usually with extension .tab in simutrans.
 * For the start only bridges.tab is read by this class.
 * Maybe we can make it a standard class for all tab-files, using the same
 * format in all.
 *
 * File format:
 *	Lines starting with '#' or ' ' are comment lines.
 *	The file content is treated as a list of objects.
 *	Objects are separated by a line starting with a dash (-)
 *	Each object can contain any number of lines in the format '<Key>=<Value>'
 *	These line are NOT ordered
 *	If keys are duplicated for one object, the first value is used
 *	Keys are not case sensitive
 *
 * @author V. Meyer
 */
class tabfile_t {
private:
	FILE *file;

	/**
	 * Read one non-comment line from input.
	 * Lines starting with ' ' are comment lines here. This differs from the
	 * global read_line() function.
	 *
	 * @return bool	false in case of eof
	 * @param s		line buffer
	 * @param size	size of line buffer
	 *
	 * @author V. Meyer
	 */
	bool read_line(char *s, int size);

	/**
	 * Return parameters and expansions
	 *
	 * @author Kieron Green
	 */
	int find_parameter_expansion(char *key, char *data, int *parameters, int *expansions, char *parameter_ptr[10], char *expansion_ptr[10]);

	/**
	 * Calculates expression provided in buffer, substituting parameters provided
	 *
	 * @author Kieron Green
	 */
	int calculate(char *expression, int parameter_value[10][256], int parameters, int combination[10]);

	/**
	 * Adds brackets to expression to ensure calculate_internal processes expression correctly
	 *
	 * @author Kieron Green
	 */
	void add_operator_brackets(char *expression, char *processed);

	/**
	 * Calculates expression provided in buffer (do not call directly!)
	 *
	 * @author Kieron Green
	 */
	int calculate_internal(char *expression, int parameter_value[10][256], int parameters, int combination[10]);

	/**
	 * Format the key string (trimright and lowercase)
	 *
	 * @author V. Meyer
	 */
	void format_key(char *key);

	/**
	 * Format the value string (trimleft and trimright)
	 *
	 * @author V. Meyer
	 */
	void format_value(char *value);

public:
	tabfile_t() : file(NULL) {}
	~tabfile_t() { close(); }

	bool open(const char *filename);

	void close();

	/**
	 * Read an entire object from the open file.
	 *
	 * @return bool	false, if empty object or eof
	 * @param &objinfo  will receive the object info
	 *
	 * @author V. Meyer
	 */
	bool read(tabfileobj_t &objinfo);
};


/*
 * This class represents an object read from a tabfile_t.
 * It contains all strings key/value pairs read by tabfile_t::read().
 * It may be reused for reading more objects.
 *
 * @author V. Meyer
 */
class tabfileobj_t {
private:
	stringhashtable_tpl<obj_info_t> objinfo;

	template<typename T>
	bool get_x_y( const char *key, T &x, T &y );

public:
	tabfileobj_t() { ; }
	~tabfileobj_t() { clear(); }

	/**
	 * prints all unused options lines in the file which do not start with a character from exclude_start_chars
	 */
	void unused( const char *exclude_start_chars );

	/*
	 * add an key/value pair - should only be used be tabfile_t::read
	 *
	 * @author V. Meyer
	 */
	bool put(const char *key, const char *value);

	/*
	 * reinitializes this object
	 *
	 * @author V. Meyer
	 */
	void clear();

	/**
	 * Get the value for a key - key must be lowercase
	 *
	 * @return const char *	returns at least an empty string, never NULL.
	 *
	 * @author V. Meyer
	 */
	const char *get(const char *key);

	/**
	 * Get the string value for a key - key must be lowercase
	 * @return def if key isn't found, value otherwise
	 * @author Hj. Malthaner
	 */
	const char *get_string(const char *key, const char * def);

	/**
	 * Get the value for a koord key - key must be lowercase
	 *
	 * @return koord	returns def, if key is not found
	 *
	 * @author V. Meyer
	 */
	const koord &get_koord(const char *key, koord def);
	const scr_coord &get_scr_coord(const char *key, scr_coord def);
	const scr_size &get_scr_size(const char *key, scr_size def);

	/**
	 * Get a color index or the next matching color when given a #AABBCC
	 * @author prissi
	 */
	uint8 get_color(const char *key, uint8 def);

	/**
	 * Get an int
	 * @author V. Meyer
	 */
	int get_int(const char *key, int def);

	/**
	 * Get an sint64 (actually uses double, thus only 48 bits are retrievable)
	 * @author prissi
	 */
	sint64 get_int64(const char *key, sint64 def);

	/**
	 * Parses a value with the format "<num 1>,<num 2>,..,<num N>"
	 * and returns an allocated int[N + 1] with
	 * N at pos. 0, <num 1> at pos 1, etc.
	 * Do not forget to "delete []" the returned value.
	 * @return const char *	returns at least an int[1], never NULL.
	 *
	 * @author V. Meyer
	 */
	int *get_ints(const char *key);
	sint64 *get_sint64s(const char *key);
};

#endif

/*
 * This file is part of the Simutrans-Extended project under the Artistic License.
 * (see LICENSE.txt)
 */

#ifndef MACROS_H
#define MACROS_H


#include "simtypes.h"

// Ensures that the argument has array type.
template<typename T, size_t N>
static inline size_t lengthof(T (&)[N]) { return N; }

#define endof(x) ((x) + lengthof(x))

#define QUOTEME_(x) #x
#define QUOTEME(x) QUOTEME_(x)

#define MEMZERON(ptr, n) memset((ptr), 0, sizeof(*(ptr)) * (n))
#define MEMZERO(obj)     MEMZERON(&(obj), 1)

namespace sim {

// make sure, a value in within the borders
template<typename T> static inline T clamp(T v, T l, T u)
{
	return v < l ? l : (v > u ? u :v);
}


template<class T> inline void swap(T& a, T& b)
{
	T t = a;
	a = b;
	b = t;
}

}
#endif


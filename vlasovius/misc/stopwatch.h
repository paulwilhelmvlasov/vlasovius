/*
 * Copyright (C) 2021 Matthias Kirchhart and Paul Wilhelm
 *
 * This file is part of vlasovius.
 *
 * vlasovius is free software; you can redistribute it and/or modify it under
 * the terms of the GNU General Public License as published by the Free
 * Software Foundation; either version 3, or (at your option) any later
 * version.
 *
 * vlasovius is distributed in the hope that it will be useful, but WITHOUT ANY
 * WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 * FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
 * details.
 *
 * You should have received a copy of the GNU General Public License along with
 * vlasovius; see the file COPYING.  If not see http://www.gnu.org/licenses.
 */
#ifndef VLASOVIUS_MISC_STOPWATCH_H
#define VLASOVIUS_MISC_STOPWATCH_H

#include <chrono>

namespace vlasovius
{

namespace misc
{

class stopwatch
{
public:
	void   reset();
	double elapsed();

private:
	using clock = std::chrono::high_resolution_clock;
	clock::time_point t0 { clock::now() };
};


inline
void stopwatch::reset()
{
	t0 = clock::now();
}

inline
double stopwatch::elapsed()
{
	using seconds = std::chrono::duration<double,std::ratio<1,1>>;

	auto tnow = clock::now();
	auto duration = std::chrono::duration_cast<seconds>( tnow - t0 );

	return duration.count();
}

}

}

#endif

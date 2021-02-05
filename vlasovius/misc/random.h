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
#ifndef VLASOVIUS_MISC_RANDOM_H
#define VLASOVIUS_MISC_RANDOM_H

#include <random>
#include <functional>

namespace vlasovius
{

namespace misc
{

class random_double
{
public:
	random_double( double min, double max );

	double operator()() const;
private:
	std::function<double()> r;
};




inline
random_double::random_double( real min, real max ):
 r( std::bind( std::uniform_real_distribution<>(min,max),
               std::default_random_engine() ) )
{}

inline
real random_double::operator()() const
{
	return r();
}


template <typename Int = int>
class random_int
{
public:
    random_int( Int min, Int max );

    Int operator()() const;

private:
    std::function<Int()> r;
};

template <typename Int>
inline random_int<Int>::random_int( Int min, Int max ):
 r( std::bind( std::uniform_int_distribution<Int>(min,max),
               std::default_random_engine() ) )
{}

template <typename Int>
inline Int random_int<Int>::operator()() const
{
	return r();
}

}

}

#endif


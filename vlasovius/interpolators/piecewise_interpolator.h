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
#ifndef VLASOVIUS_INTERPOLATORS_PIECEWISE_INTERPOLATOR_H
#define VLASOVIUS_INTERPOLATORS_PIECEWISE_INTERPOLATOR_H

#include <vector>
#include <armadillo>
#include <vlasovius/interpolators/direct_interpolator.h>

namespace vlasovius
{

namespace interpolators
{

template <typename kernel>
class piecewise_interpolator
{
public:
	piecewise_interpolator( kernel K, const arma::mat &X, const arma::mat &f,
					        size_t min_per_box, double tikhonov_mu = 0, size_t num_threads = 1 );

	arma::mat operator()( const arma::mat &Y, size_t num_threads = 1 ) const;

//private:
	arma::mat cover;
	size_t nrhs;
	std::vector< direct_interpolator<kernel> > local_interpolants;
};

}

}


#include <vlasovius/interpolators/piecewise_interpolator.tpp>
#endif

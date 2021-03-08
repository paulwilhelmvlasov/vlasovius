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
#ifndef VLASOVIUS_GEOMETRY_KD_TREE_H
#define VLASOVIUS_GEOMETRY_KD_TREE_H

#include <armadillo>

namespace vlasovius
{

namespace geometry
{

class kd_tree
{
public:
	kd_tree( const arma::mat &p_X );

	arma::mat  point_query( const arma::rowvec &q ) const;
	arma::uvec index_query( const arma::rowvec &q ) const;

	arma::mat  covering_boxes( size_t min_per_box ) const;

private:
	arma::mat  X;
	arma::uvec idx;
};

}

}

#endif

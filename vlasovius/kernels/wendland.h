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
#ifndef VLASOVIUS_KERNELS_WENDLAND_H
#define VLASOVIUS_KERNELS_WENDLAND_H

#include <armadillo>

namespace vlasovius
{

namespace kernels
{

template <size_t dim, size_t k>
double wendland( double r );

template <size_t dim, size_t k>
arma::vec wendland( const arma::vec &r );

}

}

#include <vlasovius/kernels/wendland.tpp>
#endif

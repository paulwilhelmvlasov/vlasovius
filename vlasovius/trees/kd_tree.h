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


#ifndef VLASOVIUS_TREES_KD_TREE_H
#define VLASOVIUS_TREES_KD_TREE_H


#include <vector>
#include <armadillo>

namespace vlasovius
{
	namespace trees
	{



		class kd_tree
		{
		public:
			kd_tree(arma::mat& points);


		};

	}
}

#endif /* VLASOVIUS_TREES_KD_TREE_H_ */

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


#include <deque>
#include <algorithm>

#include <armadillo>

#include <vlasovius/misc/row_iter.h>

namespace vlasovius
{
	namespace trees
	{
		struct bounding_box
		{
			arma::vec center;
			arma::vec sidelength; // Distance from center to border
			// in each direction.
		};

		struct node
		{
			int parent 		= -1;

			int firstChild  = -1;
			int secondChild = -1;

			arma::uword indexFirstElem;
			arma::uword indexLastElem;

			bounding_box box;
		};

		template<arma::uword dim> bool compVec
		(const arma::vec& first, const arma::vec& second);

		class kd_tree
		{
		public:
			kd_tree(arma::mat& points, size_t minPerBox, size_t maxPerBox);

		private:
			void buildTree(arma::mat& points, size_t currentNodeIndex, size_t minPerBox, size_t maxPerBox);
			size_t splittingDimension(size_t currentNodeIndex);
			void split(size_t currentNodeIndex, size_t dimSplit);

		private:
			size_t dim = 0;

			std::deque<node> nodes; // deque to avoid reallocating the underlying array several times
			// as the struct node might potentially be a large datatype.
		};

	}
}

#endif /* VLASOVIUS_TREES_KD_TREE_H_ */

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

#include "vlasovius/trees/kd_tree.h"

namespace vlasovius
{

	namespace trees
	{
		void kd_tree::buildTree(size_t currentNodeIndex, size_t minPerBox, size_t maxPerBox)
		{
			// Use the standard splitting rule for kd-trees, i.e.:
			// The split-dimension is the one with maximum spread
			// of points and the splitting value is the median of
			// the respective coordinates of the points in the
			// current box.

			size_t numIndicesInBox = nodes[currentNodeIndex].point_indices.size();
			if(numIndicesInBox > maxPerBox && (numIndicesInBox / 2) >= minPerBox){
				// In this case split the node:

				// Init new nodes:
				nodes.push_back(node());
				nodes.push_back(node());

				size_t firstChild = nodes.size() - 2;
				size_t secondChild = nodes.size() - 1;

				nodes[currentNodeIndex].firstChild = firstChild;
				nodes[currentNodeIndex].secondChild = secondChild;

				nodes[firstChild].parent = currentNodeIndex;
				nodes[secondChild].parent = currentNodeIndex;

				// Compute splitting dimension:
				size_t dimSplit = splittingDimension(currentNodeIndex);

				// Compute splitting value:
			}
		}


		size_t kd_tree::splittingDimension(size_t currentNodeIndex)
		{
			// We use the maximum sidelength of the current box as an
			// approximation to the maximum spread. While this might
			// not be 100% accurate in some cases, it will certainly
			// give a good approximation in most cases.

			if(dim == 1){
				return 0;
			} else{
				size_t dimSplit = 0;
				double maxLength = nodes[currentNodeIndex].box.sidelength(0);

				for(size_t i = 1; i < dim; i++)
				{
					double currLength = nodes[currentNodeIndex].box.sidelength(i);
					if(maxLength < currLength)
					{
						dimSplit = i;
						maxLength = currLength;
					}
				}

				return dimSplit;
			}
		}

		void kd_tree::split(size_t currentNodeIndex, size_t dimSplit)
		{

		}

	}
}

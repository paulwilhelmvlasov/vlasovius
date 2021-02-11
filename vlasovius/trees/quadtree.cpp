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

#include "quadtree.h"

bool vlasovius::trees::bounding_box_2d::pointInsideAABB(const arma::vec2& x, double tol)
{
	return norm(x - center, "inf") < radius + tol;
}

vlasovius::trees::quadtree::quadtree(arma::mat& points, size_t minElemPerBox, size_t maxElemPerBox)
{
	double maxY = points.row(1).max();
	double maxX = points.row(0).max();
	double minY = points.row(1).min();
	double minX = points.row(0).min();

	arma::vec center = { 0.5 * (maxX + minX), 0.5 * (maxY + minY) };
	double radius = std::max((maxX - minX), (maxY - minY));
	std::vector<size_t> children(size(points));
	std::iota(children.begin(), children.end(), 0);
	nodes.push_back({ 0, 0, 0, 0, 0, children,
		bounding_box_2d({radius, center}));

	buildTree(points, minElemPerBox, maxElemPerBox, nodes[0], 0);
}

void vlasovius::trees::quadtree::buildTree(arma::mat& points,
	size_t minElemPerBox, size_t maxElemPerBox,
	node& currNode, size_t currIndex)
{
	// This method splits a node if it has reached its maximum capacity. In this case the 4 child-nodes
	// are initialized with their correct centers, radii and then pushed back into the nodes-deque. 
	// Then all child-points of the current node are assigned to the correct child-node. 
	// When all child-points are assigned, this method is again called for each child node s.t. a 
	// quadtree is computed recursively.
	// Note: MinElemPerBox is not respected currently. There is no straigtht-forward way to ensure that each 
	// box contains at most min Elements and at most max Elements. 

	if (currNode.childIndices.size() > maxElem) { 
		// topLeft:
		nodes.push_back(node({ currIndex, 0, 0, 0, 0, {},
			bounding_box_2d({0.5 * currNode.box.radius, currNode.box.center
				+ 0.5 * currNode.box.radius * arma::vec({-1, 1})}) })); 
		size_t topLeft = nodes.size() - 1;
		currNode.topLeft = topLeft;
		// topRight:
		nodes.push_back(node({ currIndex, 0, 0, 0, 0, {},
			bounding_box_2d({0.5 * currNode.box.radius, currNode.box.center
				+ 0.5 * currNode.box.radius * arma::vec({1, 1})}) })); 
		size_t topRight = nodes.size() - 1;
		currNode.topRight = topRight;
		// botRight:
		nodes.push_back(node({ currIndex, 0, 0, 0, 0, {},
			bounding_box_2d({0.5 * currNode.box.radius, currNode.box.center
				+ 0.5 * currNode.box.radius * arma::vec({1, -1})}) })); // topLeft
		size_t botRight = nodes.size() - 1;
		currNode.botRight = botRight;
		// botLeft:
		nodes.push_back(node({ currIndex, 0, 0, 0, 0, {},
			bounding_box_2d({0.5 * currNode.box.radius, currNode.box.center
				+ 0.5 * currNode.box.radius * arma::vec({-1, -1})}) })); // topLeft
		size_t botLeft = nodes.size() - 1;
		currNode.botLeft = botLeft;

		for (size_t i = 0; i < currNode.childIndices.size(); i++) {
			// This is more stable than directly using "pointInsideAABB(...)". 
			size_t indexPoint = currNode.childIndices[i];

			double distTopLeft = norm(currNode.box.center - nodes[topLeft].box.center, "inf");
			double distTopRight = norm(currNode.box.center - nodes[topRight].box.center, "inf");
			double distBotRight = norm(currNode.box.center - nodes[botRight].box.center, "inf");
			double distBotLeft = norm(currNode.box.center - nodes[botLeft].box.center, "inf");

			arma::vec v({ distTopLeft, distTopRight, distBotRight, distBotLeft });
			size_t indexMinDist = index_min(v);

			switch (indexMinDist)
			{
			case 0:
				nodes[topLeft].childIndices.push_back(indexPoint);
				break;
			case 1:
				nodes[topRight].childIndices.push_back(indexPoint);
				break;
			case 2:
				nodes[botRight].childIndices.push_back(indexPoint);
				break;
			case 3:
				nodes[botLeft].childIndices.push_back(indexPoint);
				break;
			default:
				break;
			}
		}

		currNode.childIndices.clear();

		buildTree(points, minElemPerBox, maxElemPerBox,
			nodes[topLeft], topLeft); 
		buildTree(points, minElemPerBox, maxElemPerBox,
			nodes[topRight], topRight);
		buildTree(points, minElemPerBox, maxElemPerBox,
			nodes[botRight], botRight);
		buildTree(points, minElemPerBox, maxElemPerBox,
			nodes[botLeft], botLeft);
	}
}

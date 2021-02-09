#include <iostream>

#include <vlasovius/kernels/wendland.h>
#include <vlasovius/interpolators/directInterpolator.h>

int main() 
{
	arma::mat guenther(10, 10, arma::fill::randu);
	std::cout << "Ja moin" << std::endl;
	std::cout << inv(guenther);

	std::vector<arma::vec> points({arma::vec({0}), arma::vec({1}), arma::vec({2})});
	arma::vec f({1, 1, 1});

	using vlasovius::interpolators::directInterpolator;
	using vlasovius::kernels::wendland;
	directInterpolator< wendland<1, 2> > sfx(points, f, 2); 

	return 0;
}

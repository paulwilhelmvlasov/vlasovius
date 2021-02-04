#include <iostream>
#include <armadillo>

int main() 
{
	arma::mat guenther(10, 10, arma::fill::randu);
	std::cout << "Ja moin" << std::endl;
	std::cout << inv(guenther);

	return 0;
}
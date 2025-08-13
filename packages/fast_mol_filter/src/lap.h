/*************** CONSTANTS  *******************/

#define BIG 1.79769e+308 // max value for double

/*************** TYPES      *******************/

typedef int row;
typedef int col;
typedef double cost;

/*************** FUNCTIONS  *******************/

extern cost lap(int dim, cost **assigncost, col *rowsol, row *colsol, cost *u, cost *v);
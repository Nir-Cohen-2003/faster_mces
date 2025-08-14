/*************** CONSTANTS  *******************/

#define BIG 3.402823e+38 // max value for float

/*************** TYPES      *******************/

typedef int row;
typedef int col;
typedef float cost;

/*************** FUNCTIONS  *******************/

extern cost lap(int dim, cost **assigncost, col *rowsol, row *colsol, cost *u, cost *v);
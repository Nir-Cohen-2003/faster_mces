#include <cstdint>
#include <climits>

// *************** CONSTANTS  *******************
#define BIG (65536 / 4) // large sentinel for integer costs

/*************** TYPES      *******************/

typedef int row;
typedef int col;
typedef uint16_t cost;

/*************** FUNCTIONS  *******************/

extern cost lap(int dim, cost **assigncost, col *rowsol, row *colsol, cost *u, cost *v);
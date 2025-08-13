//
// Reading molecules - example1.cpp

#include <iostream>
#include <GraphMol/GraphMol.h>
#include <GraphMol/SmilesParse/SmilesParse.h>

int main(int argc, char **argv) {
  RDKit::ROMol *mol1 = RDKit::SmilesToMol("Cc1ccccc1");
  std::cout << "Number of atoms " << mol1->getNumAtoms() << std::endl;

  // using namespace RDKit;
  // auto mol = "C[C@H](F)c1ccc(C#N)cc1"_smiles;
  // std::cout << "Number of atoms : " << mol->getNumAtoms() << std::endl;

}

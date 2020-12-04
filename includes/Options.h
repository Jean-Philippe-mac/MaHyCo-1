#ifndef OPTIONSSCHEMA_H
#define OPTIONSSCHEMA_H

#include <iostream>
using namespace std;
#include<string>

namespace optionschemalib {

class OptionsSchema {
 public:
  struct Options {
    int nbmat = -1;
    double threshold = 1.0E-16;
    int spaceOrder = 2;
    int projectionOrder = 2;
    int projectionConservative = 0;
    int AvecProjection = 1;
    int AvecParticules = 0;
    int Adiabatique = 1;
    int Isotherme = 2;
    int AvecEquilibrage = -1;
    int pseudo_centree = 1;
    int methode_flux_masse = 0;
    int sansLagrange = 0;
    string fichier_sortie1D;
  };
  Options* options;

 private:
};
}  // namespace optionschemalib
#endif  // OPTIONSSCHEMA_H

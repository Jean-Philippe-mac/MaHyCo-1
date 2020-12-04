#ifndef CONSTANTES_H
#define CONSTANTES_H

#include "types/Types.h"  // for RealArray1D, RealArray2D
using namespace nablalib;

// dimension du code
static const int dim = 2;
static const int dimplus1 = 3;
// Nombre de matériaux maximum autorisé
static const int nbmatmax = 3;
// Nombre d'équation maximum autorisé
static const int nbequamax = 3 * nbmatmax + 2 + 1 + 1;
// (volumes, masses, energies internes) * nbmatmax
// + vitesses + energie cinétique + pseudo
// Nombre de Particules autorisées
static const int nbPartMax = 2000;

const RealArray1D<dim> ex = {{1.0, 0.0}};
const RealArray1D<dim> ey = {{0.0, 1.0}};
const RealArray1D<dim> zeroVect = {{0.0, 0.0}};
const RealArray2D<dim, dim> zeroMat = {{0.0, 0.0, 0.0, 0.0}};
const RealArray1D<nbmatmax> zeroVectmat = {{0.0, 0.0, 0.0}};
const RealArray1D<nbequamax> Uzero = {
    {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}};

// constantes
const double Pi = 3.14159265359;
const double viscosity = 1.715e-5;
const double Cp = 2.84;
const double Pr = 0.7;

#endif  // CONSTANTES_H

#ifndef CONDITIONSLIMITES_H
#define CONDITIONSLIMITES_H


namespace conditionslimiteslib
{

class ConditionsLimites {
 public:
  // conditions aux limites
  struct Cdl {
    RealArray1D<dim> ex = {{1.0, 0.0}};
    RealArray1D<dim> ey = {{0.0, 1.0}};
    RealArray1D<nbequamax> Uzero = {
        {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}};   
    int symmetry = 200;
    int imposedVelocity = 201;
    int freeSurface = 202;
    int leftFluxBC = 0;
    RealArray1D<nbequamax> leftFluxBCValue = Uzero;
    int rightFluxBC = 0;
    RealArray1D<nbequamax> rightFluxBCValue = Uzero;
    int bottomFluxBC = 0;
    RealArray1D<nbequamax> bottomFluxBCValue = Uzero;
    int topFluxBC = 0;
    RealArray1D<nbequamax> topFluxBCValue = Uzero;
    int FluxBC = leftFluxBC + rightFluxBC + bottomFluxBC + topFluxBC;
    int leftBC = 0;
    RealArray1D<dim> leftBCValue = ey;

    int rightBC = 0;
    RealArray1D<dim> rightBCValue = ey;

    int topBC = 0;
    RealArray1D<dim> topBCValue = ex;

    int bottomBC = 0;
    RealArray1D<dim> bottomBCValue = ex;
  };
  Cdl* cdl;

 private:

};
} // namespace
#endif  // CONDITIONSLIMITES_H

#ifndef EOS_H
#define EOS_H

namespace eoslib {

class EquationDetat {
 public:
  // EOS
  int Void = 100;
  int PerfectGas = 101;
  int StiffenedGas = 102;
  int Fictif = 103;
  int SolidLinear = 104;
  IntArray1D<nbmatmax> Nom = {{PerfectGas, PerfectGas, PerfectGas}};
  RealArray1D<nbmatmax> gamma = {{1.4, 1.4, 1.4}};
  RealArray1D<nbmatmax> tension_limit = {{0.01, 0.01, 0.01}};
  /**
   * 
   */
  RealArray1D<3> computeEOSGP(double gamma, double limit_tension, double rho, double energy) {
    double pression = (gamma -1.) * rho * energy;
    double sound_speed = std::sqrt(gamma * (gamma -1.) * energy);
    double dpde = (gamma -1.) * rho;
    //std::cout << " rho " << rho << "  pression " << pression << std::endl;
    return {pression, sound_speed, dpde};
  }
  
  RealArray1D<3> computeEOSVoid(double gamma, double limit_tension, double rho, double energy) {
    double pression = 0.;
    double sound_speed = 0.;
     double dpde = 0.;
    return {pression, sound_speed, dpde};
  }
  
  RealArray1D<3> computeEOSSTIFG(double gamma, double limit_tension, double rho, double energy) {
    double pression;
    double sound_speed;
    double dpde;
    if (rho !=0.) {
      pression = ((gamma- 1.) * rho * energy) - (gamma * limit_tension);
      sound_speed = sqrt((gamma/rho)*(pression+limit_tension));
      dpde = (gamma -1.) * rho;
    } else {
      pression = 0.;
      sound_speed = 0.;
      dpde = 0.;
    }
    //std::cout << " rho " << rho << "  pression " << pression << std::endl;
    return {pression, sound_speed, dpde};   
  }
  
  RealArray1D<3> computeEOSFictif(double gamma, double limit_tension, double rho, double energy) {
    double pression = 5.;
    double sound_speed = std::sqrt(gamma * (gamma -1.) * energy);
    double dpde = 0.;
    return {pression, sound_speed, dpde};   
  }
  
  RealArray1D<3> computeEOSSL(double gamma, double limit_tension, double rho, double energy) {
    std::cout << " Pas encore programmée" << std::endl;
  }
  RealArray1D<3> computeEOS(int imat, double gamma, double limit_tension, double rho, double energy){
    if (imat == Void) return computeEOSVoid(gamma, limit_tension, rho, energy);
    if (imat == PerfectGas) return computeEOSGP(gamma, limit_tension, rho, energy);
    if (imat == StiffenedGas) return computeEOSSTIFG(gamma, limit_tension, rho, energy);
    if (imat == Fictif) return computeEOSFictif(gamma, limit_tension, rho, energy);
    if (imat == SolidLinear) return computeEOSSL(gamma, limit_tension, rho, energy);
  }
   
 private:
};
}  // namespace eoslib
#endif  // EOS_H

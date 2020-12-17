#include <math.h>  // for floor, sqrt

#include <Kokkos_Core.hpp>  // for deep_copy
#include <algorithm>        // for copy
#include <array>            // for array
#include <iostream>         // for operator<<, basic_ostream::ope...
#include <vector>           // for allocator, vector

#include "Init.h"

#include "mesh/CartesianMesh2D.h"  // for CartesianMesh2D
#include "types/MathFunctions.h"   // for max, norm, dot
#include "types/MultiArray.h"      // for operator<<
#include "utils/Utils.h"           // for indexOf

namespace initlib {

void Initialisations::initVarSOD() noexcept {
  Kokkos::parallel_for(nbCells, KOKKOS_LAMBDA(const size_t& cCells) {
    double r(0.);
    double pInit;
    double rhoInit;
    double eInit;
    if (test->Nom == test->SodCaseX) r = m_cell_coord_n0(cCells)[0];
    if (test->Nom == test->SodCaseY) r = m_cell_coord_n0(cCells)[1];
    m_fracvol_env_n0(cCells)[0] = 1.;
    m_mass_fraction_env_n0(cCells)[0] = 1.;
    if (r < 0.5) {
      pInit = 1.0;
      rhoInit = 1.0;
      eInit = pInit / ((eos->gamma[0] - 1.0) * rhoInit);
    } else {
      pInit = 0.1;
      rhoInit = 0.125;
      eInit = pInit / ((eos->gamma[0] - 1.0) * rhoInit);
    }
    m_density_n0(cCells) = rhoInit;
    m_density_env_n0(cCells)[0] = rhoInit;

    m_pressure_n0(cCells) = pInit;
    m_pressure_env_n0(cCells)[0] = pInit;

    m_internal_energy_n0(cCells) = eInit;
    m_internal_energy_env_n0(cCells)[0] = eInit;

    m_speed_velocity_env_n0(cCells)[0] =
        std::sqrt(eos->gamma[0] * m_density_env_n0(cCells)[0] /
                  m_pressure_env_n0(cCells)[0]);
    m_speed_velocity_n0(cCells) = m_speed_velocity_env_n0(cCells)[0];

    // vitesses
    m_cell_velocity_n0(cCells) = {0.0, 0.0};
  });
  Kokkos::parallel_for(nbNodes, KOKKOS_LAMBDA(const size_t& pNodes) {
    m_node_velocity_n0(pNodes) = {0.0, 0.0};
  });
}
void Initialisations::initVarBiSOD() noexcept {
  Kokkos::parallel_for(nbCells, KOKKOS_LAMBDA(const size_t& cCells) {
    double r(0.);
    double pInit1, pInit2;
    double rhoInit1, rhoInit2;
    double eInit1, eInit2;
    double f1;
    double f2;
    if (test->Nom == test->BiSodCaseX) r = m_cell_coord_n0(cCells)[0];
    if (test->Nom == test->BiSodCaseY) r = m_cell_coord_n0(cCells)[1];
    if (r < 0.5) {
      f1 = 1.0;
      f2 = 0.0;
      rhoInit1 = 1.0;
      pInit1 = 1.0;
      eInit1 = pInit1 / ((eos->gamma[0] - 1.0) * rhoInit1);
      rhoInit2 = 0.0;
      pInit2 = 0.0;
      eInit2 = 0.0;
    } else {
      f1 = 0.0;
      f2 = 1.0;
      rhoInit1 = 0.0;
      pInit1 = 0.0;
      eInit1 = 0.0;
      rhoInit2 = 0.125;
      pInit2 = 0.1;
      eInit2 = pInit2 / ((eos->gamma[1] - 1.0) * rhoInit2);
    }
    m_fracvol_env_n0(cCells)[0] = f1;
    m_fracvol_env_n0(cCells)[1] = f2;
    m_mass_fraction_env_n0(cCells)[0] = f1;
    m_mass_fraction_env_n0(cCells)[1] = f2;
    m_density_n0(cCells) = f1 * rhoInit1 + f2 * rhoInit2;
    m_density_env_n0(cCells)[0] = rhoInit1;
    m_density_env_n0(cCells)[1] = rhoInit2;
    m_pressure_n0(cCells) = f1 * pInit1 + f2 * pInit2;
    m_pressure_env_n0(cCells)[0] = pInit1;
    m_pressure_env_n0(cCells)[1] = pInit2;
    m_internal_energy_n0(cCells) = f1 * eInit1 + f2 * eInit2;
    m_internal_energy_env_n0(cCells)[0] = eInit1;
    m_internal_energy_env_n0(cCells)[1] = eInit2;

    if (f1 > 0.)
      m_speed_velocity_env_n0(cCells)[0] =
        std::sqrt(eos->gamma[0] * m_density_env_n0(cCells)[0] /
                  m_pressure_env_n0(cCells)[0]);
    else
      m_speed_velocity_env_n0(cCells)[0] = 1.e20; // pour avoir le min sur 2

    if (f2 > 0.)
      m_speed_velocity_env_n0(cCells)[1] =
        std::sqrt(eos->gamma[1] * m_density_env_n0(cCells)[1] /
                  m_pressure_env_n0(cCells)[1]);
    else
      m_speed_velocity_env_n0(cCells)[1] = 1.e20; // pour avoir le min sur 1

    m_speed_velocity_n0(cCells) = min(m_speed_velocity_env_n0(cCells)[0],
                                      m_speed_velocity_env_n0(cCells)[1]);

    // vitesses
    m_cell_velocity_n0(cCells) = {0.0, 0.0};
  });
  Kokkos::parallel_for(nbNodes, KOKKOS_LAMBDA(const size_t& pNodes) {
    m_node_velocity_n0(pNodes) = {0.0, 0.0};
  });
}

}  // namespace initlib

#ifndef VNR_H
#define VNR_H

/*---------------------------------------*/
/*---------------------------------------*/

#include <stddef.h>  // for size_t

#include <Kokkos_Core.hpp>                // for KOKKOS_LAMBD
#include <OpenMP/Kokkos_OpenMP_Exec.hpp>  // for OpenMP::impl_is_initialized

/* #include <fstream> */
/* #include <iomanip> */
/* #include <type_traits> */
/* #include <limits> */
/* #include <utility> */
/* #include <cmath> */
/* #include <Kokkos_hwloc.hpp> */

#include "../includes/CasTest.h"
#include "../includes/ConditionsLimites.h"
#include "../includes/Constantes.h"
#include "../includes/CstMesh.h"
#include "../includes/Eos.h"
#include "../includes/GestionTemps.h"
#include "../includes/Limiteurs.h"
#include "../includes/Options.h"
#include "../includes/SchemaLagrange.h"
#include "../includes/Sortie.h"
#include "../includes/VariablesLagRemap.h"
#include "../initialisations/Init.h"
#include "../particle_scheme/SchemaParticules.h"
#include "../remap/Remap.h"
#include "mesh/CartesianMesh2D.h"  // for CartesianMesh2D, CartesianM...
#include "mesh/PvdFileWriter2D.h"  // for PvdFileWriter2D
#include "types/Types.h"           // for RealArray1D, RealArray2D
#include "utils/Timer.h"           // for Timer
#include "utils/kokkos/Parallel.h"

using namespace nablalib;

/******************** Free functions declarations ********************/

template <size_t x>
KOKKOS_INLINE_FUNCTION double norm(RealArray1D<x> a);
template <size_t x>
KOKKOS_INLINE_FUNCTION double dot(RealArray1D<x> a, RealArray1D<x> b);
KOKKOS_INLINE_FUNCTION
double computeLpcPlus(RealArray1D<2> xp, RealArray1D<2> xpPlus);
KOKKOS_INLINE_FUNCTION
RealArray1D<2> computeNpcPlus(RealArray1D<2> xp, RealArray1D<2> xpPlus);
KOKKOS_INLINE_FUNCTION
double crossProduct2d(RealArray1D<2> a, RealArray1D<2> b);
KOKKOS_INLINE_FUNCTION
RealArray1D<2> computeLpcNpc(RealArray1D<2> xp, RealArray1D<2> xpPlus,
                             RealArray1D<2> xpMinus);
template <size_t N>
KOKKOS_INLINE_FUNCTION RealArray1D<N> symmetricVector(RealArray1D<N> v,
                                                      RealArray1D<N> sigma);
template <size_t x>
KOKKOS_INLINE_FUNCTION RealArray1D<x> sumR1(RealArray1D<x> a, RealArray1D<x> b);
KOKKOS_INLINE_FUNCTION
double sumR0(double a, double b);
KOKKOS_INLINE_FUNCTION
double minR0(double a, double b);

/******************** Module declaration ********************/

class Vnr {
 public:
 private:
  // Mesh (can depend on previous definitions)
  CartesianMesh2D* mesh;
  schemalagrangelib::SchemaLagrangeClass::SchemaLagrange* scheme;
  optionschemalib::OptionsSchema::Options* options;
  sortielib::Sortie::SortieVariables* so;
  castestlib::CasTest::Test* test;
  particleslib::SchemaParticules* particules;
  conditionslimiteslib::ConditionsLimites::Cdl* cdl;
  limiteurslib::LimiteursClass::Limiteurs* limiteurs;
  eoslib::EquationDetat* eos;
  cstmeshlib::ConstantesMaillagesClass::ConstantesMaillages* cstmesh;
  gesttempslib::GestionTempsClass::GestTemps* gt;
  variableslagremaplib::VariablesLagRemap* varlp;
  initlib::Initialisations* init;
  Remap* remap;
  PvdFileWriter2D writer;
  int nbPartMax;
  int nbPart = 0;
  size_t nbNodes, nbCells, nbFaces, nbInnerNodes, nbNodesOfCell, nbCellsOfNode;

  // Global declarations
  int n, nbCalls;
  double lastDump;
  double m_global_total_energy_L, m_global_total_energy_T,
    m_global_total_energy_0, m_global_total_energy_L0;
  double m_global_total_masse_L, m_global_total_masse_T, m_global_total_masse_0, m_global_total_masse_L0;

  // coordonnees
  Kokkos::View<RealArray1D<dim>*> m_node_coord_n;
  Kokkos::View<RealArray1D<dim>*> m_cell_coord_n;
  Kokkos::View<RealArray1D<dim>*> m_cell_coord_nplus1;
  // volume
  Kokkos::View<double*> m_node_volume;
  Kokkos::View<double*> m_euler_volume;
  Kokkos::View<double*> m_lagrange_volume_n;
  Kokkos::View<double*> m_lagrange_volume_nplus1;
  // densite
  Kokkos::View<double*> m_density_n;
  Kokkos::View<double*> m_density_nplus1;
  Kokkos::View<RealArray1D<nbmatmax>*> m_density_env_n;
  Kokkos::View<RealArray1D<nbmatmax>*> m_density_env_nplus1;
  // masse
  Kokkos::View<double*> m_cell_mass;
  Kokkos::View<RealArray1D<nbmatmax>*> m_cell_mass_env;
  // masse aux noeuds
  Kokkos::View<double*> m_node_mass;
  // pression 
  Kokkos::View<double*> m_pressure_n;
  Kokkos::View<double*> m_pressure_nplus1;
  Kokkos::View<RealArray1D<nbmatmax>*> m_pressure_env_n;
  Kokkos::View<RealArray1D<nbmatmax>*> m_pressure_env_nplus1;
  Kokkos::View<double*> m_pressure_env1;
  Kokkos::View<double*> m_pressure_env2;
  Kokkos::View<double*> m_pressure_env3;
  // energie interne
  Kokkos::View<double*> m_internal_energy_n;
  Kokkos::View<double*> m_internal_energy_nplus1;
  Kokkos::View<RealArray1D<nbmatmax>*> m_internal_energy_env_n;
  Kokkos::View<RealArray1D<nbmatmax>*> m_internal_energy_env_nplus1;
  // vitesse aux noeuds
  Kokkos::View<RealArray1D<dim>*> m_node_velocity_n;
  Kokkos::View<RealArray1D<dim>*> m_node_velocity_nplus1;
  Kokkos::View<double*> m_x_velocity;
  Kokkos::View<double*> m_y_velocity;
  // vitesse aux mailles
  Kokkos::View<RealArray1D<dim>*> m_node_coord_nplus1;
  // vitesse du son
  Kokkos::View<double*> m_speed_velocity_n;
  Kokkos::View<double*> m_speed_velocity_nplus1;
  Kokkos::View<RealArray1D<nbmatmax>*> m_speed_velocity_env_n;
  Kokkos::View<RealArray1D<nbmatmax>*> m_speed_velocity_env_nplus1;
  // fraction volumique
  Kokkos::View<RealArray1D<nbmatmax>*> m_fracvol_env;
  Kokkos::View<double*> m_fracvol_env1;
  Kokkos::View<double*> m_fracvol_env2;
  Kokkos::View<double*> m_fracvol_env3;
  Kokkos::View<RealArray1D<nbmatmax>*> m_node_fracvol;
  // fraction massique
  Kokkos::View<RealArray1D<nbmatmax>*> m_mass_fraction_env;
  // sorties interface
  Kokkos::View<double*> m_interface12;
  Kokkos::View<double*> m_interface13;
  Kokkos::View<double*> m_interface23;
  // sortie energie et masse globales
  Kokkos::View<double*> m_total_energy_0;
  Kokkos::View<double*> m_total_energy_T;
  Kokkos::View<double*> m_total_energy_L;
  Kokkos::View<double*> m_total_energy_L0;
  Kokkos::View<double*> m_total_masse_0;
  Kokkos::View<double*> m_total_masse_T;
  Kokkos::View<double*> m_total_masse_L;
  Kokkos::View<double*> m_total_masse_L0;

  Kokkos::View<double**> m_node_cellvolume_n;
  Kokkos::View<double**> m_node_cellvolume_nplus1;
  Kokkos::View<double*> m_pseudo_viscosity_n;
  Kokkos::View<double*> m_pseudo_viscosity_nplus1;
  Kokkos::View<RealArray1D<nbmatmax>*> m_pseudo_viscosity_env_n;
  Kokkos::View<RealArray1D<nbmatmax>*> m_pseudo_viscosity_env_nplus1;
  Kokkos::View<double*> m_tau_density_n;
  Kokkos::View<double*> m_tau_density_nplus1;
  Kokkos::View<RealArray1D<nbmatmax>*> m_tau_density_env_n;
  Kokkos::View<RealArray1D<nbmatmax>*> m_tau_density_env_nplus1;
  Kokkos::View<double*> m_tau_volume_n;
  Kokkos::View<double*> m_tau_volume_nplus1;
  Kokkos::View<RealArray1D<nbmatmax>*> m_tau_volume_env_n;
  Kokkos::View<RealArray1D<nbmatmax>*> m_tau_volume_env_nplus1;
  Kokkos::View<double*> m_divu_n;
  Kokkos::View<double*> m_divu_nplus1;
  Kokkos::View<RealArray1D<dim>**> m_cqs_n;
  Kokkos::View<RealArray1D<dim>**> m_cqs_nplus1;

  utils::Timer global_timer;
  utils::Timer cpu_timer;
  utils::Timer io_timer;

  const size_t maxHardThread =
      Kokkos::DefaultExecutionSpace::impl_max_hardware_threads();

 public:
 Vnr(schemalagrangelib::SchemaLagrangeClass::SchemaLagrange* ascheme,
      optionschemalib::OptionsSchema::Options* aOptions,
      cstmeshlib::ConstantesMaillagesClass::ConstantesMaillages* acstmesh,
      gesttempslib::GestionTempsClass::GestTemps* agt,
      castestlib::CasTest::Test* aTest,
      conditionslimiteslib::ConditionsLimites::Cdl* aCdl,
      limiteurslib::LimiteursClass::Limiteurs* aLimiteurs,
      particleslib::SchemaParticules* aParticules,
      eoslib::EquationDetat* aEos, CartesianMesh2D* aCartesianMesh2D,
      variableslagremaplib::VariablesLagRemap* avarlp, Remap* aremap,
      initlib::Initialisations* ainit, 
      sortielib::Sortie::SortieVariables* asorties,
      string output)
      : scheme(ascheme),
        options(aOptions),
        cstmesh(acstmesh),
        gt(agt),
        test(aTest),
        cdl(aCdl),
        limiteurs(aLimiteurs),
        particules(aParticules),
        eos(aEos),
        mesh(aCartesianMesh2D),
        varlp(avarlp),
        remap(aremap),
        init(ainit),
        nbCalls(0),
        lastDump(0.0),
        so(asorties),
        writer("VnrRemap", output),
        nbNodes(mesh->getNbNodes()),
        nbCells(mesh->getNbCells()),
        nbFaces(mesh->getNbFaces()),
        nbNodesOfCell(CartesianMesh2D::MaxNbNodesOfCell),
        nbCellsOfNode(CartesianMesh2D::MaxNbCellsOfNode),
        m_node_coord_n("node_coord_n", nbNodes),
        m_node_coord_nplus1("node_coord_nplus1", nbNodes),
        m_node_cellvolume_n("node_cellvolume_n", nbCells, nbNodesOfCell),
        m_node_cellvolume_nplus1("node_cellvolume_nplus1", nbCells,
                                 nbNodesOfCell),
        m_node_volume("node_volume", nbNodes),
        m_total_energy_0("total_energy_0", nbCells),
        m_total_energy_T("total_energy_T", nbCells),
        m_total_energy_L("total_energy_L", nbCells),
        m_total_energy_L0("total_energy_L0", nbCells),
        m_total_masse_0("total_masse_0", nbCells),
        m_total_masse_T("total_masse_T", nbCells),
        m_total_masse_L("total_masse_L", nbCells),
        m_total_masse_L0("total_masse_L0", nbCells),
        m_density_n("density_n", nbCells),
        m_density_nplus1("density_nplus1", nbCells),
        m_density_env_n("density_env_n", nbCells),
        m_density_env_nplus1("density_env_nplus1", nbCells),
        m_pressure_n("pressure_n", nbCells),
        m_pressure_nplus1("pressure_nplus1", nbCells),
        m_pressure_env1("pressure_env1", nbCells),
        m_pressure_env2("pressure_env2", nbCells),
        m_pressure_env3("pressure_env3", nbCells),
        m_pressure_env_n("pressure_env_n", nbCells),
        m_pressure_env_nplus1("pressure_env_nplus1", nbCells),
        m_pseudo_viscosity_n("pseudo_viscosity_n", nbCells),
        m_pseudo_viscosity_nplus1("pseudo_viscosity_nplus1", nbCells),
        m_pseudo_viscosity_env_n("pseudo_viscosity_env_n", nbCells),
        m_pseudo_viscosity_env_nplus1("pseudo_viscosity_env_nplus1", nbCells),
        m_tau_density_n("tau_density_n", nbCells),
        m_tau_density_nplus1("tau_density_nplus1", nbCells),
        m_tau_density_env_n("tau_density_env_n", nbCells),
        m_tau_density_env_nplus1("tau_density_env_nplus1", nbCells),
        m_tau_volume_n("tau_volume_n", nbCells),
        m_tau_volume_nplus1("tau_volume_nplus1", nbCells),
        m_tau_volume_env_n("tau_volume_env_n", nbCells),
        m_tau_volume_env_nplus1("tau_volume_env_nplus1", nbCells),
        m_divu_n("divu_n", nbCells),
        m_divu_nplus1("divu_nplus1", nbCells),
        m_speed_velocity_n("speed_velocity_n", nbCells),
        m_speed_velocity_nplus1("speed_velocity_nplus1", nbCells),
        m_speed_velocity_env_n("speed_velocity_n", nbCells),
        m_speed_velocity_env_nplus1("speed_velocity_nplus1", nbCells),
        m_internal_energy_n("internal_energy_n", nbCells),
        m_internal_energy_nplus1("internal_energy_nplus1", nbCells),
        m_internal_energy_env_n("internal_energy_env_n", nbCells),
        m_internal_energy_env_nplus1("internal_energy_env_nplus1", nbCells),
        m_node_velocity_n("node_velocity_n", nbNodes),
        m_node_velocity_nplus1("node_velocity_nplus1", nbNodes),
        m_x_velocity("x_velocity", nbNodes),
        m_y_velocity("y_velocity", nbNodes),
        m_node_mass("node_mass", nbNodes),
        m_cell_mass("cell_mass", nbCells),
        m_cell_mass_env("cell_mass_env", nbCells),
        m_cell_coord_n("cell_coord_n", nbCells),
        m_cell_coord_nplus1("cell_coord_nplus1", nbCells),
        m_euler_volume("euler_volume", nbCells),
        m_lagrange_volume_n("lagrange_volume_n", nbCells),
        m_lagrange_volume_nplus1("lagrange_volume_nplus1", nbCells),
        m_mass_fraction_env("mass_fraction_env", nbCells),
        m_fracvol_env("fracvol_env", nbCells),
        m_node_fracvol("node_fracvol", nbNodes),
        m_fracvol_env1("fracvol_env1", nbCells),
        m_fracvol_env2("fracvol_env2", nbCells),
        m_fracvol_env3("fracvol_env3", nbCells),
        m_interface12("interface12", nbCells),
        m_interface23("interface23", nbCells),
        m_interface13("interface13", nbCells),
        m_cqs_n("cqs_n", nbCells, nbNodesOfCell),
        m_cqs_nplus1("cqs_nplus1", nbCells, nbNodesOfCell) {
    // Copy node coordinates
    const auto& gNodes = mesh->getGeometry()->getNodes();
    for (size_t rNodes = 0; rNodes < nbNodes; rNodes++) {
      m_node_coord_n(rNodes)[0] = gNodes[rNodes][0];
      m_node_coord_n(rNodes)[1] = gNodes[rNodes][1];
    }
  }

 private:
  KOKKOS_INLINE_FUNCTION
  void computeDeltaT() noexcept;
  void computeDeltaTinit() noexcept;
  void computeVariablesGlobalesInit() noexcept;
  void computeVariablesGlobalesL0() noexcept;
  void computeVariablesGlobalesL() noexcept;
  void computeVariablesGlobalesT() noexcept;
  void computeVariablesSortiesInit() noexcept;
  
  KOKKOS_INLINE_FUNCTION
  void computeTime() noexcept;

  KOKKOS_INLINE_FUNCTION
  void setUpTimeLoopN() noexcept;

  KOKKOS_INLINE_FUNCTION
  void executeTimeLoopN() noexcept;

  void dumpVariables() noexcept;

  // dans PhaseLagrange.cc

  void computeArtificialViscosity() noexcept;

  void computeCornerNormal() noexcept;
  void updateCornerNormal() noexcept;

  void updateVelocity() noexcept;
  void updateVelocitybackward() noexcept;
  void updateVelocityforward() noexcept;

  void updateVelocityWithoutLagrange() noexcept;
  
  void computeCellMass() noexcept;

  void updatePosition() noexcept;

  void updateCellPos() noexcept;

  void computeNodeMass() noexcept;

  void computeNodeVolume() noexcept;

  void computeSubVol() noexcept;

  void updateRho() noexcept;

  void computeTau() noexcept;

  void updateEnergy() noexcept;
  void updateEnergycqs() noexcept;
  void updateEnergyForTotalEnergyConservation() noexcept;
  
  void computeDivU() noexcept;

  void computeEOS();
  
  void computePressionMoyenne() noexcept;

  void updateNodeBoundaryConditions() noexcept;
  void updateCellBoundaryConditions() noexcept;
  void computeVariablesForRemap() noexcept;
  void computeFaceQuantitesForRemap() noexcept;
  void computeCellQuantitesForRemap() noexcept;
  void remapVariables() noexcept;

 public:
  void simulate();
};

#endif  // VNRREMAP_H

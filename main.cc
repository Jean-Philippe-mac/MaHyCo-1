#include <Kokkos_Core.hpp>  // for finalize, initialize
#include <cstdlib>          // for atof, atoi, exit
#include <iostream>         // for operator<<, endl, basic_o...
#include <string>           // for string
#include "Eucclhyd.h"  // for EucclhydRemap
#include "Vnr.h"
#include "lecture_donnees/LectureDonnees.h"  // for LectureDonnees
#include "mesh/CartesianMesh2D.h"            // for CartesianMesh2D
#include "mesh/CartesianMesh2DGenerator.h"   // for CartesianMesh2DGenerator

using namespace nablalib;

int main(int argc, char* argv[]) {
  // initialisation de Kokkos
  Kokkos::initialize(argc, argv);
  // class utilisables
  // schema numerique lagrange
  auto scheme = new schemalagrangelib::SchemaLagrangeClass::SchemaLagrange();
  // options des schemas 
  auto o = new optionschemalib::OptionsSchema::Options();
  // conditions aux limites
  auto cl = new conditionslimiteslib::ConditionsLimites::Cdl();
  // limiteurs de la projection
  auto lim = new limiteurslib::LimiteursClass::Limiteurs();
  // equations d'état
  auto eos = new eoslib::EquationDetat();
  // cas Test
  auto test = new castestlib::CasTest::Test();
  // paraemetres du maillage
  auto cstmesh =
      new cstmeshlib::ConstantesMaillagesClass::ConstantesMaillages();
  // gestion du temps et pas de temps de la simulation
  auto gt = new gesttempslib::GestionTempsClass::GestTemps();
  // choix des variables a sortir
  auto so = new sortielib::Sortie::SortieVariables();
  string output;
  // Lecture des donnees
  if (argc == 2) {
    LectureDonneesClass lecture;
    lecture.LectureDonnees(argv[1], scheme, o, so, cstmesh, gt, lim, eos, test);
  } else if (argc != 1) {
    std::cerr << "[ERREUR] Fichier de donnees non passé en argument "
              << std::endl;
    exit(1);
  }
  // chargement du maillage
  auto nm = CartesianMesh2DGenerator::generate(
      cstmesh->X_EDGE_ELEMS, cstmesh->Y_EDGE_ELEMS, cstmesh->X_EDGE_LENGTH,
      cstmesh->Y_EDGE_LENGTH, cstmesh->cylindrical_mesh, cstmesh->minimum_radius);

  // appel au schéma Lagrange Eucclhyd + schéma de projection ADI (en option)
  if (scheme->schema == scheme->Eucclhyd) {
    // variables de projection 
    auto varlp = new variableslagremaplib::VariablesLagRemap(nm);
    // variables et fonctions de l'initialisation
    auto init = new initlib::Initialisations(o, eos, nm, cstmesh, varlp, cl, test);
    // variables et fonctions du schema particulaires
    auto part = new particleslib::SchemaParticules(nm, cstmesh, gt, test);
    // fonctions de la projection 
    auto proj = new Remap(o, cstmesh, gt, cl, lim, nm, varlp);
    //
    auto c =
      new Eucclhyd(o, cstmesh, gt, test, cl, lim, part, eos, nm, varlp, proj, init, so, output);
    c->simulate();
    delete varlp;
    delete init;
    delete part;
  
  } else if (scheme->schema == scheme->VNR || scheme->schema == scheme->CSTS) {
    // variables de projection 
    auto varlp = new variableslagremaplib::VariablesLagRemap(nm);
    // variables et fonctions de l'initialisation
    auto init = new initlib::Initialisations(o, eos, nm, cstmesh, varlp, cl, test);
    // variables et fonctions du schema particulaires
    auto part = new particleslib::SchemaParticules(nm, cstmesh, gt, test);
    // fonctions de la projection 
    auto proj = new Remap(o, cstmesh, gt, cl, lim, nm, varlp);
    // 
    auto c = new Vnr(scheme, o, cstmesh, gt, test, cl, lim, part, eos, nm, varlp, proj, init, so, output);
    c->simulate();
    delete varlp;
    delete init;
    delete part;
  }
  delete o;
  // delete nm;
  delete cl;
  delete lim;
  delete eos;
  delete test;
  delete cstmesh;
  delete gt;
  delete scheme;

  Kokkos::finalize();

  return 0;
}

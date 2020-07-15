#ifndef SCHEMAPARTICULES_H
#define SCHEMAPARTICULES_H


using namespace nablalib;

namespace particulelib
{

class SchemaParticules {
 public:
    struct Particules {
    int DragModel;
    int Kliatchko = 20;
    int Classique = 21;
    int KliatchkoDragModel = 20;

    double Reynolds_min = 1.e-4;
    double Reynolds_max = 1.e3;
    double Drag = 10.;
  };
  Particules* particules;

 private:
  CartesianMesh2D* mesh;

};
} // namespace
#endif  // SCHEMAPARTICULES_H

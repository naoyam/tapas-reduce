#include "args.h"
#include "bound_box.h"
#ifdef CILK
#include "build_tree3.h"
#else
#include "build_tree.h"
#endif
#include "dataset.h"
#include "logger.h"
#include "traversal.h"
#include "up_down_pass.h"
#include "verify.h"
#if VTK
#include "vtk.h"
#endif

int main(int argc, char ** argv) {
  Args args(argc, argv);
  Bodies bodies, bodies2, bodies3, jbodies;
  BoundBox boundBox(args.nspawn);
  Bounds bounds;
  BuildTree buildTree(args.ncrit, args.nspawn);
  Cells cells, jcells;
  Dataset data;
  Traversal traversal(args.nspawn, args.images);
  UpDownPass upDownPass(args.theta, args.useRmax, args.useRopt);
  Verify verify;
  num_threads(args.threads);

  const real_t cycle = 2 * M_PI;
  logger::verbose = args.verbose;
  logger::printTitle("FMM Parameters");
  args.print(logger::stringLength, P);
  bodies = data.initBodies(args.numBodies, args.distribution, 0);

  for (int t=0; t<args.repeat; t++) {
    logger::printTitle("FMM Profiling");
    logger::startTimer("Total FMM");
    logger::startPAPI();
    logger::startDAG();
    bounds = boundBox.getBounds(bodies);
    std::cerr << "Bounding box: " << bounds.Xmin << "->" << bounds.Xmax << std::endl;

    cells = buildTree.buildTree(bodies, bounds);
    upDownPass.upwardPass(cells);

    traversal.dualTreeTraversal(cells, cells, cycle, args.mutual);
    jbodies = bodies;
#if 1
    {
      std::ofstream tapas_out("ref_M2L.txt");
      for (int i = 0; i < args.numBodies; ++i) {
        tapas_out << bodies[i].TRG << std::endl;
      }
    }
#endif

    upDownPass.downwardPass(cells);
    logger::printTitle("Total runtime");
    logger::stopPAPI();
    logger::stopTimer("Total FMM");
    logger::resetTimer("Total FMM");
#if WRITE_TIME
    logger::writeTime();
#endif
    const int numTargets = 100;
    bodies3 = bodies;
    data.sampleBodies(bodies, numTargets);
    bodies2 = bodies;
    data.initTarget(bodies);
    logger::startTimer("Total Direct");
    traversal.direct(bodies, jbodies, cycle);
    traversal.normalize(bodies);
    logger::stopTimer("Total Direct");
    double potDif = verify.getDifScalar(bodies, bodies2);
    double potNrm = verify.getNrmScalar(bodies);
    double accDif = verify.getDifVector(bodies, bodies2);
    double accNrm = verify.getNrmVector(bodies);
    logger::printTitle("FMM vs. direct");
    verify.print("Rel. L2 Error (pot)",std::sqrt(potDif/potNrm));
    verify.print("Rel. L2 Error (acc)",std::sqrt(accDif/accNrm));
    buildTree.printTreeData(cells);
    traversal.printTraversalData();
    logger::printPAPI();
    logger::stopDAG();
    bodies = bodies3;
    data.initTarget(bodies);
  }
  logger::writeDAG();
#if VTK
  for (B_iter B=jbodies.begin(); B!=jbodies.end(); B++) B->IBODY = 0;
  for (C_iter C=jcells.begin(); C!=jcells.end(); C++) {
    Body body;
    body.IBODY = 1;
    body.X     = C->X;
    body.SRC   = 0;
    jbodies.push_back(body);
  }
  vtk3DPlot vtk;
  vtk.setBounds(M_PI,0);
  vtk.setGroupOfPoints(jbodies);
  vtk.plot();
#endif
  return 0;
}

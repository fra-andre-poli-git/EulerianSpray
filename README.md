# EulerianSpray
This is the repository of my APSC/PACS project on Eulerian Sprays.

The license is the same as that of the library, as evinced in deal.II README:
https://www.dealii.org/9.5.0/readme.html#license

The code is developed under version 9.5.1 installed from apt-get:
https://github.com/dealii/dealii/wiki/Debian-and-Ubuntu

But then CMakeLists.txt was modified to work also (and preferentially) with the 
installation of deal.ii in the mk modules:
https://github.com/pcafrica/mk

The structure of the classes is taken from tutorial 67 of deal.II library, which
solves Euler equations. The commented code is found at:
https://www.dealii.org/current/doxygen/deal.II/step_67.html

Most of the parameters are set at run time via input file, that can be found in
./input directory. The physical dimension of the problem is set at compile time,
in the main.cpp file, as well as the polynomial degree.

To run the code you need to follow the steps below.

```bash
mkdir build

cd build

cmake ..

make release

make

mkdir results

./main ../input/INPUTFILENAME.prm
```
where INPUTFILENAME.prm is the desired input file.

A guide to INPUTFILENAME:
 - Test 1: Accuracy1d.prm, not implemented
 - Test 2: DeltaShock1d.prm, a $\delta$-shock wave travelling
 - Test 3: Vacuum1d.prm, creation of vacuum in one dimension
 - Test 4: VacuumCloseUp1d.prm, a close-up on the vacuum of the previous case
 - Test 5: not implemented
 - Test 6: OriginRadialDelta.prm, $\delta$ in the origin with radial velocity
 - Test 7: OriginCrossinDelta.prm, $\delta$ along the axis crossing in the origin
 - Test 8: AnularDelta.prm, $\delta$ in a circle
 - Test 9: CrossVacuum2d.prm, vacuum along the axis

I underline the switch to release mode, since otherwise 2d simulation would be
unfeasible. To interrogate the build type:
```bash
grep CMAKE_BUILD_TYPE CMakeCache.txt
```

In directory myresults/ I put the computations I performed. The way I store them
is:
```bash
.
├── Test1
│   ├── Degree0
│   ├── Degree1
│   └── Degree2
└── Test2
│   ├── Degree0
│   ├── Degree1
│   └── Degree2
...
```
and each DegreeN/ directory will contain some directories: inside will be found
the output files of the code (the files generated in build/results) and the
input used to produce it.
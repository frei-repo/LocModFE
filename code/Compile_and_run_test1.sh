mkdir build
cd build

# Create Makefile and compile LocModFE
cmake -DDEAL_II_DIR=/dealii-8.5.0/bin -DCMAKE_CXX_FLAGS=-Wno-deprecated-declarations ..
make -j4

#Execute Test 1
cd ../../data
../code/build/step-modfe
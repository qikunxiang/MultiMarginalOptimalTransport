# C++ compiler
CXX = clang++

# C++ flags
CXX_FLAGS = --std=c++20 -mmacosx-version-min=11.0 -DNDEBUG -O3

# Suffix of mex files on this operating system
# macOS with Apple Silicon
MEX_SUFFIX = mexmaca64

# macOS with Intel
#MEX_SUFFIX = mexmaci64

# Path to MATLAB
MATLAB_PATH = /Applications/MATLAB_R2024a.app

# Path to the source directory
SOURCE_PATH = ./src

# Path to the build directory
BUILD_PATH = ./build

# Include path (for header files)
INCLUDE_PATH = /usr/local/include

# Library path (for linking to libraries)
LIBRARY_PATH = /usr/local/lib

# Default target
all: power_diagram_intersection.$(MEX_SUFFIX) mesh_intersect_power_diagram.$(MEX_SUFFIX)
.PHONY: all

# For executing mex and create the binary file for the MATLAB function power_diagram_intersection
power_diagram_intersection.$(MEX_SUFFIX): $(SOURCE_PATH)/mex_wrapper_power_diagram_intersection.cpp $(BUILD_PATH)/kdtree.cpp.o $(BUILD_PATH)/power_diagram_intersection.cpp.o
	$(MATLAB_PATH)/bin/mex 'CXXFLAGS=$$CXXFLAGS $(CXX_FLAGS)' $(SOURCE_PATH)/mex_wrapper_power_diagram_intersection.cpp $(BUILD_PATH)/power_diagram_intersection.cpp.o $(BUILD_PATH)/kdtree.cpp.o -I$(MATLAB_PATH)/extern/include -I$(INCLUDE_PATH) -I$(SOURCE_PATH) -L$(LIBRARY_PATH) -lmpfr -lgmp -output power_diagram_intersection

# For executing mex and create the binary file for the MATLAB function mesh_intersect_power_diagram
mesh_intersect_power_diagram.$(MEX_SUFFIX): $(SOURCE_PATH)/mex_wrapper_mesh_intersect_power_diagram.cpp $(BUILD_PATH)/kdtree.cpp.o $(BUILD_PATH)/power_diagram_intersection.cpp.o
	$(MATLAB_PATH)/bin/mex 'CXXFLAGS=$$CXXFLAGS $(CXX_FLAGS)' $(SOURCE_PATH)/mex_wrapper_mesh_intersect_power_diagram.cpp $(BUILD_PATH)/power_diagram_intersection.cpp.o $(BUILD_PATH)/kdtree.cpp.o -I$(MATLAB_PATH)/extern/include -I$(INCLUDE_PATH) -I$(SOURCE_PATH) -L$(LIBRARY_PATH) -lmpfr -lgmp -output mesh_intersect_power_diagram

# For compiling the KDTree functions
$(BUILD_PATH)/kdtree.cpp.o: $(SOURCE_PATH)/kdtree-cpp/kdtree.cpp $(SOURCE_PATH)/kdtree-cpp/kdtree.hpp
	$(CXX) $(CXX_FLAGS) -c $(SOURCE_PATH)/kdtree-cpp/kdtree.cpp -o $(BUILD_PATH)/kdtree.cpp.o -I$(INCLUDE_PATH) -I$(SOURCE_PATH)

# For compiling the CGAL-related functions
$(BUILD_PATH)/power_diagram_intersection.cpp.o: $(SOURCE_PATH)/power_diagram_intersection.cpp $(SOURCE_PATH)/power_diagram_intersection.hpp
	$(CXX) $(CXX_FLAGS) -c $(SOURCE_PATH)/power_diagram_intersection.cpp -o $(BUILD_PATH)/power_diagram_intersection.cpp.o -I$(INCLUDE_PATH) -I$(SOURCE_PATH)


# For compiling both tests
test: test_intersection test_mesh
.PHONY: test

# For compiling the test script test_power_diagram_intersection
test_intersection: $(BUILD_PATH)/test_power_diagram_intersection
.PHONY: test_intersection

$(BUILD_PATH)/test_power_diagram_intersection: $(SOURCE_PATH)/test_power_diagram_intersection.cpp $(BUILD_PATH)/kdtree.cpp.o $(BUILD_PATH)/power_diagram_intersection.cpp.o
	$(CXX) $(CXX_FLAGS) $(SOURCE_PATH)/test_power_diagram_intersection.cpp $(BUILD_PATH)/kdtree.cpp.o $(BUILD_PATH)/power_diagram_intersection.cpp.o -o $(BUILD_PATH)/test_power_diagram_intersection -I$(INCLUDE_PATH) -I$(SOURCE_PATH) -L$(LIBRARY_PATH) -lmpfr -lgmp

# For compiling the test script test_power_diagram_intersection_with_mesh
test_mesh: $(BUILD_PATH)/test_mesh_intersect_power_diagram
.PHONY: test_mesh

$(BUILD_PATH)/test_mesh_intersect_power_diagram: $(SOURCE_PATH)/test_mesh_intersect_power_diagram.cpp $(BUILD_PATH)/kdtree.cpp.o $(BUILD_PATH)/power_diagram_intersection.cpp.o
	$(CXX) $(CXX_FLAGS) $(SOURCE_PATH)/test_mesh_intersect_power_diagram.cpp $(BUILD_PATH)/kdtree.cpp.o $(BUILD_PATH)/power_diagram_intersection.cpp.o -o $(BUILD_PATH)/test_mesh_intersect_power_diagram -I$(INCLUDE_PATH) -I$(SOURCE_PATH) -L$(LIBRARY_PATH) -lmpfr -lgmp

# For cleaning the build directory
clean:
	rm -rf $(BUILD_PATH)/*
	rm -f *.$(MEX_SUFFIX)
.PHONY: clean
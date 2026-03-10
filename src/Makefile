include Make_linux.inc
#include Make_msys2.inc
#include Make_osx.inc

CXXFLAGS = -std=c++17
ifdef DEBUG
CXXFLAGS += -g -O0 -Wall -fbounds-check -pedantic -D_GLIBCXX_DEBUG
CXXFLAGS2 = CXXFLAGS
else
CXXFLAGS2 = ${CXXFLAGS} -O2 -march=native -Wall 
CXXFLAGS += -O3 -march=native -Wall
endif

ALL= ant_simu.exe ant_simu_soa.exe ant_simu_omp.exe ant_simu_mpi.exe ant_simu_mpi2.exe

default:	help

all: $(ALL)

clean:
	@rm -fr *.o *.exe *~

.cpp.o:
	$(CXX) $(CXXFLAGS2) -c $^ -o $@	

ant_simu.exe : ant.o fractal_land.o renderer.o window.o ant_simu.o
	$(CXX) $(CXXFLAGS2) $^ -o $@ $(LIB)	

# === Version SoA (vectorisée) ===
renderer_soa.o: renderer.cpp
	$(CXX) $(CXXFLAGS2) -DUSE_SOA -c $< -o $@

ant_simu_soa.o: ant_simu_soa.cpp
	$(CXX) $(CXXFLAGS2) -DUSE_SOA -c $< -o $@

ant_simu_soa.exe : ant_simu_soa.o fractal_land.o renderer_soa.o window.o
	$(CXX) $(CXXFLAGS2) $^ -o $@ $(LIB)

# === Version OpenMP (parallélisée) ===
ant_simu_omp.o: ant_simu_omp.cpp
	$(CXX) $(CXXFLAGS2) -DUSE_SOA -c $< -o $@

ant_simu_omp.exe : ant_simu_omp.o fractal_land.o renderer_soa.o window.o
	$(CXX) $(CXXFLAGS2) $^ -o $@ $(LIB)

# === Version MPI (Phase 4A : full map replication) ===
ant_simu_mpi.o: ant_simu_mpi.cpp
	$(MPICXX) $(CXXFLAGS2) -DUSE_SOA -c $< -o $@

ant_simu_mpi.exe : ant_simu_mpi.o fractal_land.o renderer_soa.o window.o
	$(MPICXX) $(CXXFLAGS2) $^ -o $@ $(LIB)

# === Version MPI method 2 (Phase 4B : map partitioning) ===
ant_simu_mpi2.o: ant_simu_mpi2.cpp
	$(MPICXX) $(CXXFLAGS2) -DUSE_SOA -c $< -o $@

ant_simu_mpi2.exe : ant_simu_mpi2.o fractal_land.o
	$(MPICXX) $(CXXFLAGS2) $^ -o $@ $(LIB)

help:
	@echo "Available targets : "
	@echo "    all            : compile all executables"
	@echo "Add DEBUG=yes to compile in debug"
	@echo "Configuration :"
	@echo "    CXX      :    $(CXX)"
	@echo "    CXXFLAGS :    $(CXXFLAGS)"

%.html: %.md
	pandoc -s --toc $< --css=./github-pandoc.css --metadata pagetitle="OS202 - TD1" -o $@

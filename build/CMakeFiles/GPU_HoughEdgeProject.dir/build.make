# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.22

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /user/2/.base/gennusoa/home/CGPU-Project/GPU_HoughEdgeProject

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /user/2/.base/gennusoa/home/CGPU-Project/GPU_HoughEdgeProject/build

# Include any dependencies generated for this target.
include CMakeFiles/GPU_HoughEdgeProject.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/GPU_HoughEdgeProject.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/GPU_HoughEdgeProject.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/GPU_HoughEdgeProject.dir/flags.make

CMakeFiles/GPU_HoughEdgeProject.dir/main.cpp.o: CMakeFiles/GPU_HoughEdgeProject.dir/flags.make
CMakeFiles/GPU_HoughEdgeProject.dir/main.cpp.o: ../main.cpp
CMakeFiles/GPU_HoughEdgeProject.dir/main.cpp.o: CMakeFiles/GPU_HoughEdgeProject.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/user/2/.base/gennusoa/home/CGPU-Project/GPU_HoughEdgeProject/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/GPU_HoughEdgeProject.dir/main.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/GPU_HoughEdgeProject.dir/main.cpp.o -MF CMakeFiles/GPU_HoughEdgeProject.dir/main.cpp.o.d -o CMakeFiles/GPU_HoughEdgeProject.dir/main.cpp.o -c /user/2/.base/gennusoa/home/CGPU-Project/GPU_HoughEdgeProject/main.cpp

CMakeFiles/GPU_HoughEdgeProject.dir/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/GPU_HoughEdgeProject.dir/main.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /user/2/.base/gennusoa/home/CGPU-Project/GPU_HoughEdgeProject/main.cpp > CMakeFiles/GPU_HoughEdgeProject.dir/main.cpp.i

CMakeFiles/GPU_HoughEdgeProject.dir/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/GPU_HoughEdgeProject.dir/main.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /user/2/.base/gennusoa/home/CGPU-Project/GPU_HoughEdgeProject/main.cpp -o CMakeFiles/GPU_HoughEdgeProject.dir/main.cpp.s

# Object files for target GPU_HoughEdgeProject
GPU_HoughEdgeProject_OBJECTS = \
"CMakeFiles/GPU_HoughEdgeProject.dir/main.cpp.o"

# External object files for target GPU_HoughEdgeProject
GPU_HoughEdgeProject_EXTERNAL_OBJECTS =

../bin/GPU_HoughEdgeProject: CMakeFiles/GPU_HoughEdgeProject.dir/main.cpp.o
../bin/GPU_HoughEdgeProject: CMakeFiles/GPU_HoughEdgeProject.dir/build.make
../bin/GPU_HoughEdgeProject: CMakeFiles/GPU_HoughEdgeProject.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/user/2/.base/gennusoa/home/CGPU-Project/GPU_HoughEdgeProject/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable ../bin/GPU_HoughEdgeProject"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/GPU_HoughEdgeProject.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/GPU_HoughEdgeProject.dir/build: ../bin/GPU_HoughEdgeProject
.PHONY : CMakeFiles/GPU_HoughEdgeProject.dir/build

CMakeFiles/GPU_HoughEdgeProject.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/GPU_HoughEdgeProject.dir/cmake_clean.cmake
.PHONY : CMakeFiles/GPU_HoughEdgeProject.dir/clean

CMakeFiles/GPU_HoughEdgeProject.dir/depend:
	cd /user/2/.base/gennusoa/home/CGPU-Project/GPU_HoughEdgeProject/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /user/2/.base/gennusoa/home/CGPU-Project/GPU_HoughEdgeProject /user/2/.base/gennusoa/home/CGPU-Project/GPU_HoughEdgeProject /user/2/.base/gennusoa/home/CGPU-Project/GPU_HoughEdgeProject/build /user/2/.base/gennusoa/home/CGPU-Project/GPU_HoughEdgeProject/build /user/2/.base/gennusoa/home/CGPU-Project/GPU_HoughEdgeProject/build/CMakeFiles/GPU_HoughEdgeProject.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/GPU_HoughEdgeProject.dir/depend


# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.26

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
CMAKE_COMMAND = /home/ckengjwe/.local/lib/python3.8/site-packages/cmake/data/bin/cmake

# The command to remove a file.
RM = /home/ckengjwe/.local/lib/python3.8/site-packages/cmake/data/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/ckengjwe/dso/symforce/symforce/examples

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/ckengjwe/dso/symforce/symforce/examples/robot_3d_localization

# Include any dependencies generated for this target.
include CMakeFiles/symforce_examples.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/symforce_examples.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/symforce_examples.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/symforce_examples.dir/flags.make

CMakeFiles/symforce_examples.dir/gen/measurements.o: CMakeFiles/symforce_examples.dir/flags.make
CMakeFiles/symforce_examples.dir/gen/measurements.o: gen/measurements.cc
CMakeFiles/symforce_examples.dir/gen/measurements.o: CMakeFiles/symforce_examples.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/ckengjwe/dso/symforce/symforce/examples/robot_3d_localization/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/symforce_examples.dir/gen/measurements.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/symforce_examples.dir/gen/measurements.o -MF CMakeFiles/symforce_examples.dir/gen/measurements.o.d -o CMakeFiles/symforce_examples.dir/gen/measurements.o -c /home/ckengjwe/dso/symforce/symforce/examples/robot_3d_localization/gen/measurements.cc

CMakeFiles/symforce_examples.dir/gen/measurements.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/symforce_examples.dir/gen/measurements.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/ckengjwe/dso/symforce/symforce/examples/robot_3d_localization/gen/measurements.cc > CMakeFiles/symforce_examples.dir/gen/measurements.i

CMakeFiles/symforce_examples.dir/gen/measurements.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/symforce_examples.dir/gen/measurements.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/ckengjwe/dso/symforce/symforce/examples/robot_3d_localization/gen/measurements.cc -o CMakeFiles/symforce_examples.dir/gen/measurements.s

CMakeFiles/symforce_examples.dir/run_dynamic_size.o: CMakeFiles/symforce_examples.dir/flags.make
CMakeFiles/symforce_examples.dir/run_dynamic_size.o: run_dynamic_size.cc
CMakeFiles/symforce_examples.dir/run_dynamic_size.o: CMakeFiles/symforce_examples.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/ckengjwe/dso/symforce/symforce/examples/robot_3d_localization/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/symforce_examples.dir/run_dynamic_size.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/symforce_examples.dir/run_dynamic_size.o -MF CMakeFiles/symforce_examples.dir/run_dynamic_size.o.d -o CMakeFiles/symforce_examples.dir/run_dynamic_size.o -c /home/ckengjwe/dso/symforce/symforce/examples/robot_3d_localization/run_dynamic_size.cc

CMakeFiles/symforce_examples.dir/run_dynamic_size.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/symforce_examples.dir/run_dynamic_size.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/ckengjwe/dso/symforce/symforce/examples/robot_3d_localization/run_dynamic_size.cc > CMakeFiles/symforce_examples.dir/run_dynamic_size.i

CMakeFiles/symforce_examples.dir/run_dynamic_size.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/symforce_examples.dir/run_dynamic_size.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/ckengjwe/dso/symforce/symforce/examples/robot_3d_localization/run_dynamic_size.cc -o CMakeFiles/symforce_examples.dir/run_dynamic_size.s

CMakeFiles/symforce_examples.dir/run_fixed_size.o: CMakeFiles/symforce_examples.dir/flags.make
CMakeFiles/symforce_examples.dir/run_fixed_size.o: run_fixed_size.cc
CMakeFiles/symforce_examples.dir/run_fixed_size.o: CMakeFiles/symforce_examples.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/ckengjwe/dso/symforce/symforce/examples/robot_3d_localization/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object CMakeFiles/symforce_examples.dir/run_fixed_size.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/symforce_examples.dir/run_fixed_size.o -MF CMakeFiles/symforce_examples.dir/run_fixed_size.o.d -o CMakeFiles/symforce_examples.dir/run_fixed_size.o -c /home/ckengjwe/dso/symforce/symforce/examples/robot_3d_localization/run_fixed_size.cc

CMakeFiles/symforce_examples.dir/run_fixed_size.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/symforce_examples.dir/run_fixed_size.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/ckengjwe/dso/symforce/symforce/examples/robot_3d_localization/run_fixed_size.cc > CMakeFiles/symforce_examples.dir/run_fixed_size.i

CMakeFiles/symforce_examples.dir/run_fixed_size.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/symforce_examples.dir/run_fixed_size.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/ckengjwe/dso/symforce/symforce/examples/robot_3d_localization/run_fixed_size.cc -o CMakeFiles/symforce_examples.dir/run_fixed_size.s

# Object files for target symforce_examples
symforce_examples_OBJECTS = \
"CMakeFiles/symforce_examples.dir/gen/measurements.o" \
"CMakeFiles/symforce_examples.dir/run_dynamic_size.o" \
"CMakeFiles/symforce_examples.dir/run_fixed_size.o"

# External object files for target symforce_examples
symforce_examples_EXTERNAL_OBJECTS =

libsymforce_examples.a: CMakeFiles/symforce_examples.dir/gen/measurements.o
libsymforce_examples.a: CMakeFiles/symforce_examples.dir/run_dynamic_size.o
libsymforce_examples.a: CMakeFiles/symforce_examples.dir/run_fixed_size.o
libsymforce_examples.a: CMakeFiles/symforce_examples.dir/build.make
libsymforce_examples.a: CMakeFiles/symforce_examples.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/ckengjwe/dso/symforce/symforce/examples/robot_3d_localization/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Linking CXX static library libsymforce_examples.a"
	$(CMAKE_COMMAND) -P CMakeFiles/symforce_examples.dir/cmake_clean_target.cmake
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/symforce_examples.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/symforce_examples.dir/build: libsymforce_examples.a
.PHONY : CMakeFiles/symforce_examples.dir/build

CMakeFiles/symforce_examples.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/symforce_examples.dir/cmake_clean.cmake
.PHONY : CMakeFiles/symforce_examples.dir/clean

CMakeFiles/symforce_examples.dir/depend:
	cd /home/ckengjwe/dso/symforce/symforce/examples/robot_3d_localization && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/ckengjwe/dso/symforce/symforce/examples /home/ckengjwe/dso/symforce/symforce/examples /home/ckengjwe/dso/symforce/symforce/examples/robot_3d_localization /home/ckengjwe/dso/symforce/symforce/examples/robot_3d_localization /home/ckengjwe/dso/symforce/symforce/examples/robot_3d_localization/CMakeFiles/symforce_examples.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/symforce_examples.dir/depend
